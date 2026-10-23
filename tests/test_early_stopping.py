#!/usr/bin/env python3
"""Real-training regressions; Python standard library only, no downloaded data."""

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


TRAINER = None
PREDICTOR = None
API = None


class EarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="libffm-early-stop-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.counter = 0
        # Two fields create one interaction; positive training increases its
        # margin. Opposite labels then worsen validation on every epoch.
        self.train = self.fixture("train.ffm", "+1 0:0:1 1:1:1\n" * 16)
        self.worsening = self.fixture("worsening.ffm", "-1 0:0:1 1:1:1\n" * 4)
        self.improving = self.fixture("improving.ffm", "+1 0:0:1 1:1:1\n" * 4)
        # A single feature cannot interact: logloss is always log(2), while
        # training still changes the two-feature model on every epoch.
        self.plateau = self.fixture("plateau.ffm", "-1 0:0:1\n" * 4)
        self.empty = self.fixture("empty.ffm", "")
        self.nonfinite = self.fixture("nonfinite.ffm", "-1 0:0:nan 1:1:1\n")

    def fixture(self, name, contents):
        path = self.directory / name
        path.write_text(contents, encoding="ascii")
        return path

    def execute(self, arguments):
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        return subprocess.run(
            [str(argument) for argument in arguments],
            cwd=self.directory,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    def successful(self, arguments):
        result = self.execute(arguments)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def train_model(self, validation, iterations, auto_stop=False, patience=None):
        self.counter += 1
        model = self.directory / ("model-%d.bin" % self.counter)
        arguments = [TRAINER, "-s", "1", "-t", iterations, "-p", validation]
        if auto_stop:
            arguments.append("--auto-stop")
        if patience is not None:
            arguments.extend(["-b", patience])
        arguments.extend([self.train, model])
        result = self.successful(arguments)
        # A stop-triggering epoch is printed before its timing column, so
        # accept rows with just epoch, training loss, and validation loss.
        rows = []
        for line in result.stdout.splitlines():
            match = re.match(r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)(?:\s|$)", line)
            if match:
                rows.append((int(match[1]), float(match[2]), float(match[3])))
        self.assertTrue(rows, result.stdout)
        self.assertTrue(model.is_file(), result.stdout + result.stderr)
        return model, rows, result.stdout

    def predictions(self, model):
        output = self.directory / (model.name + ".predictions")
        self.successful([PREDICTOR, self.train, model, output])
        values = tuple(float(line) for line in output.read_text().splitlines())
        self.assertEqual(len(values), 16)
        self.assertTrue(all(math.isfinite(value) for value in values))
        return values

    def same_checkpoint(self, actual, expected):
        # Full bytes include optimizer accumulators, in addition to inference
        # weights. Predictions also prove that the saved model can be loaded.
        self.assertEqual(actual.read_bytes(), expected.read_bytes())
        self.assertEqual(self.predictions(actual), self.predictions(expected))

    def test_plateau_exhausts_patience_and_retains_earliest_checkpoint(self):
        for patience in (1, 2, 3):
            with self.subTest(patience=patience):
                best, _, _ = self.train_model(self.plateau, 1)
                later, _, _ = self.train_model(self.plateau, 1 + patience)
                self.assertNotEqual(best.read_bytes(), later.read_bytes())
                self.assertNotEqual(self.predictions(best), self.predictions(later))
                actual, rows, log = self.train_model(
                    self.plateau, 8, auto_stop=True, patience=patience)
                self.assertEqual([row[0] for row in rows], list(range(1, patience + 2)), log)
                self.assertTrue(all(row[2] == rows[0][2] for row in rows), log)
                self.same_checkpoint(actual, best)

    def test_worsening_exhausts_patience_and_restores_first_checkpoint(self):
        for patience in (1, 2, 3):
            with self.subTest(patience=patience):
                best, _, _ = self.train_model(self.worsening, 1)
                actual, rows, log = self.train_model(
                    self.worsening, 8, auto_stop=True, patience=patience)
                self.assertEqual([row[0] for row in rows], list(range(1, patience + 2)), log)
                self.assertTrue(all(left[2] < right[2] for left, right in zip(rows, rows[1:])), log)
                self.same_checkpoint(actual, best)

    def test_iteration_cap_restores_best_before_patience_expires(self):
        for validation in (self.worsening, self.plateau):
            with self.subTest(validation=validation.name):
                best, _, _ = self.train_model(validation, 1)
                actual, rows, log = self.train_model(
                    validation, 3, auto_stop=True, patience=5)
                self.assertEqual([row[0] for row in rows], [1, 2, 3], log)
                self.same_checkpoint(actual, best)

    def test_improving_validation_runs_to_cap_and_returns_last_checkpoint(self):
        final, _, _ = self.train_model(self.improving, 6)
        actual, rows, log = self.train_model(self.improving, 6, auto_stop=True, patience=2)
        self.assertEqual([row[0] for row in rows], list(range(1, 7)), log)
        self.assertTrue(all(left[2] > right[2] for left, right in zip(rows, rows[1:])), log)
        self.same_checkpoint(actual, final)

    def test_disabled_stopping_returns_last_model_despite_worsening_validation(self):
        first, _, _ = self.train_model(self.worsening, 1)
        final, rows, log = self.train_model(self.worsening, 5, patience=1)
        self.assertEqual([row[0] for row in rows], list(range(1, 6)), log)
        self.assertNotEqual(final.read_bytes(), first.read_bytes())
        self.assertTrue(all(a > b for a, b in zip(self.predictions(final), self.predictions(first))))

    def test_default_patience_remains_one(self):
        best, _, _ = self.train_model(self.worsening, 1)
        actual, rows, log = self.train_model(self.worsening, 8, auto_stop=True)
        self.assertEqual([row[0] for row in rows], [1, 2], log)
        self.same_checkpoint(actual, best)

    def test_help_documents_existing_patience_default(self):
        result = self.execute([TRAINER])
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stdout + result.stderr, r"-b <patience>:[^\n]*default 1")

    def test_cli_patience_rejects_invalid_complete_tokens(self):
        for token in ("2junk", "2.5", "2147483648", "999999999999999999999999", "0", "-1"):
            with self.subTest(token=token):
                output = self.directory / ("invalid-" + token + ".model")
                result = self.execute([
                    TRAINER, "-b", token, "--auto-stop", "-p", self.worsening,
                    self.train, output,
                ])
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(result.stderr.strip(), result.stdout)
                self.assertFalse(output.exists())

    def test_cli_patience_accepts_largest_integer(self):
        best, _, _ = self.train_model(self.worsening, 1)
        actual, rows, _ = self.train_model(
            self.worsening, 2, auto_stop=True, patience=2147483647)
        self.assertEqual([row[0] for row in rows], [1, 2])
        self.same_checkpoint(actual, best)

    def test_cli_missing_patience_value_fails_with_error(self):
        result = self.execute([TRAINER, "-b"])
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.strip(), result.stdout)

    def test_cli_missing_validation_fails_with_error(self):
        output = self.directory / "missing-validation.model"
        result = self.execute([TRAINER, "--auto-stop", self.train, output])
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.strip(), result.stdout)
        self.assertFalse(output.exists())

    def test_cli_empty_and_nonfinite_validation_fail_with_error(self):
        for validation in (self.empty, self.nonfinite):
            with self.subTest(validation=validation.name):
                output = self.directory / (validation.name + ".model")
                result = self.execute([
                    TRAINER, "--auto-stop", "-p", validation, self.train, output,
                ])
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(result.stderr.strip(), result.stdout)
                self.assertFalse(output.exists())

    def test_api_rejects_invalid_early_stopping_inputs(self):
        cases = (
            ("missing-validation", self.worsening),
            ("empty-validation", self.empty),
            ("patience-zero", self.worsening),
            ("patience-negative", self.worsening),
            ("iterations-zero", self.worsening),
            ("iterations-negative", self.worsening),
            ("nonfinite-validation", self.nonfinite),
        )
        for mode, validation in cases:
            with self.subTest(mode=mode):
                result = self.successful([
                    API, mode, self.train, validation,
                    self.directory / (mode + ".train.bin"),
                    self.directory / (mode + ".valid.bin"),
                ])
                self.assertIn("REJECTED:", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", required=True, type=lambda path: str(Path(path).resolve()))
    parser.add_argument("--predictor", required=True, type=lambda path: str(Path(path).resolve()))
    parser.add_argument("--api", required=True, type=lambda path: str(Path(path).resolve()))
    options, remaining = parser.parse_known_args()
    TRAINER, PREDICTOR, API = options.trainer, options.predictor, options.api
    unittest.main(argv=[__file__] + remaining, verbosity=2)
