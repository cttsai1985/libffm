# Minimal early-stopping repair

The owner authorized design, issue tracking, TDD, implementation, and closeout,
then explicitly narrowed libffm to a minimal change that makes early stopping
work. The broader modernization proposal is superseded by this focused scope.

## Goal and contracts

Preserve the existing solver, initialization/shuffle policy, C++11 build,
public function signatures, binary formats, and CLI. Keep `--auto-stop`, `-p`,
`-b`, and the existing default patience. Correct these early-stopping behaviors:

- The first finite validation loss establishes a checkpoint.
- Only a strictly lower finite loss replaces the checkpoint and resets patience.
- Ties and worse losses consume consecutive patience; stop at `>= patience`.
- Keep the earliest checkpoint on ties.
- Return the best checkpoint both when patience expires and when the iteration
  limit is reached. Stopping disabled continues to return the final model.
- Reject invalid patience or iteration limits for requested automatic stopping,
  missing/empty validation, and nonfinite monitored loss. Never restore an
  uninitialized/zero-filled checkpoint.
- Allocate checkpoint storage only when automatic stopping is enabled.
- Parse `-b` completely and surface early-stopping exceptions with a useful
  nonzero CLI failure. Other option parsing is outside this patch.

No new minimum-delta, seed, callback, model ownership, SIMD layout, general data
parser, CMake, or numerical-kernel feature is included.

## Source findings

`ffm.cpp` currently treats equal validation losses as improvements, and restores
the checkpoint only inside the patience-exhaustion branch. It allocates backup
weights even with stopping disabled. Library callers can silently lose requested
monitoring when validation is absent/empty. `ffm-train.cpp` uses `atoi` for `-b`.
These are source findings; actual RED results will be recorded in the issue/PR.

The historical patience branch was checked before starting. Its feature is
already present on master; merging that stale branch would revert widened weight
offsets. Start from current `origin/dev`, not that branch. Upstream master is
`af3ae37c09e31b95ebbc944d21494029122e58e5`; tag `v123` is
`7db5b4f1ad3af7eb5bd0c224b2fa5305e1a715d2`. No upstream solver update is needed.

## Agile backlog and ordered commits

- [ ] Design and tracking: publish this scope and explicit acceptance checks.
- [ ] RED: add real-training regressions and a small direct-library test driver;
  run them on the original code before editing production behavior.
- [ ] GREEN: repair checkpoint selection/restoration, guards, and `-b` handling;
  retain existing improving/worsening/default-patience behavior under tests.
- [ ] Verification: add a focused `make test` gate and Linux CI for both existing
  SSE/OpenMP and scalar/serial build configurations.
- [ ] Documentation: explain `-b`, ties, restoration at the iteration limit,
  validation requirements, and the test command.
- [ ] Closeout: independently review/run the final patch, merge the ordered
  commits to `dev`, then promote `dev` to `master`, close the issue, and remove
  the feature worktree. Preserve commits through merge commits; do not squash.

Each increment ends with its focused check. RED/GREEN commits remain ordered and
reviewable, while a pushed implementation increment ends with passing tests.

## Test design

Use Python's standard-library unittest driver and temporary datasets. Use fresh
single-thread CLI processes with the same validation input for reference prefix
runs, because the existing validation pass also consumes shuffle state.

- A constant validation margin with changing training weights exercises ties,
  exact patience exhaustion, and retention of the earliest best checkpoint.
- Positive training labels and negative validation labels on the same feature
  pattern produce worsening validation loss. Compare the selected model bytes
  and predictions against the independently trained best prefix.
- An iteration cap shorter than patience must still restore the earlier best.
- Improving validation, stopping disabled, and default patience characterize
  supported behavior. Verify actual models/predictions, not only log strings.
- A small C++ driver calls the existing public library API to test invalid
  patience/iteration counts, missing/empty validation, and nonfinite loss.
- CLI tests cover `-b` boundaries, malformed integer tokens, help/default output,
  and useful surfaced errors. No external dataset downloads or test framework.

These fixtures verify control flow and checkpoint correctness, not generalization
quality. Model-byte comparisons are within the same executable/backend build;
no cross-compiler or SIMD/scalar byte-equivalence promise is made.

## Exact file scope

Modify `ffm.cpp`, `ffm-train.cpp`, `Makefile`, and `README`.

Create `implementation_plan.md`, `docs/early_stopping.md`, `.gitignore`,
`.github/workflows/early-stopping.yml`, `tests/test_early_stopping.py`, and
`tests/early_stopping_api.cpp`. Update this scope and notify the owner before
editing another file. Keep `ffm.h`, `ffm-predict.cpp`, `Makefile.win`, timers, and
COPYRIGHT unchanged.

## Risk, verification, and done

Risk is concentrated in returning the correct saved model and defining ties.
The change intentionally makes plateaus stop and selects an earlier model when
the iteration cap is reached after deterioration. Best-weight storage remains
proportional to model size only while automatic stopping is enabled.

Verification commands are `make clean all test` and
`make clean all test DFLAG= CXXFLAGS='-Wall -O2 -std=c++11'`.
CI must fail on regression. The patch is complete only after independent runtime
verification, current-head CI, merge into `dev` and then `master`, issue closure,
and worktree cleanup. Use pstack's behavior-first tests and real-artifact proof,
with PUA's evidence and end-to-end closeout discipline.
