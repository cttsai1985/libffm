#include <exception>
#include <iostream>
#include <string>

#include "../ffm.h"

#if defined USEOMP
#include <omp.h>
#endif

// Exercise the existing public API without CLI validation hiding missing guards.
int main(int argc, char **argv) {
    if(argc != 6) {
        std::cerr << "usage: early_stopping_api MODE TRAIN VALID TRAIN_BIN VALID_BIN\n";
        return 2;
    }

#if defined USEOMP
    omp_set_num_threads(1);
#endif

    std::string mode = argv[1];
    ffm::ffm_parameter param;
    param.auto_stop = true;
    param.nr_iters = 3;
    param.nr_patience_iters = 2;

    if(mode == "patience-zero")
        param.nr_patience_iters = 0;
    else if(mode == "patience-negative")
        param.nr_patience_iters = -1;
    else if(mode == "iterations-zero")
        param.nr_iters = 0;
    else if(mode == "iterations-negative")
        param.nr_iters = -1;
    else if(mode != "missing-validation" && mode != "empty-validation" &&
            mode != "nonfinite-validation" && mode != "empty-training") {
        std::cerr << "unknown mode: " << mode << "\n";
        return 2;
    }

    // Conversion is outside the expected-error block: only training rejection
    // satisfies these tests, rather than a parser or filesystem failure.
    ffm::ffm_read_problem_to_disk(argv[2], argv[4]);
    if(mode != "missing-validation")
        ffm::ffm_read_problem_to_disk(argv[3], argv[5]);

    try {
        ffm::ffm_model model = ffm::ffm_train_on_disk(
            argv[4], mode == "missing-validation" ? "" : argv[5], param);
    } catch(const std::exception &error) {
        if(std::string(error.what()).empty()) {
            std::cerr << "training rejected the input without a useful error\n";
            return 1;
        }
        std::cout << "REJECTED: " << error.what() << "\n";
        return 0;
    }

    std::cerr << "training accepted invalid early-stopping input: " << mode << "\n";
    return 1;
}
