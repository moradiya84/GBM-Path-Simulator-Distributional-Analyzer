#pragma once

#include "gbm.hpp"
#include <string>
#include <vector>

// Stores the result of a single convergence data point (one dt value, one
// scheme)
struct ConvergenceResult {
  double dt;
  std::string scheme;
  double mean_error;
  double rmse;
  double log_dt;
  double log_error;
};

class ConvergenceRunner {
private:
  GBMConfig base_config_;

public:
  explicit ConvergenceRunner(const GBMConfig &base_config);

  /* Runs the convergence benchmark over multiple step sizes.
     For each step size:
       - Simulates num_paths using the same Brownian paths for both numerical
     and exact
       - Computes mean absolute error and RMSE at terminal time
     step_counts: vector of num_steps values to test (e.g., {10, 20, 50, 100,
     200, 500}) Returns a vector of ConvergenceResult, one per (step_count,
     scheme) pair */
  std::vector<ConvergenceResult>
  run(const std::vector<size_t> &step_counts) const;
};
