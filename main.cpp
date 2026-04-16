#include "include/cholesky.hpp"
#include "include/convergence.hpp"
#include "include/gbm.hpp"
#include "include/sde_integrator.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "--- GBM Multi-Asset Path Simulator ---\n\n";

  // 1. Configure the model for 3 correlated assets
  GBMConfig config;
  config.num_assets = 3;
  config.num_paths = 1;
  config.num_steps = 10;
  config.T = 1.0;
  config.dt = config.T / config.num_steps;

  config.S0 = {100.0, 50.0, 75.0};
  config.mu = {0.05, 0.08, 0.03};
  config.sigma = {0.2, 0.3, 0.15};
  config.correlation_matrix = {
      {1.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 1.0}};
  config.random_seed = 42;
  config.scheme = Scheme::Euler;

  // 2. Initialize Model Components
  GBM model(config);
  SDEIntegrator integrator(model);
  Cholesky cholesky(config.correlation_matrix);

  // 3. Setup Random Number Generation
  std::mt19937 generator(config.random_seed);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  // 4. Generate all correlated Brownian increments upfront
  std::vector<std::vector<double>> dW_paths(
      config.num_steps, std::vector<double>(config.num_assets));

  for (size_t step = 0; step < config.num_steps; ++step) {
    std::vector<double> z(config.num_assets);
    for (size_t i = 0; i < config.num_assets; ++i) {
      z[i] = normal_dist(generator);
    }
    std::vector<double> correlated = cholesky.generate_correlated(z);
    for (size_t i = 0; i < config.num_assets; ++i) {
      dW_paths[step][i] = std::sqrt(config.dt) * correlated[i];
    }
  }

  // 5. Simulate the multi-asset path
  auto paths = integrator.simulate_path(dW_paths);

  // 6. Print path results
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Step\tTime\t\tAsset 0\t\tAsset 1\t\tAsset 2\n";
  std::cout << std::string(72, '-') << "\n";

  for (size_t step = 0; step <= config.num_steps; ++step) {
    double time = step * config.dt;
    std::cout << step << "\t" << time;
    for (size_t i = 0; i < config.num_assets; ++i) {
      std::cout << "\t\t" << paths[step][i];
    }
    std::cout << "\n";
  }
  std::cout << std::string(72, '-') << "\n";

  // =========================================================
  // 7. Convergence Benchmark
  // =========================================================
  std::cout << "\n=== Strong Convergence Benchmark ===\n\n";

  // Use a single-asset config for convergence (cleaner benchmark)
  GBMConfig conv_config;
  conv_config.num_assets = 1;
  conv_config.num_paths = 10000;
  conv_config.num_steps = 10; // overridden by runner
  conv_config.T = 1.0;
  conv_config.dt = conv_config.T / conv_config.num_steps;
  conv_config.S0 = {100.0};
  conv_config.mu = {0.05};
  conv_config.sigma = {0.2};
  conv_config.correlation_matrix = {{1.0}};
  conv_config.random_seed = 42;
  conv_config.scheme = Scheme::Euler;

  ConvergenceRunner runner(conv_config);
  std::vector<size_t> step_counts = {10, 20, 50, 100, 200, 500};
  auto conv_results = runner.run(step_counts);

  // Print convergence results
  std::cout << std::left << std::setw(12) << "Scheme" << std::setw(14) << "dt"
            << std::setw(16) << "Mean Error" << std::setw(16) << "RMSE"
            << std::setw(14) << "log(dt)" << std::setw(14) << "log(error)"
            << "\n";
  std::cout << std::string(86, '-') << "\n";

  for (const auto &r : conv_results) {
    std::cout << std::left << std::setw(12) << r.scheme << std::setw(14) << r.dt
              << std::setw(16) << r.mean_error << std::setw(16) << r.rmse
              << std::setw(14) << r.log_dt << std::setw(14) << r.log_error
              << "\n";
  }
  std::cout << std::string(86, '-') << "\n";

  return 0;
}
