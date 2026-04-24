#include "../include/gbm.hpp"
#include "../include/sde_integrator.hpp"
#include "../include/time_grid.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Utility to check floating point equality
bool is_close(double a, double b, double tol = 1e-6) {
  return std::abs(a - b) < tol;
}

// 1. One-step Euler update test
void test_one_step_euler() {
  GBMConfig config;
  config.num_assets = 1;
  config.S0 = {100.0};
  config.mu = {0.05};
  config.sigma = {0.2};
  config.correlation_matrix = {{1.0}};
  config.scheme = Scheme::Euler;

  GBM model(config);
  SDEIntegrator integrator(model);

  double S_n = 100.0;
  double dt = 0.1;
  double dW = 0.5; // predetermined deterministic increment for testing

  // Expected Euler Step Calculation:
  // S_{n+1} = S_n + mu_i * S_n * dt + sigma_i * S_n * dW
  //         = 100 + (0.05) * 100 * 0.1 + (0.2) * 100 * 0.5
  //         = 100 + 0.5 + 10.0 = 110.5
  double S_next = integrator.euler_step(0, S_n, dt, dW);

  assert(is_close(S_next, 110.5));
  std::cout << "[PASS] test_one_step_euler\n";
}

// 2. Basic path evolution (sanity check) test
void test_basic_path_evolution() {
  GBMConfig config;
  config.num_assets = 1;
  config.S0 = {100.0};
  config.mu = {0.05};
  config.sigma = {0.2};
  config.correlation_matrix = {{1.0}};
  config.scheme = Scheme::Euler;

  GBM model(config);
  SDEIntegrator integrator(model);

  double current_price = config.S0[0];

  // Step 1: Add a positive shock
  double dt = 0.1;
  double dW1 = 0.1;
  current_price = integrator.euler_step(0, current_price, dt, dW1);

  // S_1 expected:
  // 100 + 0.05 * 100 * 0.1 + 0.2 * 100 * 0.1 = 100 + 0.5 + 2.0 = 102.5
  assert(is_close(current_price, 102.5));

  // Step 2: Add a negative shock
  double dW2 = -0.1;
  current_price = integrator.euler_step(0, current_price, dt, dW2);

  // S_2 expected:
  // 102.5 + 0.05 * 102.5 * 0.1 + 0.2 * 102.5 * (-0.1)
  // = 102.5 + 0.5125 - 2.05 = 100.9625
  assert(is_close(current_price, 100.9625));

  std::cout << "[PASS] test_basic_path_evolution\n";
}

// 3. Milstein accuracy verification
void test_milstein_accuracy() {
  size_t num_steps = 100;
  double T = 1.0;

  GBMConfig config;
  config.num_assets = 1;
  config.S0 = {100.0};
  config.mu = {0.05};
  config.sigma = {0.2};
  config.correlation_matrix = {{1.0}};

  TimeGrid grid(T, num_steps);
  double dt = grid.dt();

  unsigned int random_seed = 42;
  std::mt19937 generator(random_seed);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  // Synthesize 100-step Brownian Path
  std::vector<std::vector<double>> dW_paths(
      num_steps, std::vector<double>(config.num_assets));
  double W_T = 0.0;

  for (size_t step = 0; step < num_steps; ++step) {
    double dW = std::sqrt(dt) * normal_dist(generator);
    dW_paths[step][0] = dW;
    W_T += dW;
  }

  // Simulate Euler Path via Multi-Asset Integration Engine
  config.scheme = Scheme::Euler;
  GBM model_euler(config);
  SDEIntegrator integrator_euler(model_euler);
  auto paths_euler = integrator_euler.simulate_path(grid, dW_paths);
  double S_euler = paths_euler.back()[0];

  // Simulate Milstein Path via Multi-Asset Integration Engine
  config.scheme = Scheme::Milstein;
  GBM model_milstein(config);
  SDEIntegrator integrator_milstein(model_milstein);
  auto paths_milstein = integrator_milstein.simulate_path(grid, dW_paths);
  double S_milstein = paths_milstein.back()[0];

  // Target true mathematical exact scalar
  double S_exact = model_euler.exact_solution_T(0, T, W_T);

  double euler_error = std::abs(S_euler - S_exact);
  double milstein_error = std::abs(S_milstein - S_exact);

  std::cout << "[INFO] Exact: " << S_exact << ", Euler: " << S_euler
            << " (Err: " << euler_error << ")"
            << ", Milstein: " << S_milstein << " (Err: " << milstein_error
            << ")\n";

  assert(milstein_error < euler_error);
  std::cout << "[PASS] test_milstein_accuracy (Milstein converges physically "
               "tighter than Euler)\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "Starting SDE Integrator Tests...\n";
  std::cout << "========================================\n";

  test_one_step_euler();
  test_basic_path_evolution();
  test_milstein_accuracy();

  std::cout << "========================================\n";
  std::cout << "All integrations passed successfully.\n";
  std::cout << "========================================\n";
  return 0;
}
