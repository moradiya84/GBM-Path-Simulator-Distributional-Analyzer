#include "../include/gbm.hpp"
#include "../include/sde_integrator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// Utility to check floating point equality
bool is_close(double a, double b, double tol = 1e-6) {
  return std::abs(a - b) < tol;
}

// 1. One-step Euler update test
void test_one_step_euler() {
  GBMConfig config;
  config.num_assets = 1;
  config.num_paths = 1;
  config.num_steps = 10;
  config.T = 1.0;
  config.dt = 0.1;
  config.S0 = {100.0};
  config.mu = {0.05};
  config.sigma = {0.2};
  config.rho = {{1.0}};
  config.random_seed = 42;
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
  config.num_paths = 1;
  config.num_steps = 2; // Testing 2 steps
  config.T = 0.2;
  config.dt = 0.1;
  config.S0 = {100.0};
  config.mu = {0.05};
  config.sigma = {0.2};
  config.rho = {{1.0}};
  config.random_seed = 42;
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

int main() {
  std::cout << "========================================\n";
  std::cout << "Starting SDE Integrator Tests...\n";
  std::cout << "========================================\n";

  test_one_step_euler();
  test_basic_path_evolution();

  std::cout << "========================================\n";
  std::cout << "All integrations passed successfully.\n";
  std::cout << "========================================\n";
  return 0;
}
