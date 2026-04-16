#include "include/cholesky.hpp"
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
    // Step a: Generate M independent standard normals
    std::vector<double> z(config.num_assets);
    for (size_t i = 0; i < config.num_assets; ++i) {
      z[i] = normal_dist(generator);
    }

    // Step b: Multiply by Cholesky factor L to correlate
    std::vector<double> correlated = cholesky.generate_correlated(z);

    // Step c: Scale by sqrt(dt) to get actual Brownian increments
    for (size_t i = 0; i < config.num_assets; ++i) {
      dW_paths[step][i] = std::sqrt(config.dt) * correlated[i];
    }
  }

  // 5. Simulate the multi-asset path
  auto paths = integrator.simulate_path(dW_paths);

  // 6. Print results
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
  return 0;
}
