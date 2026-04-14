#include "include/cholesky.hpp"
#include "include/gbm.hpp"
#include "include/sde_integrator.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "--- GBM Path Simulator (Temporary Driver) ---\n";

  // 1. Configure the model for 1 asset
  GBMConfig config;
  config.num_assets = 1;
  config.num_paths = 1;  // Simulating 1 path
  config.num_steps = 10; // 10 steps for brevity
  config.T = 1.0;        // 1 year
  config.dt = config.T / config.num_steps;

  config.S0 = {100.0};
  config.mu = {0.05};   // 5% drift
  config.sigma = {0.2}; // 20% volatility
  config.correlation_matrix = {{1.0}};
  config.random_seed = 42;
  config.scheme = Scheme::Euler;

  // 2. Initialize Model Components
  GBM model(config);
  SDEIntegrator integrator(model);
  Cholesky cholesky(config.correlation_matrix);

  // 3. Setup Random Number Generation
  std::mt19937 generator(config.random_seed);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  // 4. Simulate the Path
  double current_price = config.S0[0];
  double current_time = 0.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Step\tTime\t\tPrice\n";
  std::cout << "0\t" << current_time << "\t\t" << current_price << "\n";

  for (size_t step = 1; step <= config.num_steps; ++step) {
    // Generate independent normals
    std::vector<double> z = {normal_dist(generator)};

    // Correlate normals (Using our Cholesky module)
    std::vector<double> x = cholesky.generate_correlated(z);

    // Scale to Brownian increment: dW = sqrt(dt) * x
    double dW = std::sqrt(config.dt) * x[0];

    // Perform Euler step
    current_price = integrator.euler_step(0, current_price, config.dt, dW);
    current_time += config.dt;

    std::cout << step << "\t" << current_time << "\t\t" << current_price
              << "\n";
  }

  std::cout << "---------------------------------------------\n";
  return 0;
}
