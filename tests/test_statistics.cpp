#include "../include/cholesky.hpp"
#include "../include/gbm.hpp"
#include "../include/sde_integrator.hpp"
#include "../include/statistics.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

bool is_close(double a, double b, double tol = 1e-6) {
  return std::abs(a - b) < tol;
}

// 1. Validate correctness on known inputs
void test_known_inputs() {
  // Known dataset: {2, 4, 4, 4, 5, 5, 7, 9}
  // Mean = 5.0
  // Population Variance = 4.0
  // Skewness ≈ 0.0
  // Excess Kurtosis ≈ -0.7675 (platykurtic)

  GBMConfig config;
  config.num_assets = 1;
  config.num_paths = 8;
  config.num_steps = 1;
  config.T = 1.0;
  config.dt = 1.0;
  config.S0 = {100.0};
  config.mu = {0.05};
  config.sigma = {0.2};
  config.correlation_matrix = {{1.0}};
  config.random_seed = 42;
  config.scheme = Scheme::Euler;

  GBM model(config);
  Statistics stats(model);

  std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};

  double mean = stats.compute_mean(data);
  assert(is_close(mean, 5.0));
  std::cout << "[PASS] Mean = " << mean << " (expected 5.0)\n";

  double variance = stats.compute_variance(data);
  assert(is_close(variance, 4.0));
  std::cout << "[PASS] Variance = " << variance << " (expected 4.0)\n";

  double skewness = stats.compute_skewness(data);
  // Dataset {2,4,4,4,5,5,7,9} is slightly right-skewed
  assert(skewness > 0.0);
  std::cout << "[PASS] Skewness = " << skewness
            << " (positive, right-skewed)\n";

  double kurtosis = stats.compute_kurtosis(data);
  // Platykurtic (lighter tails than normal)
  assert(kurtosis < 0.0);
  std::cout << "[PASS] Excess Kurtosis = " << kurtosis
            << " (negative, platykurtic)\n";
}

// 2. Compare GBM empirical vs theoretical using Monte Carlo
void test_empirical_vs_theoretical() {
  GBMConfig config;
  config.num_assets = 1;
  config.num_paths = 100000;
  config.num_steps = 100;
  config.T = 1.0;
  config.dt = config.T / config.num_steps;
  config.S0 = {100.0};
  config.mu = {0.05};
  config.sigma = {0.2};
  config.correlation_matrix = {{1.0}};
  config.random_seed = 42;
  config.scheme = Scheme::Euler;

  GBM model(config);
  SDEIntegrator integrator(model);
  Cholesky cholesky(config.correlation_matrix);
  Statistics stats(model);

  std::mt19937 generator(config.random_seed);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  // Simulate N paths and collect terminal values
  std::vector<double> terminal_values(config.num_paths);

  for (size_t path = 0; path < config.num_paths; ++path) {
    double current_price = config.S0[0];
    for (size_t step = 0; step < config.num_steps; ++step) {
      double z = normal_dist(generator);
      double dW = std::sqrt(config.dt) * z;
      current_price = integrator.euler_step(0, current_price, config.dt, dW);
    }
    terminal_values[path] = current_price;
  }

  AssetStatistics result = stats.compute_asset_statistics(0, terminal_values);

  std::cout << "\n--- Empirical vs Theoretical (100k paths) ---\n";
  std::cout << "Mean:     Empirical = " << result.mean_empirical
            << ",  Theoretical = " << result.mean_theoretical << "\n";
  std::cout << "Variance: Empirical = " << result.variance_empirical
            << ",  Theoretical = " << result.variance_theoretical << "\n";
  std::cout << "Skewness: Empirical = " << result.skewness_empirical << "\n";
  std::cout << "Kurtosis: Empirical = " << result.kurtosis_empirical << "\n";

  // Mean should be within 1% of theoretical
  double mean_rel_err =
      std::abs(result.mean_empirical - result.mean_theoretical) /
      result.mean_theoretical;
  assert(mean_rel_err < 0.01);
  std::cout << "[PASS] Mean relative error = " << mean_rel_err * 100 << "%\n";

  // Variance should be within 5% of theoretical
  double var_rel_err =
      std::abs(result.variance_empirical - result.variance_theoretical) /
      result.variance_theoretical;
  assert(var_rel_err < 0.05);
  std::cout << "[PASS] Variance relative error = " << var_rel_err * 100
            << "%\n";

  // GBM terminal distribution is lognormal, so skewness should be positive
  assert(result.skewness_empirical > 0.0);
  std::cout << "[PASS] Skewness is positive (lognormal property)\n";

  // Lognormal excess kurtosis should also be positive
  assert(result.kurtosis_empirical > 0.0);
  std::cout << "[PASS] Excess kurtosis is positive (lognormal heavy tails)\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "Starting Statistics Tests...\n";
  std::cout << "========================================\n";

  test_known_inputs();
  test_empirical_vs_theoretical();

  std::cout << "========================================\n";
  std::cout << "All statistics tests passed.\n";
  std::cout << "========================================\n";
  return 0;
}
