#include "include/cholesky.hpp"
#include "include/convergence.hpp"
#include "include/gbm.hpp"
#include "include/sde_integrator.hpp"
#include "include/statistics.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "--- GBM Multi-Asset Path Simulator ---\n\n";

  // 1. Configure the model for 3 correlated assets
  GBMConfig config;
  config.num_assets = 3;
  config.num_paths = 10000;
  config.num_steps = 100;
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
  Statistics stats(model);

  // 3. Setup Random Number Generation
  std::mt19937 generator(config.random_seed);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  // =========================================================
  // 2. Simulate N paths, collect terminal values per asset
  // =========================================================
  std::cout << "Simulating " << config.num_paths << " paths with "
            << config.num_steps << " steps...\n";

  // terminal_values[asset_idx][path] = S_T for that path
  std::vector<std::vector<double>> terminal_values(
      config.num_assets, std::vector<double>(config.num_paths));

  // Store a few sample paths for CSV export (first 5 paths)
  size_t sample_path_count = 5;
  // sample_paths[path][step][asset]
  std::vector<std::vector<std::vector<double>>> sample_paths;

  for (size_t path = 0; path < config.num_paths; ++path) {
    std::vector<double> current_prices = config.S0;

    // Optionally store this path
    bool store_path = (path < sample_path_count);
    if (store_path) {
      sample_paths.push_back({current_prices}); // step 0
    }

    for (size_t step = 0; step < config.num_steps; ++step) {
      std::vector<double> z(config.num_assets);
      for (size_t i = 0; i < config.num_assets; ++i) {
        z[i] = normal_dist(generator);
      }
      std::vector<double> correlated = cholesky.generate_correlated(z);

      for (size_t i = 0; i < config.num_assets; ++i) {
        double dW = std::sqrt(config.dt) * correlated[i];
        current_prices[i] =
            integrator.euler_step(i, current_prices[i], config.dt, dW);
      }

      if (store_path) {
        sample_paths.back().push_back(current_prices);
      }
    }

    for (size_t i = 0; i < config.num_assets; ++i) {
      terminal_values[i][path] = current_prices[i];
    }
  }
  std::cout << "Simulation complete.\n\n";

  // =========================================================
  // 3. Compute and display statistics
  // =========================================================
  std::cout << "--- Statistics (Empirical vs Theoretical) ---\n";
  std::vector<AssetStatistics> all_stats;
  for (size_t i = 0; i < config.num_assets; ++i) {
    AssetStatistics s = stats.compute_asset_statistics(i, terminal_values[i]);
    all_stats.push_back(s);

    std::cout << "Asset " << i << ": Mean=" << s.mean_empirical << " (theo "
              << s.mean_theoretical << ")"
              << ", Var=" << s.variance_empirical << " (theo "
              << s.variance_theoretical << ")"
              << ", Skew=" << s.skewness_empirical
              << ", Kurt=" << s.kurtosis_empirical << "\n";
  }

  // =========================================================
  // 4. Write stats.csv
  // =========================================================
  {
    std::ofstream file("data/stats.csv");
    file << "asset_id,mean_empirical,mean_theoretical,variance_empirical,"
            "variance_theoretical,skewness_empirical,kurtosis_empirical\n";
    for (const auto &s : all_stats) {
      file << s.asset_idx << "," << s.mean_empirical << ","
           << s.mean_theoretical << "," << s.variance_empirical << ","
           << s.variance_theoretical << "," << s.skewness_empirical << ","
           << s.kurtosis_empirical << "\n";
    }
    std::cout << "\n[CSV] Wrote data/stats.csv\n";
  }

  // =========================================================
  // 5. Convergence Benchmark
  // =========================================================
  std::cout << "\n=== Strong Convergence Benchmark ===\n";

  GBMConfig conv_config;
  conv_config.num_assets = 1;
  conv_config.num_paths = 10000;
  conv_config.num_steps = 10;
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

  // Print to console
  std::cout << std::fixed << std::setprecision(6);
  for (const auto &r : conv_results) {
    std::cout << r.scheme << "  dt=" << r.dt << "  error=" << r.mean_error
              << "  rmse=" << r.rmse << "\n";
  }

  // =========================================================
  // 6. Write convergence.csv
  // =========================================================
  {
    std::ofstream file("data/convergence.csv");
    file << "dt,scheme,mean_error,rmse,log_dt,log_error\n";
    for (const auto &r : conv_results) {
      file << r.dt << "," << r.scheme << "," << r.mean_error << "," << r.rmse
           << "," << r.log_dt << "," << r.log_error << "\n";
    }
    std::cout << "\n[CSV] Wrote data/convergence.csv\n";
  }

  // =========================================================
  // 7. Write paths.csv (sample paths only)
  // =========================================================
  {
    std::ofstream file("data/paths.csv");
    file << "path_id,time,asset_id,value\n";
    for (size_t p = 0; p < sample_paths.size(); ++p) {
      for (size_t step = 0; step <= config.num_steps; ++step) {
        double time = step * config.dt;
        for (size_t i = 0; i < config.num_assets; ++i) {
          file << p << "," << time << "," << i << ","
               << sample_paths[p][step][i] << "\n";
        }
      }
    }
    std::cout << "[CSV] Wrote data/paths.csv\n";
  }

  std::cout << "\nDone.\n";
  return 0;
}
