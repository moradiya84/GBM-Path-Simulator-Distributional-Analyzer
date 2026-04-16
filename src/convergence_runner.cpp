#include "../include/cholesky.hpp"
#include "../include/convergence.hpp"
#include "../include/sde_integrator.hpp"
#include <random>

ConvergenceRunner::ConvergenceRunner(const GBMConfig &base_config)
    : base_config_(base_config) {}

std::vector<ConvergenceResult>
ConvergenceRunner::run(const std::vector<size_t> &step_counts) const {
  std::vector<ConvergenceResult> results;
  Cholesky cholesky(base_config_.correlation_matrix);

  // Test both schemes
  std::vector<Scheme> schemes = {Scheme::Euler, Scheme::Milstein};

  for (Scheme scheme : schemes) {
    for (size_t num_steps : step_counts) {
      // Build config for this specific dt
      GBMConfig config = base_config_;
      config.num_steps = num_steps;
      config.dt = config.T / num_steps;
      config.scheme = scheme;

      GBM model(config);
      SDEIntegrator integrator(model);

      double sum_abs_error = 0.0;
      double sum_sq_error = 0.0;

      for (size_t path = 0; path < config.num_paths; ++path) {
        // Seed each path deterministically but uniquely
        std::mt19937 generator(config.random_seed +
                               static_cast<unsigned int>(path));
        std::normal_distribution<double> normal_dist(0.0, 1.0);

        // Track current prices and accumulated W_T per asset
        std::vector<double> current_prices = config.S0;
        std::vector<double> W_T(config.num_assets, 0.0);

        for (size_t step = 0; step < num_steps; ++step) {
          // Generate independent normals
          std::vector<double> z(config.num_assets);
          for (size_t i = 0; i < config.num_assets; ++i) {
            z[i] = normal_dist(generator);
          }

          // Correlate via Cholesky
          std::vector<double> correlated = cholesky.generate_correlated(z);

          // Scale to Brownian increments and step each asset
          for (size_t i = 0; i < config.num_assets; ++i) {
            double dW = std::sqrt(config.dt) * correlated[i];
            W_T[i] += dW;

            if (scheme == Scheme::Milstein) {
              current_prices[i] =
                  integrator.milstein_step(i, current_prices[i], config.dt, dW);
            } else {
              current_prices[i] =
                  integrator.euler_step(i, current_prices[i], config.dt, dW);
            }
          }
        }

        // Compute error against exact solution for each asset
        for (size_t i = 0; i < config.num_assets; ++i) {
          double S_exact = model.exact_solution_T(i, W_T[i]);
          double error = std::abs(current_prices[i] - S_exact);
          sum_abs_error += error;
          sum_sq_error += error * error;
        }
      }

      double total_samples =
          static_cast<double>(config.num_paths * config.num_assets);
      double mean_error = sum_abs_error / total_samples;
      double rmse = std::sqrt(sum_sq_error / total_samples);

      ConvergenceResult result;
      result.dt = config.dt;
      result.scheme = (scheme == Scheme::Euler) ? "Euler" : "Milstein";
      result.mean_error = mean_error;
      result.rmse = rmse;
      result.log_dt = std::log(config.dt);
      result.log_error = std::log(mean_error);

      results.push_back(result);
    }
  }

  return results;
}
