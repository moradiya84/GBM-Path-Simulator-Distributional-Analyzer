#include "../include/cholesky.hpp"
#include "../include/convergence.hpp"
#include "../include/sde_integrator.hpp"
#include <random>

#include "../include/time_grid.hpp"
#include "../include/rng.hpp"

ConvergenceRunner::ConvergenceRunner(const GBMConfig &base_config, double T, size_t num_paths, unsigned int random_seed)
    : base_config_(base_config), T_(T), num_paths_(num_paths), random_seed_(random_seed) {}

std::vector<ConvergenceResult>
ConvergenceRunner::run(const std::vector<size_t> &step_counts) const {
  std::vector<ConvergenceResult> results;
  Cholesky cholesky(base_config_.correlation_matrix);

  // Test both schemes
  std::vector<Scheme> schemes = {Scheme::Euler, Scheme::Milstein};

  for (Scheme scheme : schemes) {
    for (size_t num_steps : step_counts) {
      // Build config for this specific scheme
      GBMConfig config = base_config_;
      config.scheme = scheme;
      
      TimeGrid grid(T_, num_steps);

      GBM model(config);
      SDEIntegrator integrator(model);

      double sum_abs_error = 0.0;
      double sum_sq_error = 0.0;

      for (size_t path = 0; path < num_paths_; ++path) {
        // Seed each path deterministically but uniquely
        PseudoRNG rng(random_seed_ + static_cast<unsigned int>(path));

        // Track current prices and accumulated W_T per asset
        std::vector<double> current_prices = config.S0;
        std::vector<double> W_T(config.num_assets, 0.0);

        for (size_t step = 0; step < num_steps; ++step) {
          // Generate independent normals
          std::vector<double> z = rng.generate_standard_normal(config.num_assets);

          // Correlate via Cholesky
          std::vector<double> correlated = cholesky.generate_correlated(z);

          // Scale to Brownian increments and step each asset
          for (size_t i = 0; i < config.num_assets; ++i) {
            double dW = std::sqrt(grid.dt()) * correlated[i];
            W_T[i] += dW;

            if (scheme == Scheme::Milstein) {
              current_prices[i] =
                  integrator.milstein_step(i, current_prices[i], grid.dt(), dW);
            } else {
              current_prices[i] =
                  integrator.euler_step(i, current_prices[i], grid.dt(), dW);
            }
          }
        }

        // Compute error against exact solution for each asset
        for (size_t i = 0; i < config.num_assets; ++i) {
          double S_exact = model.exact_solution_T(i, T_, W_T[i]);
          double error = std::abs(current_prices[i] - S_exact);
          sum_abs_error += error;
          sum_sq_error += error * error;
        }
      }

      double total_samples =
          static_cast<double>(num_paths_ * config.num_assets);
      double mean_error = sum_abs_error / total_samples;
      double rmse = std::sqrt(sum_sq_error / total_samples);

      ConvergenceResult result;
      result.dt = grid.dt();
      result.scheme = (scheme == Scheme::Euler) ? "Euler" : "Milstein";
      result.mean_error = mean_error;
      result.rmse = rmse;
      result.log_dt = std::log(grid.dt());
      result.log_error = std::log(mean_error);

      results.push_back(result);
    }
  }

  return results;
}
