#pragma once

#include "gbm.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

// SDEIntegrator Class:
// Responsible for progressing the state of the asset prices using numerical
// schemes.
class SDEIntegrator
{
private:
  const GBM &model_;

public:
  // Initialize with a reference to the mathematical model
  explicit SDEIntegrator(const GBM &model) : model_(model) {}

  // Performs a single Euler-Maruyama step for a specific asset
  // asset_idx : Index of the asset being simulated
  // S_n       : The asset's price at the beginning of the time step
  // dt        : The time step size
  // dW        : The Brownian motion increment for this asset over this step
  //
  // Returns S_{n+1} = S_n + drift * dt + diffusion * dW
  double euler_step(size_t asset_idx, double S_n, double dt, double dW) const
  {
    double mu_S = model_.drift(asset_idx, S_n);
    double sigma_S = model_.diffusion(asset_idx, S_n);
    // s_n + ds_n and dS_n = mu_S * dt + sigma_S * dW
    return S_n + mu_S * dt + sigma_S * dW;
  }

  // Performs a single Milstein step for a specific asset
  // Returns S_{n+1} = S_n + drift*dt + diffusion*dW + 0.5*diffusion*diffusion_derivative*(dW^2 - dt)
  double milstein_step(size_t asset_idx, double S_n, double dt, double dW) const
  {
    double mu_S = model_.drift(asset_idx, S_n);
    double sigma_S = model_.diffusion(asset_idx, S_n);
    double d_sigma_S = model_.diffusion_derivative(asset_idx, S_n);

    return S_n + mu_S * dt + sigma_S * dW + 0.5 * sigma_S * d_sigma_S * (dW * dW - dt);
  }

  // Simulates full multi-asset state across the configured number of steps.
  // dW_paths: 2D vector where dW_paths[step] is a vector of correlated
  // increments for all assets. Returns: a 2D vector of size (num_steps + 1) x
  // num_assets containing simulated prices.
  std::vector<std::vector<double>>
  simulate_path(const std::vector<std::vector<double>> &dW_paths) const
  {
    size_t num_steps = model_.get_config().num_steps;
    size_t num_assets = model_.get_config().num_assets;
    double dt = model_.get_config().dt;
    std::vector<double> current_prices = model_.get_config().S0;

    if (dW_paths.size() != num_steps)
    {
      throw std::invalid_argument(
          "Brownian increments path size must equal num_steps");
    }

    std::vector<std::vector<double>> paths;
    paths.reserve(num_steps + 1);
    paths.push_back(current_prices); // Push start prices

    for (size_t step = 0; step < num_steps; ++step)
    {
      if (dW_paths[step].size() != num_assets)
      {
        throw std::invalid_argument(
            "Each step in dW_paths must have size equal to num_assets");
      }

      std::vector<double> next_prices(num_assets);
      for (size_t i = 0; i < num_assets; ++i)
      {
        if (model_.get_config().scheme == Scheme::Milstein)
        {
          next_prices[i] =
              milstein_step(i, current_prices[i], dt, dW_paths[step][i]);
        }
        else
        {
          next_prices[i] =
              euler_step(i, current_prices[i], dt, dW_paths[step][i]);
        }
      }
      current_prices = next_prices;
      paths.push_back(current_prices);
    }

    return paths;
  }
};
