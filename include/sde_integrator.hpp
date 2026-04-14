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

  // Simulates an entire path for a single asset across configured number of
  // steps dW_path: vector of Brownian increments of size num_steps Returns a
  // vector of simulated prices of size (num_steps + 1) including S0
  std::vector<double> simulate_path(size_t asset_idx, const std::vector<double> &dW_path) const
  {
    size_t num_steps = model_.get_config().num_steps;
    double dt = model_.get_config().dt;
    double current_price = model_.get_config().S0[asset_idx];

    if (dW_path.size() != num_steps)
    {
      throw std::invalid_argument("Brownian increments path size must equal num_steps");
    }

    std::vector<double> path;
    path.push_back(current_price); // Push start price

    for (size_t step = 0; step < num_steps; ++step)
    {
      current_price = euler_step(asset_idx, current_price, dt, dW_path[step]);
      path.push_back(current_price);
    }

    return path;
  }
};
