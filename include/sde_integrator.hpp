#pragma once

#include "gbm.hpp"
#include <cstddef>

// SDEIntegrator Class:
// Responsible for progressing the state of the asset prices using numerical
// schemes.
class SDEIntegrator {
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
  double euler_step(size_t asset_idx, double S_n, double dt, double dW) const {
    double mu_S = model_.drift(asset_idx, S_n);
    double sigma_S = model_.diffusion(asset_idx, S_n);
    // s_n + ds_n and dS_n = mu_S * dt + sigma_S * dW
    return S_n + mu_S * dt + sigma_S * dW;
  }
};
