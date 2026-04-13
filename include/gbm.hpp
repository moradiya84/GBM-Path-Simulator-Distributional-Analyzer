#pragma once

#include <cmath>
#include <vector>

enum class Scheme { Euler, Milstein };

struct GBMConfig {
  size_t num_assets;
  size_t num_paths;
  size_t num_steps;
  double T;  // Maturity time
  double dt; // Time step size = T / num_steps

  std::vector<double> S0;    // Initial prices [num_assets]
  std::vector<double> mu;    // Drifts [num_assets]
  std::vector<double> sigma; // Volatilities [num_assets]
  std::vector<std::vector<double>>
      correlation_matrix; // Correlation matrix [num_assets x num_assets]

  unsigned int random_seed;
  Scheme scheme;
};

// GBM Class:
// Stores parameters, provides drift/diffusion functions, and the exact
// solution.
class GBM {
private:
  GBMConfig config_;

public:
  explicit GBM(const GBMConfig &config) : config_(config) {}

  const GBMConfig &get_config() const { return config_; }

  // Drift term for asset i given current price S
  // a(t, S) = mu_i * S
  double drift(size_t asset_idx, double S) const {
    return config_.mu[asset_idx] * S;
  }

  // Diffusion term for asset i given current price S
  // b(t, S) = sigma_i * S
  double diffusion(size_t asset_idx, double S) const {
    return config_.sigma[asset_idx] * S;
  }

  // Derivative of the diffusion term with respect to S
  // b'(t, S) = sigma_i
  // (Used in the Milstein scheme)
  double diffusion_derivative(size_t asset_idx, double /*S*/) const {
    return config_.sigma[asset_idx];
  }

  // Exact theoretical solution for asset i at time t, given accumulated
  // Brownian motion W_t
  /*   we have ds = mu*s*dt + sigma*s*dW_t
       now let's assume some function f(s) = ln(s)
       then df = (1/s)ds - (1/2)*(1/s^2)ds^2
       substitute ds = mu*s*dt + sigma*s*dW_t
       now in ds^2 dw^2 term will become dt because of quadratic variation of
     Brownian motion

       df = mu*dt + sigma*dW_t - (1/2)*sigma^2*dt
       df = (mu - 0.5*sigma^2)*dt + sigma*dW_t
       integrate both sides
       ln(s_t) - ln(s_0) = (mu - 0.5*sigma^2)*t + sigma*W_t
       ln(s_t/s_0) = (mu - 0.5*sigma^2)*t + sigma*W_t
       s_t/s_0 = exp((mu - 0.5*sigma^2)*t + sigma*W_t)
       s_t = s_0*exp((mu - 0.5*sigma^2)*t + sigma*W_t)
  */
  // S_t^(i) = S_0^(i) * exp((mu_i - 0.5 * sigma_i^2) * t + sigma_i * W_t)
  double exact_solution(size_t asset_idx, double t, double W_t) const {
    double mu = config_.mu[asset_idx];
    double sigma = config_.sigma[asset_idx];
    double S0 = config_.S0[asset_idx];

    return S0 * std::exp((mu - 0.5 * sigma * sigma) * t + sigma * W_t);
  }

  // Exact theoretical solution for asset i at maturity T
  double exact_solution_T(size_t asset_idx, double W_T) const {
    return exact_solution(asset_idx, config_.T, W_T);
  }
};
