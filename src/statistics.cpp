#include "../include/statistics.hpp"
#include <cmath>
#include <stdexcept>

Statistics::Statistics(const GBM &model) : model_(model) {}

double
Statistics::compute_mean(const std::vector<double> &terminal_values) const {
  if (terminal_values.empty()) {
    throw std::invalid_argument("Terminal values vector cannot be empty");
  }

  double sum = 0.0;
  for (double val : terminal_values) {
    sum += val;
  }
  return sum / terminal_values.size();
}

double
Statistics::compute_variance(const std::vector<double> &terminal_values) const {
  if (terminal_values.size() < 2) {
    throw std::invalid_argument("Need at least 2 values to compute variance");
  }

  double mean = compute_mean(terminal_values);
  double sum_sq = 0.0;
  for (double val : terminal_values) {
    double diff = val - mean;
    sum_sq += diff * diff;
  }
  // Using population variance (N) to match theoretical formula
  return sum_sq / terminal_values.size();
}

double
Statistics::compute_skewness(const std::vector<double> &terminal_values) const {
  if (terminal_values.size() < 3) {
    throw std::invalid_argument("Need at least 3 values to compute skewness");
  }

  double mean = compute_mean(terminal_values);
  double variance = compute_variance(terminal_values);
  double std_dev = std::sqrt(variance);
  size_t n = terminal_values.size();

  if (std_dev == 0.0) {
    return 0.0;
  }

  double sum_cubed = 0.0;
  for (double val : terminal_values) {
    double diff = (val - mean) / std_dev;
    sum_cubed += diff * diff * diff;
  }
  return sum_cubed / n;
}

double
Statistics::compute_kurtosis(const std::vector<double> &terminal_values) const {
  if (terminal_values.size() < 4) {
    throw std::invalid_argument("Need at least 4 values to compute kurtosis");
  }

  double mean = compute_mean(terminal_values);
  double variance = compute_variance(terminal_values);
  double std_dev = std::sqrt(variance);
  size_t n = terminal_values.size();

  if (std_dev == 0.0) {
    return 0.0;
  }

  double sum_fourth = 0.0;
  for (double val : terminal_values) {
    double diff = (val - mean) / std_dev;
    sum_fourth += diff * diff * diff * diff;
  }
  // Excess kurtosis (subtract 3 so normal distribution = 0)
  return (sum_fourth / n) - 3.0;
}

// E[S_T] = S0 * exp(mu * T)
double Statistics::theoretical_mean(size_t asset_idx) const {
  const auto &cfg = model_.get_config();
  return cfg.S0[asset_idx] * std::exp(cfg.mu[asset_idx] * cfg.T);
}

// Var(S_T) = S0^2 * exp(2*mu*T) * (exp(sigma^2 * T) - 1)
double Statistics::theoretical_variance(size_t asset_idx) const {
  const auto &cfg = model_.get_config();
  double S0 = cfg.S0[asset_idx];
  double mu = cfg.mu[asset_idx];
  double sigma = cfg.sigma[asset_idx];
  double T = cfg.T;

  return S0 * S0 * std::exp(2.0 * mu * T) * (std::exp(sigma * sigma * T) - 1.0);
}

AssetStatistics Statistics::compute_asset_statistics(
    size_t asset_idx, const std::vector<double> &terminal_values) const {
  AssetStatistics stats;
  stats.asset_idx = asset_idx;
  stats.mean_empirical = compute_mean(terminal_values);
  stats.mean_theoretical = theoretical_mean(asset_idx);
  stats.variance_empirical = compute_variance(terminal_values);
  stats.variance_theoretical = theoretical_variance(asset_idx);
  stats.skewness_empirical = compute_skewness(terminal_values);
  stats.kurtosis_empirical = compute_kurtosis(terminal_values);
  return stats;
}
