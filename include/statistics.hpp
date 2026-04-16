#pragma once

#include "gbm.hpp"
#include <vector>

struct AssetStatistics {
  size_t asset_idx;
  double mean_empirical;
  double mean_theoretical;
  double variance_empirical;
  double variance_theoretical;
  double skewness_empirical;
  double kurtosis_empirical;
};

class Statistics {
private:
  const GBM &model_;

public:
  explicit Statistics(const GBM &model);

  // Empirical moment computations over terminal values across N paths
  double compute_mean(const std::vector<double> &terminal_values) const;
  double compute_variance(const std::vector<double> &terminal_values) const;
  double compute_skewness(const std::vector<double> &terminal_values) const;
  double compute_kurtosis(const std::vector<double> &terminal_values) const;

  // Theoretical GBM moments
  // E[S_T] = S0 * exp(mu * T)
  double theoretical_mean(size_t asset_idx) const;
  // Var(S_T) = S0^2 * exp(2*mu*T) * (exp(sigma^2 * T) - 1)
  double theoretical_variance(size_t asset_idx) const;

  // Computes all stats for a single asset given its terminal values across
  // paths
  AssetStatistics
  compute_asset_statistics(size_t asset_idx,
                           const std::vector<double> &terminal_values) const;
};
