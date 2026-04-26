#pragma once

#include <vector>

namespace hedging {

struct HedgingMCConfig {
  double S0;             // Initial spot
  double mu;             // Real-world drift for simulation
  double sigma_sim;      // Real-world vol for simulation
  double K;              // Strike
  double r;              // Risk-free rate
  double sigma_hedge;    // Vol assumed by hedger (can differ from sigma_sim)
  double T;              // Maturity
  size_t num_steps;      // Rebalancing frequency
  size_t num_paths;      // Monte Carlo paths
  unsigned int seed;
  bool is_call;
  double cost_per_unit;  // Transaction cost (default 0.0)
};

struct HedgingMCSummary {
  double mean_pnl;
  double std_pnl;
  double min_pnl;
  double max_pnl;
  
  // Attribution averages across all paths
  double mean_theta;
  double mean_gamma;
  double mean_financing;
  double mean_txn_cost;
  double mean_residual;
  
  std::vector<double> all_pnls;  // raw P&L per path
};

class HedgingRunner {
public:
  explicit HedgingRunner(const HedgingMCConfig &config);
  HedgingMCSummary run() const;
private:
  HedgingMCConfig config_;
};

} // namespace hedging
