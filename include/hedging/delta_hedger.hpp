#pragma once

#include "../time_grid.hpp"
#include <vector>

namespace hedging {

// Results of a single path's discrete hedging over time
struct HedgingResult {
  double terminal_pnl; // Cash balance after all unwinds at maturity
  std::vector<double> stock_positions; // Stock quantity held at each time step
  std::vector<double> cash_account;    // Cash account balance at each time step
};

// DeltaHedger simulates dynamic hedging of a short option position.
// It tracks cash flow and stock positions across discrete time intervals.
class DeltaHedger {
public:
  // K: Strike price of the option
  // r: Risk-free rate (for both cash accrual and Black-Scholes Delta pricing)
  // sigma_hedge: Implied volatility assumed by the hedger
  // is_call: true if hedging a short Call, false if hedging a short Put
  DeltaHedger(double K, double r, double sigma_hedge, bool is_call);

  // Simulates rolling the hedge across the given discrete time grid.
  // spot_path: the realized stock prices (should have size == grid.num_steps()
  // + 1)
  HedgingResult backtest_path(const TimeGrid &grid,
                              const std::vector<double> &spot_path) const;

  // Runs the same simulation but performs full P&L decomposition into Greeks.
  // cost_per_unit allows applying a proportional transaction cost per share
  // traded.

private:
  double K_;
  double r_;
  double sigma_hedge_;
  bool is_call_;
};

} // namespace hedging
