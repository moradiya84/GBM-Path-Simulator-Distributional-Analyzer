#pragma once

#include <vector>

namespace hedging {

// Per-step P&L breakdown using Taylor expansion of option value:
//   dV ≈ Δ·dS + ½Γ·(dS)² + Θ·dt
//
// Since we are short the option and hold Δ shares:
//   theta_pnl    = −Θ·dt      (positive = we earn time decay)
//   gamma_pnl    = −½Γ·(dS)²  (negative = cost of being short convexity)
//   financing    = r·cash·dt   (interest on cash balance)
//   txn_cost     = cost_per_unit · |trade_qty| · S
//   residual     = actual_step_pnl − (theta + gamma + financing − txn_cost)
struct StepAttribution {
  double theta_pnl;
  double gamma_pnl;
  double financing_pnl;
  double transaction_cost;
  double residual;
};

// Path-level totals aggregated from all steps, plus the raw terminal P&L
struct PathAttribution {
  double terminal_pnl;
  double total_theta;
  double total_gamma;
  double total_financing;
  double total_transaction_cost;
  double total_residual;
  std::vector<StepAttribution> steps;
};

} // namespace hedging
