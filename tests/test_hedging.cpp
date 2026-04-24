#include "../include/bs/black_scholes.hpp"
#include "../include/hedging/delta_hedger.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

// Utility to check floating point equality
bool is_close(double a, double b, double tol = 1e-4) {
  return std::abs(a - b) < tol;
}

// Test 1: Manual Step-by-Step Cashflow Tracking
void test_cash_flow_tracking() {
  double K = 100.0;
  double r = 0.05;
  double sigma = 0.2;
  double T = 1.0;
  size_t num_steps = 2; // 3 points: t=0, 0.5, 1.0
  double dt = T / num_steps;

  hedging::DeltaHedger hedger(K, r, sigma, true /* short call */);
  TimeGrid grid(T, num_steps);

  // Hardcode a spot path spanning 3 points
  std::vector<double> spot_path = {100.0, 110.0, 105.0};

  hedging::HedgingResult result = hedger.backtest_path(grid, spot_path);

  // Calculate manually to verify
  // 1. Initial State
  bs::BlackScholes bs0(100.0, K, r, sigma, T);
  double initial_cash = bs0.call_price();
  double initial_delta = bs0.call_delta();

  double expected_cash_0 = initial_cash - (initial_delta * 100.0);
  assert(is_close(result.stock_positions[0], initial_delta));
  assert(is_close(result.cash_account[0], expected_cash_0));

  // 2. Step 1 (t=0.5)
  double accrued_cash_1 = expected_cash_0 * std::exp(r * dt);
  bs::BlackScholes bs1(110.0, K, r, sigma, 0.5);
  double delta_1 = bs1.call_delta();
  double trade_qty_1 = delta_1 - initial_delta;
  double expected_cash_1 = accrued_cash_1 - (trade_qty_1 * 110.0);

  assert(is_close(result.stock_positions[1], delta_1));
  assert(is_close(result.cash_account[1], expected_cash_1));

  // 3. Step 2 (maturity t=1.0)
  double accrued_cash_2 = expected_cash_1 * std::exp(r * dt);
  // Unwind stock
  double final_cash_before_payoff = accrued_cash_2 + (delta_1 * 105.0);
  // Payoff for Call at S=105, K=100 is 5.0
  double payoff = 5.0;
  double expected_cash_2 = final_cash_before_payoff - payoff;

  assert(is_close(result.stock_positions[2], 0.0));
  assert(is_close(result.cash_account[2], expected_cash_2));
  assert(is_close(result.terminal_pnl, expected_cash_2));

  std::cout << "[PASS] test_cash_flow_tracking\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "Starting Hedging Backtester Tests...\n";
  std::cout << "========================================\n";

  test_cash_flow_tracking();

  std::cout << "========================================\n";
  std::cout << "All Hedging Backtester tests passed.\n";
  std::cout << "========================================\n";
  return 0;
}
