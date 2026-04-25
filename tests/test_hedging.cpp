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

// Test 2: Verify P&L attribution identity
void test_attribution_sums_to_pnl() {
  double K = 100.0;
  double r = 0.05;
  double sigma = 0.2;
  double T = 1.0;
  size_t num_steps = 10;
  
  hedging::DeltaHedger hedger(K, r, sigma, true);
  TimeGrid grid(T, num_steps);

  // Simple linear path
  std::vector<double> spot_path(num_steps + 1);
  for (size_t i = 0; i <= num_steps; ++i) spot_path[i] = 100.0 + i;

  hedging::PathAttribution attr = hedger.backtest_path_attributed(grid, spot_path, 0.0);

  // Sum components
  double sum_components = attr.total_theta + attr.total_gamma + 
                          attr.total_financing - attr.total_transaction_cost + 
                          attr.total_residual;

  // The sum of all attribution components must EXACTLY equal the terminal P&L change
  // Note: terminal_pnl is the raw cash. We must compare against the difference 
  // from initial portfolio value, but the way we set it up in the code,
  // terminal_pnl is exactly cash, and the sum of step actual_PnLs equals 
  // (Final_Portfolio - Initial_Portfolio). 
  // Our initial portfolio is Cash + Delta*S - Option. 
  // Oh wait, actual_step_pnl sum equals Final_Portfolio - Initial_Portfolio.
  // Final_Portfolio = attr.terminal_pnl (since stock and option are 0).
  // Initial_Portfolio = Cash_0 + Pos_0 * S_0 - Opt_0 
  // But wait, Cash_0 was set to Premium - Pos_0 * S_0!
  // So Initial_Portfolio = (Premium - Pos_0*S_0) + Pos_0*S_0 - Premium = 0 !
  // Since Initial Portfolio is 0, the sum of all step PnLs IS EXACTLY the final cash!
  assert(is_close(sum_components, attr.terminal_pnl, 1e-6));
  
  std::cout << "[PASS] test_attribution_sums_to_pnl\n";
}

#include "../include/hedging/hedging_runner.hpp"

// Test 3: MC Mean PnL converges to ~0 with matching vol
void test_mc_mean_pnl_near_zero() {
  hedging::HedgingMCConfig config;
  config.S0 = 100.0;
  config.mu = 0.05;
  config.sigma_sim = 0.2;
  config.K = 100.0;
  config.r = 0.05;
  config.sigma_hedge = 0.2; // MATCHES!
  config.T = 1.0;
  config.num_steps = 50;  // Frequent rebalancing
  config.num_paths = 1000;
  config.seed = 42;
  config.is_call = true;
  config.cost_per_unit = 0.0;

  hedging::HedgingRunner runner(config);
  hedging::HedgingMCSummary summary = runner.run();

  // The mean PnL should be very small (close to 0) since we delta hedged
  assert(std::abs(summary.mean_pnl) < 1.0); 
  std::cout << "[PASS] test_mc_mean_pnl_near_zero (Mean PnL = " << summary.mean_pnl << ")\n";
}

// Test 4: Volatility Mismatch
void test_vol_mismatch_creates_pnl() {
  hedging::HedgingMCConfig config;
  config.S0 = 100.0;
  config.mu = 0.05;
  config.sigma_sim = 0.3; // Realized vol is HIGH
  config.K = 100.0;
  config.r = 0.05;
  config.sigma_hedge = 0.2; // Hedger priced it using LOW vol (sold it too cheap)
  config.T = 1.0;
  config.num_steps = 100; 
  config.num_paths = 500;
  config.seed = 42;
  config.is_call = true;
  config.cost_per_unit = 0.0;

  hedging::HedgingRunner runner(config);
  hedging::HedgingMCSummary summary = runner.run();

  // If realized vol > implied vol and we sold the option, we should lose money systematically
  // (Gamma bleed is larger than Theta earned)
  assert(summary.mean_pnl < -1.0); 
  std::cout << "[PASS] test_vol_mismatch_creates_pnl (Sys Loss = " << summary.mean_pnl << ")\n";
}

// Test 5: Transaction Costs
void test_transaction_costs_reduce_pnl() {
  hedging::HedgingMCConfig config;
  config.S0 = 100.0;
  config.mu = 0.05;
  config.sigma_sim = 0.2;
  config.K = 100.0;
  config.r = 0.05;
  config.sigma_hedge = 0.2; 
  config.T = 1.0;
  config.num_steps = 50; 
  config.num_paths = 100;
  config.seed = 42;
  config.is_call = true;
  
  // Run without costs
  config.cost_per_unit = 0.0;
  double pnl_no_cost = hedging::HedgingRunner(config).run().mean_pnl;

  // Run with costs
  config.cost_per_unit = 0.01; // 1 cent per share traded
  double pnl_with_cost = hedging::HedgingRunner(config).run().mean_pnl;

  assert(pnl_with_cost < pnl_no_cost);
  std::cout << "[PASS] test_transaction_costs_reduce_pnl (" << pnl_with_cost << " < " << pnl_no_cost << ")\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "Starting Hedging Backtester Tests...\n";
  std::cout << "========================================\n";

  test_cash_flow_tracking();
  test_attribution_sums_to_pnl();
  test_mc_mean_pnl_near_zero();
  test_vol_mismatch_creates_pnl();
  test_transaction_costs_reduce_pnl();

  std::cout << "========================================\n";
  std::cout << "All Hedging Backtester tests passed.\n";
  std::cout << "========================================\n";
  return 0;
}
