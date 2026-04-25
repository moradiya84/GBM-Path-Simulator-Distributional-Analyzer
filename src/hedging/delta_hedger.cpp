#include "../../include/hedging/delta_hedger.hpp"
#include "../../include/bs/black_scholes.hpp"
#include "../../include/bs/payoff.hpp"

#include <cmath>
#include <stdexcept>

namespace hedging {

DeltaHedger::DeltaHedger(double K, double r, double sigma_hedge, bool is_call)
    : K_(K), r_(r), sigma_hedge_(sigma_hedge), is_call_(is_call) {}

HedgingResult
DeltaHedger::backtest_path(const TimeGrid &grid,
                           const std::vector<double> &spot_path) const {
  size_t num_steps = grid.num_steps();
  if (spot_path.size() != num_steps + 1) {
    throw std::invalid_argument("spot_path size must precisely match the time "
                                "grid points (num_steps + 1).");
  }

  double dt = grid.dt();
  double T = grid.T();

  HedgingResult result;
  result.stock_positions.reserve(num_steps + 1);
  result.cash_account.reserve(num_steps + 1);

  // 1. Initial State (t = 0)
  double S0 = spot_path[0];
  bs::BlackScholes bs0(S0, K_, r_, sigma_hedge_, T);

  // We are short the option. We collect the premium upfront.
  double cash = is_call_ ? bs0.call_price() : bs0.put_price();

  // Initial delta is computed. Since we are short the option, our net option
  // delta is -OptionDelta. We need to buy +OptionDelta units of stock to become
  // delta-neutral.
  double position = is_call_ ? bs0.call_delta() : bs0.put_delta();

  // Buy the shares, deducting cost from cash account
  cash -= position * S0;

  result.stock_positions.push_back(position);
  result.cash_account.push_back(cash);

  // 2. Rebalancing Loop (t_1 to t_{N-1})
  for (size_t i = 1; i < num_steps; ++i) {
    // A step of time has passed. Accrue interest on standard cash balance.
    // cash *= e^(r * dt)
    cash *= std::exp(r_ * dt);

    double S_t = spot_path[i];
    double t = grid.get_times()[i];
    double time_to_maturity = T - t;

    // Reprice to get the new delta
    bs::BlackScholes bs_current(S_t, K_, r_, sigma_hedge_, time_to_maturity);
    double new_target_delta =
        is_call_ ? bs_current.call_delta() : bs_current.put_delta();

    // Rebalance
    double trade_qty = new_target_delta - position;
    cash -= trade_qty * S_t;
    position = new_target_delta;

    result.stock_positions.push_back(position);
    result.cash_account.push_back(cash);
  }

  // 3. Maturity (t = T)
  // Accrue final step interest
  cash *= std::exp(r_ * dt);

  double S_T = spot_path[num_steps];

  // Unwind stock: sell everything at S_T
  cash += position * S_T;
  position = 0.0;

  // Option reaches maturity and exercises if ITM. We are short, so we payout.
  double payoff = 0.0;
  if (is_call_) {
    bs::CallPayoff c(K_);
    payoff = c(S_T);
  } else {
    bs::PutPayoff p(K_);
    payoff = p(S_T);
  }

  cash -= payoff;

  result.stock_positions.push_back(position);
  result.cash_account.push_back(cash);
  result.terminal_pnl = cash;

  return result;
}

} // namespace hedging
