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

PathAttribution
DeltaHedger::backtest_path_attributed(const TimeGrid &grid,
                                      const std::vector<double> &spot_path,
                                      double cost_per_unit) const {
  size_t num_steps = grid.num_steps();
  if (spot_path.size() != num_steps + 1) {
    throw std::invalid_argument("spot_path size must precisely match the time "
                                "grid points (num_steps + 1).");
  }

  double dt = grid.dt();
  double T = grid.T();

  PathAttribution attr;
  attr.total_theta = 0.0;
  attr.total_gamma = 0.0;
  attr.total_financing = 0.0;
  attr.total_transaction_cost = 0.0;
  attr.total_residual = 0.0;
  attr.steps.reserve(num_steps);

  // 1. Initial State (t = 0)
  double S0 = spot_path[0];
  bs::BlackScholes bs0(S0, K_, r_, sigma_hedge_, T);

  double cash = is_call_ ? bs0.call_price() : bs0.put_price();
  double position = is_call_ ? bs0.call_delta() : bs0.put_delta();

  // Transaction cost on initial hedge setup
  double initial_trade_qty = position; // assumed starting from 0
  double initial_txn_cost = cost_per_unit * std::abs(initial_trade_qty) * S0;

  cash -= (position * S0) + initial_txn_cost;
  attr.total_transaction_cost += initial_txn_cost;

  // Track previous step values for attribution
  double prev_S = S0;
  double prev_cash = cash;
  double prev_theta = is_call_ ? bs0.call_theta() : bs0.put_theta();
  double prev_gamma = bs0.gamma(); // Same for call/put

  // 2. Rebalancing Loop (t_1 to t_{N-1} AND maturity t_N)
  for (size_t i = 1; i <= num_steps; ++i) {
    StepAttribution step_attr;

    // Financing PnL over the dt step
    // Using simple compounding for exact attribution decoupling,
    // real cash uses continuous discounting.
    // e^(r*dt) ≈ 1 + r*dt. For attribution of exactly what is accrued:
    double accrued_cash = prev_cash * std::exp(r_ * dt);
    step_attr.financing_pnl = accrued_cash - prev_cash;

    // Market movement
    double S_t = spot_path[i];
    double dS = S_t - prev_S;

    // Greek PnL attribution
    // We are short the option. If theta is positive (time decay), we gain it.
    // The BS theta is an annual rate. Since we are SHORT the option,
    // the option value drops by Theta*dt, meaning we MAKE Theta*dt.
    // (Note: BS theta returned is often the derivative wrt time (e.g.,
    // negative). Let's rely on standard Taylor: dV = Theta*dt. Short option ->
    // -dV = -Theta*dt.)
    step_attr.theta_pnl = -prev_theta * dt;

    // Gamma PnL: short option means short gamma (-0.5*Gamma*dS^2)
    step_attr.gamma_pnl = -0.5 * prev_gamma * (dS * dS);

    double t = grid.get_times()[i];
    double time_to_maturity = T - t;

    double trade_qty = 0.0;
    step_attr.transaction_cost = 0.0;

    double actual_step_pnl = 0.0;

    if (i < num_steps) {
      // Rebalance
      bs::BlackScholes bs_current(S_t, K_, r_, sigma_hedge_, time_to_maturity);
      double new_target_delta =
          is_call_ ? bs_current.call_delta() : bs_current.put_delta();

      trade_qty = new_target_delta - position;
      step_attr.transaction_cost = cost_per_unit * std::abs(trade_qty) * S_t;

      // Update cash with trade
      cash = accrued_cash - (trade_qty * S_t) - step_attr.transaction_cost;

      // Actual PnL measure: change in portfolio value
      // Portfolio Value = Cash + Position * S - Option_Value
      double old_port_val =
          prev_cash + position * prev_S -
          (is_call_ ? bs0.call_price()
                    : bs0.put_price()); // approximation for tracking
      // To compute exact actual step PnL, it's easier to determine the total
      // change in Portfolio MTM.
      double current_option_price =
          is_call_ ? bs_current.call_price() : bs_current.put_price();
      // Portfolio before rebalancing (at S_t)
      double port_val_before_rebalance =
          accrued_cash + position * S_t - current_option_price;

      // For accurate historical tracking of the Option Price at previous step
      // (we need to bring it forward) Since it's complex to track exact Option
      // MTM step-by-step cleanly here, the true Greek definition says:
      // d(Portfolio) = d(Cash + Delta*S - Call). Let's compute actual_step_pnl
      // as Portfolio_t - Portfolio_{t-1}. Portfolio_{t-1} pre-transition: We
      // need bs_prev option price.
      bs::BlackScholes bs_prev(prev_S, K_, r_, sigma_hedge_,
                               T - grid.get_times()[i - 1]);
      double prev_opt_price =
          is_call_ ? bs_prev.call_price() : bs_prev.put_price();
      double prev_port_val = prev_cash + position * prev_S - prev_opt_price;
      actual_step_pnl = port_val_before_rebalance - prev_port_val;

      // Update state for next step
      position = new_target_delta;
      prev_S = S_t;
      prev_cash = cash; // Note: this is cash AFTER rebalancing
      prev_theta = is_call_ ? bs_current.call_theta() : bs_current.put_theta();
      prev_gamma = bs_current.gamma();
    } else {
      // Maturity
      // Unwind all stock
      trade_qty = -position;
      step_attr.transaction_cost = cost_per_unit * std::abs(trade_qty) * S_t;

      double payoff = 0.0;
      if (is_call_) {
        bs::CallPayoff c(K_);
        payoff = c(S_t);
      } else {
        bs::PutPayoff p(K_);
        payoff = p(S_t);
      }

      cash =
          accrued_cash + (position * S_t) - step_attr.transaction_cost - payoff;

      // Step PnL at maturity
      bs::BlackScholes bs_prev(prev_S, K_, r_, sigma_hedge_,
                               T - grid.get_times()[i - 1]);
      double prev_opt_price =
          is_call_ ? bs_prev.call_price() : bs_prev.put_price();
      double prev_port_val = prev_cash + position * prev_S - prev_opt_price;

      double port_val_final = cash; // No stock, no option liability left
      actual_step_pnl = port_val_final - prev_port_val;
    }

    step_attr.residual = actual_step_pnl -
                         (step_attr.theta_pnl + step_attr.gamma_pnl +
                          step_attr.financing_pnl - step_attr.transaction_cost);

    attr.steps.push_back(step_attr);

    attr.total_theta += step_attr.theta_pnl;
    attr.total_gamma += step_attr.gamma_pnl;
    attr.total_financing += step_attr.financing_pnl;
    attr.total_transaction_cost += step_attr.transaction_cost;
    attr.total_residual += step_attr.residual;
  }

  attr.terminal_pnl = cash;
  return attr;
}

} // namespace hedging
