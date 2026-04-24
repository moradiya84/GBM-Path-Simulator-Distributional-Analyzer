#include "../include/bs/black_scholes.hpp"
#include "../include/bs/payoff.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

// ─── Helpers ─────────────────────────────────────────────────────────────────

bool is_close(double a, double b, double tol = 1e-10)
{
  return std::abs(a - b) < tol;
}

bool is_close_rel(double a, double b, double rel_tol = 1e-4)
{
  return std::abs(a - b) / std::max(std::abs(b), 1e-12) < rel_tol;
}

// ─── Test 1: Put-call parity ─────────────────────────────────────────────────
// C - P = S - K*e^(-rT)  (must hold to machine precision)
void test_put_call_parity()
{
  bs::BlackScholes bs(100.0, 105.0, 0.05, 0.2, 1.0);

  double C = bs.call_price();
  double P = bs.put_price();
  double parity_lhs = C - P;
  double parity_rhs = 100.0 - 105.0 * bs.discount();

  assert(is_close(parity_lhs, parity_rhs));
  std::cout << "[PASS] test_put_call_parity"
            << "  C=" << C << "  P=" << P
            << "  diff=" << std::abs(parity_lhs - parity_rhs) << "\n";
}

// ─── Test 2: Digital put-call parity ─────────────────────────────────────────
// DC + DP = e^(-rT)  (must hold to machine precision)
void test_digital_parity()
{
  bs::BlackScholes bs(100.0, 100.0, 0.03, 0.25, 0.5);

  double DC = bs.digital_call_price();
  double DP = bs.digital_put_price();
  double sum = DC + DP;

  assert(is_close(sum, bs.discount()));
  std::cout << "[PASS] test_digital_parity"
            << "  DC=" << DC << "  DP=" << DP
            << "  sum=" << sum << "  e^(-rT)=" << bs.discount() << "\n";
}

// ─── Test 3: Known reference price ───────────────────────────────────────────
// ATM 1-year call: S=100, K=100, r=0.05, sigma=0.2, T=1
// Textbook result: C ≈ 10.4506
void test_reference_call_price()
{
  bs::BlackScholes bs(100.0, 100.0, 0.05, 0.2, 1.0);
  double C = bs.call_price();

  // Tolerance 0.001 (i.e. within a fraction of a cent)
  assert(is_close(C, 10.4506, 1e-3));
  std::cout << "[PASS] test_reference_call_price"
            << "  C=" << C << "  (expected ≈ 10.4506)\n";
}

// ─── Test 4: Payoff evaluation ───────────────────────────────────────────────
// Verify each payoff class returns the correct value for ITM / ATM / OTM spots
void test_payoff_evaluation()
{
  const double K = 100.0;

  bs::CallPayoff        call(K);
  bs::PutPayoff         put(K);
  bs::DigitalCallPayoff dc(K);
  bs::DigitalPutPayoff  dp(K);

  // ITM call / OTM put (S = 120)
  assert(is_close(call(120.0), 20.0));
  assert(is_close(put(120.0),   0.0));
  assert(is_close(dc(120.0),    1.0));
  assert(is_close(dp(120.0),    0.0));

  // OTM call / ITM put (S = 80)
  assert(is_close(call(80.0),  0.0));
  assert(is_close(put(80.0),  20.0));
  assert(is_close(dc(80.0),    0.0));
  assert(is_close(dp(80.0),    1.0));

  // ATM (S = K = 100): call payoff 0, put payoff 0, digital call fires, digital put does NOT
  assert(is_close(call(100.0), 0.0));
  assert(is_close(put(100.0),  0.0));
  assert(is_close(dc(100.0),   1.0)); // S >= K
  assert(is_close(dp(100.0),   0.0)); // S is NOT < K

  std::cout << "[PASS] test_payoff_evaluation (ITM / OTM / ATM)\n";
}

// ─── Test 5: Edge cases ───────────────────────────────────────────────────────

// Deep ITM call: S >> K → price ≈ S - K*e^(-rT)
void test_deep_itm_call()
{
  bs::BlackScholes bs(1000.0, 10.0, 0.05, 0.2, 1.0);
  double C = bs.call_price();
  double intrinsic = 1000.0 - 10.0 * bs.discount();
  // Should be very close to intrinsic for deep ITM
  assert(is_close_rel(C, intrinsic, 1e-3));
  std::cout << "[PASS] test_deep_itm_call  C=" << C << "  intrinsic=" << intrinsic << "\n";
}

// Deep OTM call: S << K → price ≈ 0
void test_deep_otm_call()
{
  bs::BlackScholes bs(10.0, 1000.0, 0.05, 0.2, 1.0);
  double C = bs.call_price();
  assert(C < 1e-10);
  std::cout << "[PASS] test_deep_otm_call  C=" << C << "  (expected ≈ 0)\n";
}

// Very short expiry: T → 0+, call on ITM option ≈ intrinsic
void test_short_expiry()
{
  bs::BlackScholes bs(110.0, 100.0, 0.05, 0.3, 1e-6);
  double C = bs.call_price();
  double intrinsic = 110.0 - 100.0; // ≈ 10 (discount negligible)
  assert(is_close_rel(C, intrinsic, 0.01)); // within 1%
  std::cout << "[PASS] test_short_expiry  C=" << C << "  intrinsic=" << intrinsic << "\n";
}

// ─── Test 6: Analytical Greeks vs Finite Difference ───────────────────────────
void test_analytical_greeks()
{
  double S = 100.0;
  double K = 100.0;
  double r = 0.05;
  double sigma = 0.2;
  double T = 1.0;

  bs::BlackScholes bs_base(S, K, r, sigma, T);

  double hS = 1e-4; // Finite difference bump for Spot
  bs::BlackScholes bs_up(S + hS, K, r, sigma, T);
  bs::BlackScholes bs_dn(S - hS, K, r, sigma, T);

  double hSigma = 1e-4; // For Vega
  bs::BlackScholes bs_vol_up(S, K, r, sigma + hSigma, T);
  bs::BlackScholes bs_vol_dn(S, K, r, sigma - hSigma, T);

  double hT = 1e-5; // For Theta
  // Note on Theta FD: time steps forward, meaning T decreases
  bs::BlackScholes bs_time_up(S, K, r, sigma, T - hT);
  
  double hr = 1e-4; // For Rho
  bs::BlackScholes bs_r_up(S, K, r + hr, sigma, T);
  bs::BlackScholes bs_r_dn(S, K, r - hr, sigma, T);

  // 1. Delta (dC/dS)
  double fd_call_delta = (bs_up.call_price() - bs_dn.call_price()) / (2.0 * hS);
  double fd_put_delta  = (bs_up.put_price() - bs_dn.put_price()) / (2.0 * hS);
  assert(is_close(bs_base.call_delta(), fd_call_delta, 1e-5));
  assert(is_close(bs_base.put_delta(), fd_put_delta, 1e-5));
  // Identity verify
  assert(is_close(bs_base.call_delta() - bs_base.put_delta(), 1.0));

  // 2. Gamma (d2C/dS2)
  double fd_gamma = (bs_up.call_price() - 2.0 * bs_base.call_price() + bs_dn.call_price()) / (hS * hS);
  assert(is_close(bs_base.gamma(), fd_gamma, 1e-5));

  // 3. Vega (dC/dSigma)
  double fd_vega = (bs_vol_up.call_price() - bs_vol_dn.call_price()) / (2.0 * hSigma);
  assert(is_close(bs_base.vega(), fd_vega, 1e-4));

  // 4. Theta (-dC/dT, note negative sign for time decay)
  double fd_call_theta = (bs_time_up.call_price() - bs_base.call_price()) / hT;
  double fd_put_theta = (bs_time_up.put_price() - bs_base.put_price()) / hT;
  assert(is_close_rel(bs_base.call_theta(), fd_call_theta, 1e-3));
  assert(is_close_rel(bs_base.put_theta(), fd_put_theta, 1e-3));

  // 5. Rho (dC/dr)
  double fd_call_rho = (bs_r_up.call_price() - bs_r_dn.call_price()) / (2.0 * hr);
  double fd_put_rho = (bs_r_up.put_price() - bs_r_dn.put_price()) / (2.0 * hr);
  assert(is_close(bs_base.call_rho(), fd_call_rho, 1e-4));
  assert(is_close(bs_base.put_rho(), fd_put_rho, 1e-4));

  std::cout << "[PASS] test_analytical_greeks (Delta, Gamma, Vega, Theta, Rho match FD limits)\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
  std::cout << "========================================\n";
  std::cout << "Starting Black-Scholes Tests...\n";
  std::cout << "========================================\n";

  test_put_call_parity();
  test_digital_parity();
  test_reference_call_price();
  test_payoff_evaluation();
  test_deep_itm_call();
  test_deep_otm_call();
  test_short_expiry();
  test_analytical_greeks();

  std::cout << "========================================\n";
  std::cout << "All Black-Scholes tests passed.\n";
  std::cout << "========================================\n";

  return 0;
}
