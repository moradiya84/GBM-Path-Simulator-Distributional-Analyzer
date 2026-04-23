#include "../../include/bs/black_scholes.hpp"
#include "../../include/bs/normal.hpp"

#include <cmath>
#include <stdexcept>

namespace bs
{

BlackScholes::BlackScholes(double S, double K, double r, double sigma, double T)
    : S_(S), K_(K), r_(r), sigma_(sigma), T_(T)
{
  if (S <= 0.0)
    throw std::invalid_argument("BlackScholes: spot price S must be > 0");
  if (K <= 0.0)
    throw std::invalid_argument("BlackScholes: strike K must be > 0");
  if (sigma <= 0.0)
    throw std::invalid_argument("BlackScholes: volatility sigma must be > 0");
  if (T <= 0.0)
    throw std::invalid_argument("BlackScholes: time-to-expiry T must be > 0");

  // Pre-compute d1, d2, and discount factor once for all pricing calls
  //   d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
  //   d2 = d1 - sigma*sqrt(T)
  double sqrt_T = std::sqrt(T_);
  d1_ = (std::log(S_ / K_) + (r_ + 0.5 * sigma_ * sigma_) * T_) / (sigma_ * sqrt_T);
  d2_ = d1_ - sigma_ * sqrt_T;

  // Continuous discount factor
  discount_ = std::exp(-r_ * T_);
}

// C = S*N(d1) - K*e^(-rT)*N(d2)
double BlackScholes::call_price() const
{
  return S_ * normal_cdf(d1_) - K_ * discount_ * normal_cdf(d2_);
}

// P = K*e^(-rT)*N(-d2) - S*N(-d1)
double BlackScholes::put_price() const
{
  return K_ * discount_ * normal_cdf(-d2_) - S_ * normal_cdf(-d1_);
}

// DC = e^(-rT)*N(d2)
double BlackScholes::digital_call_price() const
{
  return discount_ * normal_cdf(d2_);
}

// DP = e^(-rT)*N(-d2)
double BlackScholes::digital_put_price() const
{
  return discount_ * normal_cdf(-d2_);
}

// ─── Greeks ──────────────────────────────────────────────────────────────────

// Call Delta = N(d1)
double BlackScholes::call_delta() const
{
  return normal_cdf(d1_);
}

// Put Delta = N(d1) - 1
double BlackScholes::put_delta() const
{
  return normal_cdf(d1_) - 1.0;
}

// Gamma = phi(d1) / (S * sigma * sqrt(T))
double BlackScholes::gamma() const
{
  return normal_pdf(d1_) / (S_ * sigma_ * std::sqrt(T_));
}

// Call Theta = - (S * phi(d1) * sigma) / (2 * sqrt(T)) - r * K * e^(-rT) * N(d2)
double BlackScholes::call_theta() const
{
  double term1 = -(S_ * normal_pdf(d1_) * sigma_) / (2.0 * std::sqrt(T_));
  double term2 = r_ * K_ * discount_ * normal_cdf(d2_);
  return term1 - term2;
}

// Put Theta = - (S * phi(d1) * sigma) / (2 * sqrt(T)) + r * K * e^(-rT) * N(-d2)
double BlackScholes::put_theta() const
{
  double term1 = -(S_ * normal_pdf(d1_) * sigma_) / (2.0 * std::sqrt(T_));
  double term2 = r_ * K_ * discount_ * normal_cdf(-d2_);
  return term1 + term2;
}

// Vega = S * phi(d1) * sqrt(T)
double BlackScholes::vega() const
{
  return S_ * normal_pdf(d1_) * std::sqrt(T_);
}

// Call Rho = K * T * e^(-rT) * N(d2)
double BlackScholes::call_rho() const
{
  return K_ * T_ * discount_ * normal_cdf(d2_);
}

// Put Rho = -K * T * e^(-rT) * N(-d2)
double BlackScholes::put_rho() const
{
  return -K_ * T_ * discount_ * normal_cdf(-d2_);
}

} // namespace bs
