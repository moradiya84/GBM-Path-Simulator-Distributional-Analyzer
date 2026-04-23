#pragma once

namespace bs
{

// Closed-form Black-Scholes pricing engine for European options.
//
// Vanilla Call:    C  = S·N(d1) − K·e^(−rT)·N(d2)
// Vanilla Put:     P  = K·e^(−rT)·N(−d2) − S·N(−d1)
// Digital Call:    DC = e^(−rT)·N(d2)
// Digital Put:     DP = e^(−rT)·N(−d2)
//
// where:
//   d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
//   d2 = d1 − σ·√T
//
// Put-call parity:         C − P  = S − K·e^(−rT)
// Digital put-call parity: DC + DP = e^(−rT)
class BlackScholes
{
public:
  // S     : current spot price  (must be > 0)
  // K     : strike price        (must be > 0)
  // r     : risk-free rate      (continuously compounded)
  // sigma : volatility          (must be > 0)
  // T     : time-to-expiry in years (must be > 0)
  BlackScholes(double S, double K, double r, double sigma, double T);

  // ─── Prices ──────────────────────────────────────────────────────────────

  double call_price()         const;
  double put_price()          const;
  double digital_call_price() const;
  double digital_put_price()  const;

  // ─── Intermediate values (exposed for inspection / greeks later) ────────

  double d1() const { return d1_; }
  double d2() const { return d2_; }

  // Discount factor e^(−rT)
  double discount() const { return discount_; }

private:
  double S_;
  double K_;
  double r_;
  double sigma_;
  double T_;

  // Pre-computed on construction to avoid redundant work across price calls
  double d1_;
  double d2_;
  double discount_; // e^(-r*T)
};

} // namespace bs
