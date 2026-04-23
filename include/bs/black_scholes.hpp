#pragma once

namespace bs {

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
class BlackScholes {
public:
  // S     : current spot price  (must be > 0)
  // K     : strike price        (must be > 0)
  // r     : risk-free rate      (continuously compounded)
  // sigma : volatility          (must be > 0)
  // T     : time-to-expiry in years (must be > 0)
  BlackScholes(double S, double K, double r, double sigma, double T);

  // ─── Prices ──────────────────────────────────────────────────────────────

  double call_price() const;
  double put_price() const;
  double digital_call_price() const;
  double digital_put_price() const;

  // ─── Greeks ──────────────────────────────────────────────────────────────

  /*
   * Delta measures the rate of change of the theoretical option value with
     respect to changes in the underlying asset's price.
     (means partial derivative of option value with respect to underlying price)
     del(c(t,x))/del(x)
   * Role: Used for directional hedging. A delta-neutral portfolio is immune
     to small movements in the underlying asset.
   */
  double call_delta() const;
  double put_delta() const;

  /*
   * Gamma measures the rate of change in the delta with respect to changes
     in the underlying price. (Second derivative of value w.r.t underlying
   price).
   * Note: Gamma is identical for both vanilla calls and puts.
   * Role: Measures the convexity of the option. High gamma means delta changes
     rapidly, requiring frequent re-hedging to maintain delta neutrality.
   */
  double gamma() const;

  /*
   * Theta measures the sensitivity of the value of the option to the passage
     of time (time decay).
   * Note: This returns the annualized abstract continuous theta.
   * Role: Represents the daily/annual cost of holding an option due to the
     shrinking time value. Options naturally lose value as expiration
   approaches.
   */
  double call_theta() const;
  double put_theta() const;

  /*
   * Vega measures sensitivity to volatility. It is the derivative of the
     option value with respect to the volatility of the underlying asset.
   * Note: Vega is identical for both vanilla calls and puts.
   * Role: Helps traders manage exposure to implied volatility changes.
   */
  double vega() const;

  /*
   * Rho measures sensitivity to the interest rate. It is the derivative of the
     option value with respect to the risk-free interest rate.
   * Role: Used to hedge interest rate risk, typically only significant for
     long-dated options (LEAPS).
   */
  double call_rho() const;
  double put_rho() const;

  // Intermediate values (exposed for inspection)
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
