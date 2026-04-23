#pragma once
#include <cstddef>

namespace bs {

// Abstract base class for option payoffs.
// Concrete subclasses implement operator()(double S) to return the payoff
// at terminal spot price S. This interface allows payoff objects to be
// passed generically to a Monte Carlo pricer.
class Payoff {
public:
  virtual ~Payoff();

  // Evaluate the payoff at terminal spot price S
  virtual double operator()(double S) const = 0;
};

// ─── Vanilla payoffs ─────────────────────────────────────────────────────────

// Call payoff: max(S - K, 0)
class CallPayoff : public Payoff {
public:
  explicit CallPayoff(double K);
  double operator()(double S) const override;

private:
  double K_; // Strike price
};

// Put payoff: max(K - S, 0)
class PutPayoff : public Payoff {
public:
  explicit PutPayoff(double K);
  double operator()(double S) const override;

private:
  double K_; // Strike price
};

// ─── Digital (binary) payoffs
// ─────────────────────────────────────────────────

// Digital call payoff: 1 if S >= K, else 0
class DigitalCallPayoff : public Payoff {
public:
  explicit DigitalCallPayoff(double K);
  double operator()(double S) const override;

private:
  double K_; // Strike price
};

// Digital put payoff: 1 if S < K, else 0
class DigitalPutPayoff : public Payoff {
public:
  explicit DigitalPutPayoff(double K);
  double operator()(double S) const override;

private:
  double K_; // Strike price
};

} // namespace bs
