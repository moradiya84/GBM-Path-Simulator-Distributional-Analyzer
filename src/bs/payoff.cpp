#include "../../include/bs/payoff.hpp"

#include <algorithm> // std::max

namespace bs
{

// ─── Base ────────────────────────────────────────────────────────────────────

// Out-of-line virtual destructor ensures the vtable is emitted in exactly
// one translation unit (this .cpp), avoiding duplicate-symbol issues.
Payoff::~Payoff() = default;

// ─── CallPayoff ──────────────────────────────────────────────────────────────

CallPayoff::CallPayoff(double K) : K_(K) {}

// max(S - K, 0)
double CallPayoff::operator()(double S) const
{
  return std::max(S - K_, 0.0);
}

// ─── PutPayoff ───────────────────────────────────────────────────────────────

PutPayoff::PutPayoff(double K) : K_(K) {}

// max(K - S, 0)
double PutPayoff::operator()(double S) const
{
  return std::max(K_ - S, 0.0);
}

// ─── DigitalCallPayoff ───────────────────────────────────────────────────────

DigitalCallPayoff::DigitalCallPayoff(double K) : K_(K) {}

// 1 if S >= K, else 0
double DigitalCallPayoff::operator()(double S) const
{
  return (S >= K_) ? 1.0 : 0.0;
}

// ─── DigitalPutPayoff ────────────────────────────────────────────────────────

DigitalPutPayoff::DigitalPutPayoff(double K) : K_(K) {}

// 1 if S < K, else 0
double DigitalPutPayoff::operator()(double S) const
{
  return (S < K_) ? 1.0 : 0.0;
}

} // namespace bs
