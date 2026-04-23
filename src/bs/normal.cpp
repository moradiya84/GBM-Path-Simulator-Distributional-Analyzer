#include "../../include/bs/normal.hpp"

#include <cmath>

namespace bs {

// Standard normal PDF: phi(x) = (1/sqrt(2*pi)) * exp(-x^2 / 2)
double normal_pdf(double x) {
  constexpr double inv_sqrt_2pi = 0.3989422804014327; // 1 / sqrt(2*pi)
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// Standard normal CDF using the complementary error function:
//   Phi(x) = 0.5 * erfc(-x / sqrt(2))
// std::erfc is available in <cmath> (C++11) and is numerically stable
// for both large positive and large negative x.
double normal_cdf(double x) {
  constexpr double inv_sqrt_2 = 0.7071067811865476; // 1 / sqrt(2)
  return 0.5 * std::erfc(-x * inv_sqrt_2);
  // erfc is 1 - erf(x)
  // erf is error function integral of e^(-t^2) dt from 0 to x
}

} // namespace bs
