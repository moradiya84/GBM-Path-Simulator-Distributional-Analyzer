#pragma once

namespace bs
{

// Standard normal probability density function
// phi(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
double normal_pdf(double x);

// Standard normal cumulative distribution function
// Phi(x) = integral from -inf to x of phi(t) dt
// Implemented via std::erfc for numerical accuracy
double normal_cdf(double x);

} // namespace bs
