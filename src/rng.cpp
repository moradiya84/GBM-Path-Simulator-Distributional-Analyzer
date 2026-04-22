#include "../include/rng.hpp"

PseudoRNG::PseudoRNG(unsigned int seed)
    : generator_(seed), normal_dist_(0.0, 1.0)
{
}

std::vector<double> PseudoRNG::generate_standard_normal(size_t dim)
{
  std::vector<double> z(dim);
  for (size_t i = 0; i < dim; ++i)
  {
    z[i] = normal_dist_(generator_);
  }
  return z;
}
