#pragma once

#include <vector>
#include <cstddef>
#include <random>

class RNG
{
public:
  virtual ~RNG() = default;

  // Generates a vector of independent standard normal variables
  virtual std::vector<double> generate_standard_normal(size_t dim) = 0;
};

class PseudoRNG : public RNG
{
private:
  std::mt19937 generator_;
  std::normal_distribution<double> normal_dist_;

public:
  explicit PseudoRNG(unsigned int seed);

  std::vector<double> generate_standard_normal(size_t dim) override;
};
