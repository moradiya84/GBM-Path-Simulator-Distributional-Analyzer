#pragma once

#include <vector>

class Cholesky {
private:
  std::vector<std::vector<double>> L_; // The lower triangular matrix factor L
  size_t size_;

  // Internal helper to decompose the correlation matrix
  void decompose(const std::vector<std::vector<double>> &matrix);

public:
  /* The constructor will take the correlation matrix, validate it, and compute
    L immediately.*/
  explicit Cholesky(const std::vector<std::vector<double>> &correlation_matrix);

  /* Access to the generated L matrix (mostly useful for testing/debugging)*/
  const std::vector<std::vector<double>> &get_L() const { return L_; }

  /* Takes a vector of independent standard normals `z` (N(0, 1))
    Computes and returns the correlated variables `x = L * z`*/
  std::vector<double>
  generate_correlated(const std::vector<double> &independent_normals) const;
};
