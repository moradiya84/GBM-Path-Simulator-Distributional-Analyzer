#include "../include/cholesky.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Utility to check floating point equality
bool is_close(double a, double b, double tol = 1e-4) {
  return std::abs(a - b) < tol;
}

// 1. Verify L * L^T = original matrix
void test_reconstruction() {
  std::vector<std::vector<double>> rho = {
      {1.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 1.0}};

  // Generate decomposition
  Cholesky chol(rho);
  const auto &L = chol.get_L();
  size_t n = rho.size();

  // Compute Reconstruction = L * L^T
  std::vector<std::vector<double>> recon(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      // Native matrix multiplication L * L^T
      // Note: L^T[k][j] mathematically corresponds to L[j][k]
      for (size_t k = 0; k < n; ++k) {
        sum += L[i][k] * L[j][k];
      }
      recon[i][j] = sum;

      // Assert reconstruction accurately retrieves source correlation grid
      assert(is_close(recon[i][j], rho[i][j]));
    }
  }
  std::cout << "[PASS] test_reconstruction (L * L^T reconstructed exactly)\n";
}

// 2. Verify generated samples approximate correlation
void test_empirical_correlation() {
  std::vector<std::vector<double>> rho = {{1.0, 0.6}, {0.6, 1.0}};
  size_t n = rho.size();
  Cholesky chol(rho);

  size_t num_samples = 100000;
  std::vector<std::vector<double>> samples(num_samples, std::vector<double>(n));

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  double mean_x = 0.0, mean_y = 0.0;

  // Generate heavy empirical sample blocks
  for (size_t s = 0; s < num_samples; ++s) {
    std::vector<double> z = {dist(gen), dist(gen)};
    std::vector<double> x = chol.generate_correlated(z);

    samples[s] = x;
    mean_x += x[0];
    mean_y += x[1];
  }

  mean_x /= num_samples;
  mean_y /= num_samples;

  double cov = 0.0, var_x = 0.0, var_y = 0.0;
  for (size_t s = 0; s < num_samples; ++s) {
    double dx = samples[s][0] - mean_x;
    double dy = samples[s][1] - mean_y;
    cov += dx * dy;
    var_x += dx * dx;
    var_y += dy * dy;
  }

  // Solve for r
  double empirical_rho = cov / std::sqrt(var_x * var_y);

  // Asserts empirical rho converges tightly against targeted 0.6 margin
  // Tolerated variance threshold set comfortably at exactly 1% against 100k
  // samples
  assert(is_close(empirical_rho, 0.6, 0.01));
  std::cout << "[PASS] test_empirical_correlation (Empirical ~ "
            << empirical_rho << " vs Target 0.6)\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "Starting Cholesky Verification...\n";
  std::cout << "========================================\n";

  test_reconstruction();
  test_empirical_correlation();

  std::cout << "========================================\n";
  std::cout << "All tests passed successfully.\n";
  std::cout << "========================================\n";
  return 0;
}
