#include "../include/cholesky.hpp"

Cholesky::Cholesky(const std::vector<std::vector<double>> &correlation_matrix) {
  if (correlation_matrix.empty()) {
    throw std::invalid_argument("Correlation matrix cannot be empty.");
  }

  size_ = correlation_matrix.size();

  // Initialize L_ to be a size_ x size_ lower triangular matrix filled with 0s
  L_ = std::vector<std::vector<double>>(size_, std::vector<double>(size_, 0.0));

  // Compute the decomposition securely
  decompose(correlation_matrix);
}

void Cholesky::decompose(const std::vector<std::vector<double>> &matrix) {
  for (size_t i = 0; i < size_; i++) {
    // Validation: square matrix
    if (matrix[i].size() != size_) {
      throw std::invalid_argument("Correlation matrix must be square.");
    }

    for (size_t j = 0; j <= i; j++) {
      double sum = 0.0;

      // Validation: Symmetry (checking both sides of the mirror match)
      if (matrix[i][j] != matrix[j][i]) {
        throw std::invalid_argument(
            "Correlation matrix must be strictly symmetric.");
      }

      if (j == i) {
        // Diagonal elements processing
        for (size_t k = 0; k < j; k++) {
          sum += (L_[j][k] * L_[j][k]);
        }

        double diff = matrix[j][j] - sum;
        if (diff <= 0.0) {
          throw std::invalid_argument(
              "Correlation matrix must be positively definite.");
        }
        L_[j][j] = std::sqrt(diff);
      } else {
        // Non-diagonal elements processing
        for (size_t k = 0; k < j; k++) {
          sum += (L_[i][k] * L_[j][k]);
        }
        L_[i][j] = (matrix[i][j] - sum) / L_[j][j];
      }
    }
  }
}

std::vector<double> Cholesky::generate_correlated(
    const std::vector<double> &independent_normals) const {
  if (independent_normals.size() != size_) {
    throw std::invalid_argument("Independent normals vector size must "
                                "precisely match the correlation matrix size.");
  }

  std::vector<double> correlated(size_, 0.0);

  // We compute x = L * z
  for (size_t i = 0; i < size_; i++) {
    double sum = 0.0;
    // L is lower triangular, so we only need to sum up to k = i
    for (size_t j = 0; j <= i; j++) {
      sum += L_[i][j] * independent_normals[j];
    }
    correlated[i] = sum;
  }

  return correlated;
}
