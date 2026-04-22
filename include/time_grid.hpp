#pragma once

#include <vector>
#include <cstddef>

class TimeGrid
{
private:
  std::vector<double> times_;
  double dt_;

public:
  TimeGrid(double T, size_t num_steps);

  const std::vector<double> &get_times() const { return times_; }
  double dt() const { return dt_; }
  size_t num_steps() const { return times_.size() - 1; }
  double T() const { return times_.back(); }
};
