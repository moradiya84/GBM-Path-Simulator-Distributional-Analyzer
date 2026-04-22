#include "../include/time_grid.hpp"
#include <stdexcept>

TimeGrid::TimeGrid(double T, size_t num_steps)
{
  if (T <= 0.0)
  {
    throw std::invalid_argument("Time T must be positive");
  }
  if (num_steps == 0)
  {
    throw std::invalid_argument("num_steps must be greater than 0");
  }

  dt_ = T / num_steps;
  times_.reserve(num_steps + 1);

  for (size_t i = 0; i <= num_steps; ++i)
  {
    times_.push_back(i * dt_);
  }
}
