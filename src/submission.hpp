#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data; // flat grid data stored by rows

public:
  using iterator = std::vector<double>::iterator;
  using const_iterator = std::vector<double>::const_iterator;

  // Initializes a zero-filled grid with specified size
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data(rows * cols) {}

  double &operator()(std::size_t i, std::size_t j) {
    return data[i * cols_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return (*const_cast<Grid*>(this))(i, j);
  }

  std::size_t get_rows() const { return rows_; }
  std::size_t get_cols() const { return cols_; }

  // Iterator to access grid data, provides abstraction over raw pointer access
  iterator data_begin() { return data.begin(); }
  const_iterator data_begin() const { return data.cbegin(); }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  const std::size_t rows = old_grid.get_rows();
  const std::size_t cols = old_grid.get_cols();

  if (cols == 0 || rows == 0) return;

  const auto old_data = old_grid.data_begin();
  const auto new_data = new_grid.data_begin();

  for (std::size_t j = 0; j < cols; ++j) {
    new_data[j] = old_data[j];
  }

  if (rows > 1) {
    const std::size_t last_row = (rows - 1) * cols;
    for (std::size_t j = 0; j < cols; ++j) {
      new_data[last_row + j] = old_data[last_row + j];
    }
  }

  if (rows < 3) return;

  if (cols < 3) {
    for (std::size_t i = 1; i < rows - 1; ++i) {
      const std::size_t row = i * cols;
      new_data[row] = old_data[row];
      if (cols == 2) new_data[row + 1] = old_data[row + 1];
    }
    return;
  }

  // Apply kernel to interior cells
  #pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    // Pointer to adjacent rows, may be beneficial to cache locality
    const auto above = old_data + (i - 1) * cols;
    const auto center = old_data + i * cols;
    const auto below = old_data + (i + 1) * cols;
    const auto output = new_data + i * cols;

    output[0] = center[0];
    output[cols - 1] = center[cols - 1];

    #pragma omp simd
    for (std::size_t j = 1; j < cols - 1; ++j) {
      output[j] = 0.125 * (above[j] + center[j - 1] +
                           center[j + 1] + below[j]) +
                  0.5 * center[j];
    }
  }
}
