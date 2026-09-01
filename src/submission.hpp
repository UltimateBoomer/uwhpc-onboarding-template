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
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  auto h = old_grid.get_rows();
  auto w = old_grid.get_cols();

  if (w == 0 || h == 0) return;

  // Copy boundary
  for (std::ptrdiff_t i = 0; i < h; ++i) {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, w - 1) = old_grid(i, w - 1);
  }
  for (std::ptrdiff_t j = 1; j < w - 1; ++j) {
    new_grid(0, j) = old_grid(0, j);
    new_grid(h - 1, j) = old_grid(h - 1, j);
  }

  // Apply kernel to interior cells
  #pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 1; i < h - 1; ++i) {
    #pragma omp simd
    for (std::ptrdiff_t j = 1; j < w - 1; ++j) {
      double s1 = old_grid(i - 1, j);
      double s2 = old_grid(i, j - 1);
      double s3 = old_grid(i, j);
      double s4 = old_grid(i, j + 1);
      double s5 = old_grid(i + 1, j);

      new_grid(i, j) = 0.125 * (s1 + s2 + s4 + s5) + 0.5 * s3;
    }
  }
}
