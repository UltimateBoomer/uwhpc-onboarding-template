#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

inline constexpr std::size_t openmp_min_cells = 8192;

// Apply one logical row through random-access iterators. std::vector iterators
// compile to the same address calculations as pointers in optimized builds.
template <typename InputIterator, typename OutputIterator>
inline void apply_stencil_row(InputIterator above, InputIterator center,
                              InputIterator below, OutputIterator output,
                              std::size_t cols) noexcept {
  output[0] = center[0];
  output[cols - 1] = center[cols - 1];

  // Using SIMD between cells due to independence, apply the stencil kernel to
  // the interior of the row.
#pragma omp simd
  for (std::size_t j = 1; j < cols - 1; ++j) {
    output[j] = 0.125 * (above[j] + center[j - 1] + center[j + 1] + below[j]) +
                0.5 * center[j];
  }
}

} // namespace

class Grid;
inline void apply_stencil(const Grid &old_grid, Grid &new_grid);

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data;

  using iterator = std::vector<double>::iterator;
  using const_iterator = std::vector<double>::const_iterator;

public:
  // Initializes a zero-filled grid with the specified dimensions.
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data(rows * cols) {}

  double &operator()(std::size_t i, std::size_t j) {
    return data[i * cols_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return data[i * cols_ + j];
  }

  std::size_t get_rows() const noexcept { return rows_; }
  std::size_t get_cols() const noexcept { return cols_; }

  // Returns iterator to the beginning of a row.
  iterator row_begin(std::size_t row) noexcept {
    return data.begin() + row * cols_;
  }

  // Returns iterator to the beginning of a row.
  const_iterator row_begin(std::size_t row) const noexcept {
    return data.cbegin() + row * cols_;
  }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
inline void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  const std::size_t rows = old_grid.get_rows();
  const std::size_t cols = old_grid.get_cols();

  // Empty grid
  if (cols == 0 || rows == 0)
    return;

  // Copy boundary rows
  std::copy_n(old_grid.row_begin(0), cols, new_grid.row_begin(0));

  if (rows > 1) {
    std::copy_n(old_grid.row_begin(rows - 1), cols,
                new_grid.row_begin(rows - 1));
  }

  // With less than 3 rows, the copied rows cover the entire grid.
  if (rows < 3)
    return;

  // With less than 3 columns, every cell is on a boundary.
  if (cols < 3) {
    for (std::size_t i = 1; i < rows - 1; ++i) {
      const auto old_row = old_grid.row_begin(i);
      const auto new_row = new_grid.row_begin(i);
      new_row[0] = old_row[0];
      if (cols == 2)
        new_row[1] = old_row[1];
    }
    return;
  }

  // Avoid parallelization when the grid is too small.
  if (rows * cols < openmp_min_cells) {
    for (std::size_t i = 1; i < rows - 1; ++i) {
      apply_stencil_row(old_grid.row_begin(i - 1), old_grid.row_begin(i),
                        old_grid.row_begin(i + 1), new_grid.row_begin(i), cols);
    }
    return;
  }

  // Each row is independent, giving every thread exclusive cache-local output.
#pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    apply_stencil_row(old_grid.row_begin(i - 1), old_grid.row_begin(i),
                      old_grid.row_begin(i + 1), new_grid.row_begin(i), cols);
  }
}
