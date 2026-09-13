#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

inline constexpr std::size_t openmp_min_cells = 4096;

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

public:
  using iterator = std::vector<double>::iterator;
  using const_iterator = std::vector<double>::const_iterator;

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

namespace {

// Apply one logical row through random-access iterators. Grid iterators retain
// the storage abstraction while compiling to direct address calculations.
inline void apply_stencil_row(Grid::const_iterator above,
                              Grid::const_iterator center,
                              Grid::const_iterator below, Grid::iterator output,
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

inline void apply_stencil_interior(const Grid &old_grid, Grid &new_grid) {
  const std::size_t rows = old_grid.get_rows();
  const std::size_t cols = old_grid.get_cols();

  // Process each row in parallel, giving every thread exclusive cache-local
  // output.
#pragma omp parallel for schedule(static) if (rows * cols >= openmp_min_cells)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    apply_stencil_row(old_grid.row_begin(i - 1), old_grid.row_begin(i),
                      old_grid.row_begin(i + 1), new_grid.row_begin(i), cols);
  }
}

// Copy boundary rows and handle grids with no interior cells. Returns true when
// the boundary covers the entire grid and no stencil rows remain to process.
inline bool apply_stencil_boundary(const Grid &old_grid, Grid &new_grid) {
  const std::size_t rows = old_grid.get_rows();
  const std::size_t cols = old_grid.get_cols();

  if (rows == 0 || cols == 0)
    return true;

  std::copy_n(old_grid.row_begin(0), cols, new_grid.row_begin(0));

  if (rows > 1) {
    std::copy_n(old_grid.row_begin(rows - 1), cols,
                new_grid.row_begin(rows - 1));
  }

  if (rows < 3)
    return true;

  if (cols < 3) {
    for (std::size_t i = 1; i < rows - 1; ++i) {
      const auto old_row = old_grid.row_begin(i);
      const auto new_row = new_grid.row_begin(i);
      new_row[0] = old_row[0];
      if (cols == 2)
        new_row[1] = old_row[1];
    }
    return true;
  }

  return false;
}

} // namespace

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
inline void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  if (apply_stencil_boundary(old_grid, new_grid))
    return;

  apply_stencil_interior(old_grid, new_grid);
}
