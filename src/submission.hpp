#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

inline constexpr std::size_t OPENMP_MIN_CELLS = 4096;
inline constexpr std::size_t CACHE_LINE_SIZE = 64;
inline constexpr std::size_t CACHE_LINE_ELEMENTS =
    CACHE_LINE_SIZE / sizeof(double);
inline constexpr std::size_t ROW_PREFIX_ELEMENTS = CACHE_LINE_ELEMENTS - 1;

template <typename T, std::size_t Alignment> class AlignedAllocator {
public:
  using value_type = T;

  static_assert(Alignment >= alignof(T));
  static_assert((Alignment & (Alignment - 1)) == 0);

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length{};

    return static_cast<T *>(
        ::operator new(count * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T *pointer, std::size_t) noexcept {
    ::operator delete(pointer, std::align_val_t{Alignment});
  }

  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment> &,
                const AlignedAllocator<U, Alignment> &) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment> &,
                const AlignedAllocator<U, Alignment> &) noexcept {
  return false;
}

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  using storage_type =
      std::vector<double, AlignedAllocator<double, CACHE_LINE_SIZE>>;

  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  storage_type data;

public:
  using iterator = storage_type::iterator;
  using const_iterator = storage_type::const_iterator;

  // Initializes a zero-filled grid with the specified dimensions.
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols),
        stride_((cols + ROW_PREFIX_ELEMENTS + CACHE_LINE_ELEMENTS - 1) /
                CACHE_LINE_ELEMENTS * CACHE_LINE_ELEMENTS),
        data(rows * stride_) {}

  double &operator()(std::size_t i, std::size_t j) {
    return data[ROW_PREFIX_ELEMENTS + i * stride_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return data[ROW_PREFIX_ELEMENTS + i * stride_ + j];
  }

  std::size_t get_rows() const noexcept { return rows_; }
  std::size_t get_cols() const noexcept { return cols_; }

  // Returns iterator to the beginning of a row.
  iterator row_begin(std::size_t row) noexcept {
    return data.begin() + ROW_PREFIX_ELEMENTS + row * stride_;
  }

  // Returns iterator to the beginning of a row.
  const_iterator row_begin(std::size_t row) const noexcept {
    return data.cbegin() + ROW_PREFIX_ELEMENTS + row * stride_;
  }
};

namespace {

// Apply one logical row from iterators to its aligned first interior cell.
inline void apply_stencil_row(Grid::const_iterator above,
                              Grid::const_iterator center,
                              Grid::const_iterator below, Grid::iterator output,
                              std::size_t interior_cols) noexcept {
  output[-1] = center[-1];
  output[interior_cols] = center[interior_cols];

  // Extract raw pointers, required for compiler to output aligned SIMD code.
  const double *const above_data = &*above;
  const double *const center_data = &*center;
  const double *const below_data = &*below;
  double *const output_data = &*output;

  // Using SIMD between cells due to independence, apply the stencil kernel to
  // the interior of the row.
#pragma omp simd aligned(above_data, center_data, below_data,                  \
                             output_data : CACHE_LINE_SIZE)
  for (std::size_t j = 0; j < interior_cols; ++j) {
    output_data[j] = 0.125 * (above_data[j] + center_data[j - 1] +
                              center_data[j + 1] + below_data[j]) +
                     0.5 * center_data[j];
  }
}

inline void apply_stencil_interior(const Grid &old_grid, Grid &new_grid) {
  const std::size_t rows = old_grid.get_rows();
  const std::size_t cols = old_grid.get_cols();

  // Process each row in parallel, giving every thread exclusive cache-local
  // output.
#pragma omp parallel for schedule(static) if (rows * cols >= OPENMP_MIN_CELLS)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    apply_stencil_row(old_grid.row_begin(i - 1) + 1, old_grid.row_begin(i) + 1,
                      old_grid.row_begin(i + 1) + 1, new_grid.row_begin(i) + 1,
                      cols - 2);
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
