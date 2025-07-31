#pragma once

#include <sycl/sycl.hpp>

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Helper function to round up to nearest multiple of work group size
SYCL_EXTERNAL inline uint32_t RoundUpToWorkGroupSize(uint32_t n,
                                                     uint32_t work_group_size) {
  return ((n + work_group_size - 1) / work_group_size) * work_group_size;
}

// Returns the componentwise min and max of two Vector3<double>.
// The first element of the pair is the min, the second is the max.
SYCL_EXTERNAL inline Vector3<double> ComponentwiseMin(
    const Vector3<double>& a, const Vector3<double>& b) {
  Vector3<double> min_v;
  for (int i = 0; i < 3; ++i) {
    min_v[i] = sycl::min(a[i], b[i]);
  }
  return min_v;
}

// Returns the componentwise max of two Vector3<double>.
SYCL_EXTERNAL inline Vector3<double> ComponentwiseMax(
    const Vector3<double>& a, const Vector3<double>& b) {
  Vector3<double> max_v;
  for (int i = 0; i < 3; ++i) {
    max_v[i] = sycl::max(a[i], b[i]);
  }
  return max_v;
}

// Returns a unique key for a pair of integers
SYCL_EXTERNAL inline uint64_t key(uint32_t i, uint32_t j) {
  return static_cast<uint64_t>(i) << 32 | j;
}

// Return pair of integers from a unique key
SYCL_EXTERNAL inline std::pair<uint32_t, uint32_t> key_to_pair(uint64_t key) {
  return std::make_pair(static_cast<uint32_t>(key >> 32),
                        static_cast<uint32_t>(key & 0xFFFFFFFF));
}

SYCL_EXTERNAL inline bool AABBsIntersect(const Vector3<double>& lower_A,
                                         const Vector3<double>& upper_A,
                                         const Vector3<double>& node_lower,
                                         const Vector3<double>& node_upper) {
  for (int i = 0; i < 3; ++i) {
    if (node_upper[i] < lower_A[i]) return false;
    if (upper_A[i] < node_lower[i]) return false;
  }
  return true;
}

SYCL_EXTERNAL inline bool PressuresIntersect(const double min_A,
                                             const double max_A,
                                             const double min_B,
                                             const double max_B) {
  return !(max_B < min_A || max_A < min_B);
}

// AI Generated - tetsted externally
template <class RandomIt, class T, class Compare = std::less<>>
SYCL_EXTERNAL RandomIt upper_bound_device(RandomIt first, RandomIt last,
                                          const T& value,
                                          Compare comp = Compare{}) {
  while (first < last) {
    RandomIt mid = first + (last - first) / 2;
    if (comp(value, *mid))  // value < *mid
      last = mid;
    else  // value >= *mid
      first = mid + 1;
  }
  return first;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake