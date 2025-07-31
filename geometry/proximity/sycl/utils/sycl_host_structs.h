#pragma once

#include <cstdint>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/sycl/utils/sycl_bvh_structs.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Struct to hold a deep-copied host version of BVH data.
struct HostBVH {
  std::vector<BVHPackedNodeHalf> node_lowers;
  std::vector<BVHPackedNodeHalf> node_uppers;
  std::vector<int> node_parents;
  int root_index;  // Dereferenced from device root pointer.
  int max_nodes;
  int num_nodes;
  int num_leaf_nodes;
  // Add other fields as needed (e.g., primitive_indices if required).
};

struct HostIndicesAll {
  std::vector<uint32_t> indicesAll;
};

struct HostMeshPairCollidingIndices {
  std::vector<uint32_t> collision_indices_A;
  std::vector<uint32_t> collision_indices_B;
};

struct HostMeshACollisionCounters {
  std::vector<uint32_t> collision_counts;
  uint32_t total_collisions = 0;
  uint32_t last_element_collision_count = 0;
};

struct HostMeshData {
  std::vector<uint32_t> element_offsets;
  std::vector<Vector3<double>> element_aabb_min_W;
  std::vector<Vector3<double>> element_aabb_max_W;
  uint32_t total_elements;
};
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake