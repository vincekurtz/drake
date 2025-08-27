#pragma once

#include <cstdint>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Struct's for BVH broad phase implementation
// Reference: Warp (https://github.com/NVIDIA/warp/blob/main/warp/native/bvh.h)
struct BVHPackedNodeHalf {
  double x;
  double y;
  double z;
  // For non-leaf nodes:
  // - 'lower.i' represents the index of the left child node.
  // - 'upper.i' represents the index of the right child node.
  //
  // For leaf nodes:
  // - 'lower.i' indicates the start index of the primitives (AABB) in
  // 'primitive_indices'.
  // - 'upper.i' indicates the index just after the last primitive (AABB) in
  // 'primitive_indices'
  unsigned int i : 31;
  unsigned int b : 1;
};
struct BVH {
  BVHPackedNodeHalf* node_lowers;  // See BVHPackedNodeHalf for details
  BVHPackedNodeHalf* node_uppers;  // See BVHPackedNodeHalf for details

  // used for fast refits
  int* node_parents;
  // node_counts are the number of nodes that are children to each node in this
  // BVH Not owned by the BVH, just points to num_childrenAll in DeviceBVHData
  int* node_counts;
  // reordered primitive indices corresponds to the ordering of leaf nodes
  // Not owned by the BVH, just points to indicesAll in DeviceBVHData
  uint32_t* primitive_indices;

  int max_depth;
  int max_nodes;
  int num_nodes;
  // since we use packed leaf nodes, the number of them is no longer the number
  // of items, but variable
  int num_leaf_nodes;

  // pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
  // representing the root of the tree, this is not always the first node
  // for bottom-up builders
  int* root;

  // item bounds are not owned by the BVH but by the caller
  Vector3<double>* item_lowers;
  Vector3<double>* item_uppers;
  int num_items;
};
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake