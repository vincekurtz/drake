#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "geometry/proximity/sycl/utils/sycl_kernel_utils.h"
#include "geometry/proximity/sycl/utils/sycl_memory_manager.h"
#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Overload that takes Vector3<double> by value
SYCL_EXTERNAL inline BVHPackedNodeHalf make_node(const Vector3<double>& bound,
                                                 int child, bool leaf) {
  BVHPackedNodeHalf n;
  n.x = bound.x();
  n.y = bound.y();
  n.z = bound.z();
  n.i = static_cast<unsigned int>(child);
  n.b = static_cast<unsigned int>(leaf ? 1 : 0);

  return n;
}

// variation of make_node through volatile pointers used in build_hierarchy
SYCL_EXTERNAL inline void make_node(volatile BVHPackedNodeHalf* n,
                                    const Vector3<double>& bound, int child,
                                    bool leaf) {
  n->x = bound.x();
  n->y = bound.y();
  n->z = bound.z();
  n->i = static_cast<unsigned int>(child);
  n->b = static_cast<unsigned int>(leaf ? 1 : 0);
}

// TODO - Can apparently be done more efficiently by loading as float4 and then
// converting to BVHPackedNodeHalf (according to Warp). Try later
SYCL_EXTERNAL inline BVHPackedNodeHalf bvh_load_node(
    const BVHPackedNodeHalf* nodes, int index) {
  return nodes[index];
}

// Interleaves with 2 zeroes between each bit
SYCL_EXTERNAL inline uint32_t part1by2(uint32_t n) {
  n = (n ^ (n << 16)) & 0xff0000ff;
  n = (n ^ (n << 8)) & 0x0300f00f;
  n = (n ^ (n << 4)) & 0x030c30c3;
  n = (n ^ (n << 2)) & 0x09249249;

  return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of
// length 3*lwp2(dim) bits
template <int dim>
SYCL_EXTERNAL inline uint32_t morton3(float x, float y, float z) {
  uint32_t ux = sycl::clamp(int(x * dim), 0, dim - 1);
  uint32_t uy = sycl::clamp(int(y * dim), 0, dim - 1);
  uint32_t uz = sycl::clamp(int(z * dim), 0, dim - 1);
  // If dim = 2014, then 10 bit + 10 bit + 10 bit = 30 bit Morton code
  return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

// Broad Phase
// Build the mesh BVH's if not already built otherwise just traverse it

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH
// construction
// this is a bottom-up clustering method that outputs one node per-leaf
// This class creates BVHs for all meshes in parallel
class BVHBroadPhase {
 public:
  BVHBroadPhase() = default;
  ~BVHBroadPhase() = default;

  void BroadPhase(
      const DeviceMeshData& mesh_data,
      const std::vector<Vector3<double>>& sorted_total_lower,
      const std::vector<Vector3<double>>& sorted_total_upper,
      DeviceBVHData& bvh_data, sycl::event& element_aabb_event,
      std::unordered_map<uint64_t, std::pair<DeviceMeshACollisionCounters,
                                             DeviceMeshPairCollidingIndices>>&
          collision_candidates_to_data,
      uint32_t num_mesh_collisions,
      DeviceCollidingIndicesMemoryChunk& pair_chunk_,
      DeviceCollisionCountersMemoryChunk& counters_chunk_,
      DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk_,
      DeviceMeshPairIds& mesh_pair_ids, SyclMemoryManager& memory_manager,
      sycl::queue& q_device);
  // Construct and return BVH for all meshes in the scene
  // They will be indexed by same order of sorted_geometry ids
  // q is waited on becaue memory needs to be released
  void build(const DeviceMeshData& mesh_data,
             const std::vector<Vector3<double>>& sorted_total_lower,
             const std::vector<Vector3<double>>& sorted_total_upper,
             DeviceBVHData& bvh_data, sycl::event& element_aabb_event,
             SyclMemoryManager& memory_manager, sycl::queue& q_device);
  bool IsBVHBuilt() const { return bvh_built_; }
  bool IsBVHRefitted() const { return bvh_refitted_; }
  void SetBVHRefitted(bool refitted) { bvh_refitted_ = refitted; }

 private:
  // BVH construction parameters
  enum class BVHParams : int {
    kMinPrimitivesPerLeaf = 8,  // Minimum primitives before creating leaf
    kMaxDepth = 32              // Maximum tree depth before forcing leaf
  };
  sycl::event refit(const DeviceMeshData& mesh_data, DeviceBVHData& bvh_data,
                    sycl::event& element_aabb_event,
                    SyclMemoryManager& memory_manager, const uint32_t* mesh_as,
                    const uint32_t* mesh_bs, const uint32_t num_mesh_as,
                    sycl::queue& q_device);
  sycl::event ComputeCollisionCounts(const uint32_t mesh_a,
                                     const uint32_t mesh_b,
                                     const DeviceBVHData& bvh_data,
                                     const DeviceMeshData& mesh_data,
                                     DeviceMeshACollisionCounters& cc,
                                     sycl::event& refit_event,
                                     sycl::queue& q_device);
  sycl::event ComputeCollisionCountsAll(
      const uint32_t* meshAs, const uint32_t* meshBs,
      const DeviceBVHData& bvh_data, const DeviceMeshData& mesh_data,
      DeviceCollisionCountersMemoryChunk& counters_chunk,
      DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk,
      sycl::event& refit_event, sycl::queue& q_device);
  sycl::event ComputeCollisionPairs(const uint32_t mesh_a,
                                    const uint32_t mesh_b,
                                    const DeviceBVHData& bvh_data,
                                    const DeviceMeshData& mesh_data,
                                    DeviceMeshACollisionCounters& cc,
                                    DeviceMeshPairCollidingIndices& ci,
                                    sycl::queue& q_device);
  sycl::event ComputeCollisionPairsAll(
      const uint32_t* meshAs, const uint32_t* meshBs,
      const DeviceBVHData& bvh_data, const DeviceMeshData& mesh_data,
      DeviceCollisionCountersMemoryChunk& counters_chunk,
      DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk,
      DeviceCollidingIndicesMemoryChunk& pair_chunk, sycl::queue& q_device);

  bool bvh_built_ = false;
  bool bvh_refitted_ = false;
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake