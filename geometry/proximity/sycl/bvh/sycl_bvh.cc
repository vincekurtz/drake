#include "geometry/proximity/sycl/bvh/sycl_bvh.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <oneapi/dpl/algorithm>  // For sort_by_key
#include <oneapi/dpl/execution>  // For execution policies
#include <oneapi/dpl/numeric>    // For exclusive_scan
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {
// Forward declarations for kernel names
class ComputeInvEdgesKernel;
class ComputeMortonCodesKernel;
class ComputeKeyDeltasKernel;
class BuildLeavesKernel;
class BuildTreeKernel;
class PackLeavesKernel;
class RefitKernel;
class ComputeCollisionCountsKernel;
class ComputeCollisionPairsKernel;
class ComputeCollisionCountsAllKernel;
class ComputeCollisionPairsAllKernel;

void BVHBroadPhase::build(
    const DeviceMeshData& mesh_data,
    const std::vector<Vector3<double>>& sorted_total_lower,
    const std::vector<Vector3<double>>& sorted_total_upper,
    DeviceBVHData& bvh_data, sycl::event& element_aabb_event,
    SyclMemoryManager& memory_manager, sycl::queue& q_device) {
  // Run all OneAPI algos on device
  auto policy = oneapi::dpl::execution::make_device_policy(q_device);
  int num_geometries = sorted_total_lower.size();

  // Copy the mesh wise AABB lower and upper to the device
  auto copy_mesh_aabb_event1 = memory_manager.CopyToDevice(
      bvh_data.total_lowerAll, sorted_total_lower.data(), num_geometries);
  auto copy_mesh_aabb_event2 = memory_manager.CopyToDevice(
      bvh_data.total_upperAll, sorted_total_upper.data(), num_geometries);

  std::vector<int> mesh_ids(num_geometries);
  std::iota(mesh_ids.begin(), mesh_ids.end(), 0);

  // // Compute the total bounds of the meshes
  // std::vector<sycl::event> total_bound_events;
  // for (int mesh_id : mesh_ids) {
  //   uint32_t this_geom_start = mesh_data.element_offsets[mesh_id];
  //   uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
  //   auto element_aabb_min_W = mesh_data.element_aabb_min_W + this_geom_start;
  //   auto element_aabb_max_W = mesh_data.element_aabb_max_W + this_geom_start;
  //   auto total_lowerAll = bvh_data.total_lowerAll + mesh_id;
  //   auto total_upperAll = bvh_data.total_upperAll + mesh_id;
  // }

  // Compute inverse edges of all mesh level AABBs
  // This is used to compute the Morton Codes
  auto inv_edges_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({copy_mesh_aabb_event1, copy_mesh_aabb_event2});
    h.parallel_for<ComputeInvEdgesKernel>(
        num_geometries,
        [=, total_upperAll = bvh_data.total_upperAll,
         total_lowerAll = bvh_data.total_lowerAll,
         total_inv_edgesAll = bvh_data.total_inv_edgesAll](sycl::item<1> item)
            [[intel::kernel_args_restrict]] {
              int index = item.get_id(0);
              Vector3<double> edge =
                  (total_upperAll[index] - total_lowerAll[index]);
              edge += Vector3<double>(1e-6, 1e-6, 1e-6);
              total_inv_edgesAll[index] =
                  Vector3<double>(1.0 / edge[0], 1.0 / edge[1], 1.0 / edge[2]);
            });
  });

  // Now we compute the morton codes for all the elements in all the meshes
  // simultaneously
  // For this we need our AABBs
  auto compute_morton_codes_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({element_aabb_event, inv_edges_event});
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.parallel_for<ComputeMortonCodesKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, total_elements = mesh_data.total_elements,
         total_lowerAll = bvh_data.total_lowerAll,
         total_inv_edgesAll = bvh_data.total_inv_edgesAll,
         element_mesh_ids = mesh_data.element_mesh_ids,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         keysAll = bvh_data.keysAll] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);  // Global Element Index
          if (global_eI < total_elements) {
            Vector3<double> min_W = element_aabb_min_W[global_eI];
            Vector3<double> max_W = element_aabb_max_W[global_eI];
            Vector3<double> center_W = (min_W + max_W) / 2.0;

            uint32_t geom_index =
                element_mesh_ids[global_eI];  // Geometry this Index
                                              // belongs to
            Vector3<double> geom_inv_edges = total_inv_edgesAll[geom_index];
            Vector3<double> geom_lower_W = total_lowerAll[geom_index];

            // Normalize
            float local_x = (center_W[0] - geom_lower_W[0]) * geom_inv_edges[0];
            float local_y = (center_W[1] - geom_lower_W[1]) * geom_inv_edges[1];
            float local_z = (center_W[2] - geom_lower_W[2]) * geom_inv_edges[2];

            // 10-bit Morton codes stored in lower 30bits (1024^3 effective
            // resolution)
            uint32_t key = morton3<1024>(local_x, local_y, local_z);

            indicesAll[global_eI] = global_eI;
            keysAll[global_eI] = key;
          }
        });
  });

  // Wait for this event since we need to sort based on it
  compute_morton_codes_event.wait_and_throw();

  // oneapi::dpl::for_each does not work
  // ================================
  // seg sort
  // ================================
  for (int mesh_id : mesh_ids) {
    uint32_t this_geom_start = mesh_data.element_offsets[mesh_id];
    uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
    auto keys_begin = bvh_data.keysAll + this_geom_start;
    auto keys_end = keys_begin + this_geom_count;
    auto indices_begin = bvh_data.indicesAll + this_geom_start;
    oneapi::dpl::sort_by_key(policy, keys_begin, keys_end, indices_begin);
    // BVH.primitive_indices stores the index of where to find the AABB of that
    // primitive. We just make it point to the right place in indicesAll
    bvh_data.bvhAll[mesh_id].primitive_indices =
        bvh_data.indicesAll + this_geom_start;
  }
  q_device.wait();

  // Launch the compute key deltas between nearby keys
  auto compute_key_deltas_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.parallel_for<ComputeKeyDeltasKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, keysAll = bvh_data.keysAll,
         total_elements = mesh_data.total_elements,
         element_mesh_ids = mesh_data.element_mesh_ids,
         deltasAll = bvh_data.deltasAll] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < total_elements - 1) {
            uint32_t key = keysAll[global_eI];
            uint32_t next_key = keysAll[global_eI + 1];
            // We need to ignore next key if it is different mesh from key
            if (element_mesh_ids[global_eI] ==
                element_mesh_ids[global_eI + 1]) {
              uint32_t delta = key ^ next_key;
              // No clz because we are always comparing
              // left and right keys and never require the
              // absolute key value
              deltasAll[global_eI] = delta;  //__clz(delta)
            }
          }
        });
  });

  // For this we need to go back to mesh local indices and we do this
  //  by using the offsets array to subtract the sorted index values We also
  //  need the per mesh element counts to assign the right amount of memory
  for (int mesh_id : mesh_ids) {
    uint32_t this_geom_start = mesh_data.element_offsets[mesh_id];
    uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
    auto primitive_indices_begin = bvh_data.bvhAll[mesh_id].primitive_indices;
    auto primitive_indices_end = primitive_indices_begin + this_geom_count;
    oneapi::dpl::transform(
        policy, primitive_indices_begin, primitive_indices_end,
        primitive_indices_begin,
        [=, element_offsets = mesh_data.element_offsets](uint32_t val) {
          return val - element_offsets[mesh_id];
        });
  }

  for (int mesh_id : mesh_ids) {
    // Also allocate memory for the nodes of each BVH
    // Another loop for this so that its non blocking on the host with the above
    // transform
    uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
    uint32_t max_nodes = 2 * this_geom_count - 1;
    bvh_data.bvhAll[mesh_id].max_nodes = max_nodes;
    SyclMemoryHelper::AllocateBVHSingleMeshMemory(
        memory_manager, bvh_data.bvhAll[mesh_id], max_nodes);

    // Initialize parent arrays to -1 (no parent initially)
    q_device.fill(bvh_data.bvhAll[mesh_id].node_parents, -1, max_nodes);
  }
  q_device.wait();

  // Initialize the global num_childrenAll array to 0 for atomic operations
  auto init_children_event =
      memory_manager.Memset(bvh_data.num_childrenAll, bvh_data.total_nodes);
  init_children_event.wait();

  // Build the leaves of the BVH
  // Most of the complication is making the mesh local index to the global index
  // and vice versa General idea is to use local indexing for anything that is
  // used by or goes into bvh_data.bvhAll
  //
  // IMPORTANT: INDEXING SCHEME for range_leftsAll and range_rightsAll
  // ================================================================
  // These arrays store ranges for both leaf and internal nodes using a
  // consistent global node indexing scheme:
  //
  // Layout: [mesh0_nodes(leaves+internal), mesh1_nodes(leaves+internal), ...]
  //
  // For each mesh:
  // - Leaves are stored at: global_node_offset + [0, mesh_element_count-1]
  // - Internal nodes at: global_node_offset + [mesh_element_count,
  // 2*mesh_element_count-2]
  //
  // This ensures both build_leaves and build_tree kernels use the same indexing
  // pattern: global_node_offset + local_index
  // ================================================================
  auto build_leaves_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.parallel_for<BuildLeavesKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, indicesAll = bvh_data.indicesAll, bvhAll = bvh_data.bvhAll,
         total_elements = mesh_data.total_elements,
         element_mesh_ids = mesh_data.element_mesh_ids,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         element_offsets = mesh_data.element_offsets,
         range_leftsAll = bvh_data.range_leftsAll,
         range_rightsAll = bvh_data.range_rightsAll,
         node_offsets = bvh_data.node_offsets] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < total_elements) {
            const uint32_t mesh_id = element_mesh_ids[global_eI];
            const uint32_t geom_local_primitive_index = indicesAll[global_eI];
            const uint32_t ele_offset = element_offsets[mesh_id];
            const uint32_t local_array_indexer = global_eI - ele_offset;
            const uint32_t global_node_offset = node_offsets[mesh_id];

            Vector3<double> lower_W =
                element_aabb_min_W[geom_local_primitive_index + ele_offset];
            Vector3<double> upper_W =
                element_aabb_max_W[geom_local_primitive_index + ele_offset];

            // Create the nodes
            // node_lowers and node_uppers store the nodes in sorted order of
            // morton code
            bvhAll[mesh_id].node_lowers[local_array_indexer] =
                make_node(lower_W, geom_local_primitive_index, true);
            bvhAll[mesh_id].node_uppers[local_array_indexer] =
                make_node(upper_W, geom_local_primitive_index, false);
            // Write leaf key ranges
            // Store ranges using global node indexing (mesh-local leaf index +
            // global node offset)
            uint32_t global_leaf_range_index =
                global_node_offset + local_array_indexer;
            range_leftsAll[global_leaf_range_index] = local_array_indexer;
            range_rightsAll[global_leaf_range_index] = local_array_indexer;
          }
        });
  });
  // Build the entire tree hierarchicy and update the internal node bounds
  auto build_tree_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.depends_on({build_leaves_event});
    h.parallel_for<BuildTreeKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, bvhAll = bvh_data.bvhAll, total_elements = mesh_data.total_elements,
         element_mesh_ids = mesh_data.element_mesh_ids,
         indicesAll = bvh_data.indicesAll, deltasAll = bvh_data.deltasAll,
         range_leftsAll = bvh_data.range_leftsAll,
         range_rightsAll = bvh_data.range_rightsAll,
         num_childrenAll = bvh_data.num_childrenAll,
         element_offsets = mesh_data.element_offsets,
         element_counts = mesh_data.element_counts,
         node_offsets = bvh_data.node_offsets] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < total_elements) {
            const uint32_t mesh_id = element_mesh_ids[global_eI];
            const uint32_t ele_offset = element_offsets[mesh_id];
            const uint32_t mesh_element_count = element_counts[mesh_id];
            const uint32_t local_leaf_index = global_eI - ele_offset;
            const uint32_t internal_offset =
                mesh_element_count;  // Internal nodes start after leaves
            const uint32_t global_node_offset =
                node_offsets[mesh_id];  // Global offset for this mesh's nodes

            // Current node index (starts as leaf, local to this mesh) and acts
            // on sorted arrays
            uint32_t current_index = local_leaf_index;

            for (;;) {
              // Get range for current node using consistent global node
              // indexing Both leaves and internal nodes use: global_node_offset
              // + current_index
              uint32_t global_range_index = global_node_offset + current_index;

              // (left, right) is the range of prrimitives contained within the
              // current node left and right are mesh local primitive indices
              // They can only be used to point to morton code sorted arrays
              // (see build_leaves and how they are stored for the leaves)
              int left = range_leftsAll[global_range_index];
              int right = range_rightsAll[global_range_index];

              // Check if we are the root node for this mesh
              if (left == 0 && right == mesh_element_count - 1) {
                // Set root for this mesh - root is a pointer to int, so we
                // dereference it
                *bvhAll[mesh_id].root = current_index;
                bvhAll[mesh_id].node_parents[current_index] =
                    -1;  // Root has no parent
                break;
              }

              // Determine parents
              int child_count = 0;
              uint32_t parent_local_index;
              bool parent_right = false;

              // If right delta is smaller merge to right
              // If both deltas are equal, do a coin flip ( decision is made
              // using the XOR result of whether the keys before and after the
              // internal node are divisible by 2) to promote balanced tree
              // Otherwise merge to left For extremums (right most or left most
              // node of mesh) merge the other direction
              if (left == 0) {
                parent_right = true;  // No left neighbor, must group with right
              } else if (right != mesh_element_count - 1) {
                // Need to map local indices back to global for deltasAll access
                // deltasAll is indexed by global element indices
                uint32_t right_global = ele_offset + right;
                uint32_t left_minus_1_global = ele_offset + left - 1;

                if (deltasAll[right_global] <= deltasAll[left_minus_1_global]) {
                  if (deltasAll[right_global] ==
                      deltasAll[left_minus_1_global]) {
                    // Tie breaking using primitive indices (global indices)
                    parent_right = (indicesAll[left_minus_1_global] % 2) ^
                                   (indicesAll[right_global] % 2);
                  } else {
                    parent_right = true;
                  }
                }
              }

              // Assign to parent and update parent's range
              if (parent_right) {
                parent_local_index = right + internal_offset;

                // Set parent's left child
                bvhAll[mesh_id].node_parents[current_index] =
                    parent_local_index;
                bvhAll[mesh_id].node_lowers[parent_local_index].i =
                    current_index;

                // Update parent's left range in global arrays
                uint32_t parent_global_index =
                    global_node_offset + parent_local_index;
                range_leftsAll[parent_global_index] = left;

                // Memory fence to ensure writes are visible before atomic
                // increment
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::device);

                // Atomic increment of child count for this parent
                sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device>
                    atomic_children(num_childrenAll[parent_global_index]);
                child_count =
                    atomic_children.fetch_add(1, sycl::memory_order::acq_rel);
              } else {
                parent_local_index = left + internal_offset - 1;

                // Set parent's right child
                bvhAll[mesh_id].node_parents[current_index] =
                    parent_local_index;
                bvhAll[mesh_id].node_uppers[parent_local_index].i =
                    current_index;

                // Update parent's right range in global arrays
                uint32_t parent_global_index =
                    global_node_offset + parent_local_index;
                range_rightsAll[parent_global_index] = right;

                // Memory fence to ensure writes are visible before atomic
                // increment
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::device);

                // Atomic increment of child count for this parent
                sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device>
                    atomic_children(num_childrenAll[parent_global_index]);
                child_count =
                    atomic_children.fetch_add(1, sycl::memory_order::acq_rel);
              }

              // If we're the second child (completing the parent), update
              // bounds and continue
              if (child_count == 1) {
                // Get child indices
                const uint32_t left_child =
                    bvhAll[mesh_id].node_lowers[parent_local_index].i;
                const uint32_t right_child =
                    bvhAll[mesh_id].node_uppers[parent_local_index].i;

                // Get child bounds
                Vector3<double> left_lower(
                    bvhAll[mesh_id].node_lowers[left_child].x,
                    bvhAll[mesh_id].node_lowers[left_child].y,
                    bvhAll[mesh_id].node_lowers[left_child].z);
                Vector3<double> left_upper(
                    bvhAll[mesh_id].node_uppers[left_child].x,
                    bvhAll[mesh_id].node_uppers[left_child].y,
                    bvhAll[mesh_id].node_uppers[left_child].z);
                Vector3<double> right_lower(
                    bvhAll[mesh_id].node_lowers[right_child].x,
                    bvhAll[mesh_id].node_lowers[right_child].y,
                    bvhAll[mesh_id].node_lowers[right_child].z);
                Vector3<double> right_upper(
                    bvhAll[mesh_id].node_uppers[right_child].x,
                    bvhAll[mesh_id].node_uppers[right_child].y,
                    bvhAll[mesh_id].node_uppers[right_child].z);

                // Compute union bounds
                Vector3<double> lower(sycl::min(left_lower[0], right_lower[0]),
                                      sycl::min(left_lower[1], right_lower[1]),
                                      sycl::min(left_lower[2], right_lower[2]));
                Vector3<double> upper(sycl::max(left_upper[0], right_upper[0]),
                                      sycl::max(left_upper[1], right_upper[1]),
                                      sycl::max(left_upper[2], right_upper[2]));

                // Update parent nodes using the volatile version of make_node
                // for synchronization
                make_node((volatile BVHPackedNodeHalf*)&bvhAll[mesh_id]
                              .node_lowers[parent_local_index],
                          lower, left_child, false);
                make_node((volatile BVHPackedNodeHalf*)&bvhAll[mesh_id]
                              .node_uppers[parent_local_index],
                          upper, right_child, false);

                // Move up to process the parent
                current_index = parent_local_index;
              } else {
                // We're the first child, parent not ready yet - terminate this
                // thread
                break;
              }
            }
          }
        });
  });

  // Finally, lets pack the leaf nodes based on threshold number of minimuim
  // primitives it must contain This is done to reduce the height of the tree
  // which will help speed up tree traversal
  auto pack_leaves_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 1024;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(bvh_data.total_nodes, work_group_size);
    h.depends_on({build_tree_event});
    h.parallel_for<PackLeavesKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, bvhAll = bvh_data.bvhAll, range_leftsAll = bvh_data.range_leftsAll,
         range_rightsAll = bvh_data.range_rightsAll,
         node_mesh_ids =
             bvh_data.node_mesh_ids] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_node_index = item.get_global_id(0);
          if (global_node_index < bvh_data.total_nodes) {
            uint32_t mesh_id = node_mesh_ids[global_node_index];
            uint32_t local_node_index =
                global_node_index - bvh_data.node_offsets[mesh_id];
            int depth = 1;
            int parent = bvhAll[mesh_id].node_parents[local_node_index];

            while (parent != -1) {
              int old_parent = parent;
              parent = bvhAll[mesh_id].node_parents[parent];
              depth++;
            }

            // converts LBB range left <= i <= right
            // to convention: left <= i < right
            int left = range_leftsAll[global_node_index];
            int right = range_rightsAll[global_node_index] + 1;

            if (right - left <=
                    static_cast<int>(BVHParams::kMinPrimitivesPerLeaf) ||
                depth >= static_cast<int>(BVHParams::kMaxDepth)) {
              bvh_data.bvhAll[mesh_id].node_lowers[local_node_index].b =
                  1;  // Make leaf
              // Set leaf ranges
              bvh_data.bvhAll[mesh_id].node_lowers[local_node_index].i = left;
              bvh_data.bvhAll[mesh_id].node_uppers[local_node_index].i = right;
            }
          }
        });
  });

  q_device.wait();

  SyclMemoryHelper::FreeBVHAllMeshTempMemory(memory_manager, bvh_data);
  bvh_built_ = true;
}

sycl::event BVHBroadPhase::refit(const DeviceMeshData& mesh_data,
                                 DeviceBVHData& bvh_data,
                                 sycl::event& element_aabb_event,
                                 SyclMemoryManager& memory_manager,
                                 sycl::queue& q_device) {
  // Reinitialize the num_childrenAll array to 0 for atomic operations
  auto init_children_event =
      memory_manager.Memset(bvh_data.num_childrenAll, bvh_data.total_nodes);

  // Reinitialize the indicesAll array to 0 for atomic operations
  const uint32_t work_group_size = 1024;
  const uint32_t global_elements =
      RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
  auto refit_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({init_children_event, element_aabb_event});
    h.parallel_for<RefitKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         element_mesh_ids = mesh_data.element_mesh_ids,
         bvhAll = bvh_data.bvhAll, num_childrenAll = bvh_data.num_childrenAll,
         node_offsets = bvh_data.node_offsets] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < mesh_data.total_elements) {
            uint32_t mesh_id = element_mesh_ids[global_eI];
            uint32_t global_element_offset = element_offsets[mesh_id];
            uint32_t global_node_offset = node_offsets[mesh_id];
            uint32_t local_element_index = global_eI - global_element_offset;
            BVH& bvh = bvhAll[mesh_id];
            bool is_leaf = bvh.node_lowers[local_element_index].b;
            int parent = bvh.node_parents[local_element_index];

            if (!is_leaf) {
              return;
            }
            // Set new bounding boxes for the leaf
            BVHPackedNodeHalf& lower = bvh.node_lowers[local_element_index];
            BVHPackedNodeHalf& upper = bvh.node_uppers[local_element_index];

            // Only set these new bounding boxes if the leaf is not a
            // muted leaf (parent of leaf has not been made leaf in
            // packing)
            if (!bvh.node_lowers[parent].b) {
              const uint32_t start = lower.i;
              const uint32_t end = upper.i;
              // Compute new AABB
              Vector3<double> lower_W(std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max());
              Vector3<double> upper_W(std::numeric_limits<double>::min(),
                                      std::numeric_limits<double>::min(),
                                      std::numeric_limits<double>::min());
              for (uint32_t local_primitive_index = start;
                   local_primitive_index < end; local_primitive_index++) {
                uint32_t unsorted_local_primitive_index =
                    indicesAll[local_primitive_index + global_element_offset];
                Vector3<double> lower_W_i =
                    element_aabb_min_W[unsorted_local_primitive_index +
                                       global_element_offset];
                Vector3<double> upper_W_i =
                    element_aabb_max_W[unsorted_local_primitive_index +
                                       global_element_offset];
                lower_W = ComponentwiseMin(lower_W, lower_W_i);
                upper_W = ComponentwiseMax(upper_W, upper_W_i);
              }
              // Set the new bounds for the leaf
              lower.x = lower_W[0];
              lower.y = lower_W[1];
              lower.z = lower_W[2];
              upper.x = upper_W[0];
              upper.y = upper_W[1];
              upper.z = upper_W[2];
            }

            // Now update hierarchy by moving upwards
            while (parent != -1) {
              uint32_t parent_global_index = global_node_offset + parent;
              sycl::atomic_fence(sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device);
              sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                               sycl::memory_scope::device>
                  atomic_children(num_childrenAll[parent_global_index]);
              int finished =
                  atomic_children.fetch_add(1, sycl::memory_order::acq_rel);

              if (finished == 1) {
                BVHPackedNodeHalf& parent_lower = bvh.node_lowers[parent];
                BVHPackedNodeHalf& parent_upper = bvh.node_uppers[parent];
                if (parent_lower.b) {
                  // a packed leaf node can still be a parent in LBVH,
                  // we need to recompute its bounds since we've lost
                  // its left and right child node index in the muting
                  // process
                  int parent_parent = bvh.node_parents[parent];

                  if (parent_parent == -1) {
                    // Root node is a leaf (very rare case where we have only a
                    // handful of elements in the mesh) Update the bounds like
                    // its a leaf node
                    const uint32_t start = parent_lower.i;
                    const uint32_t end = parent_upper.i;
                    Vector3<double> lower_W(std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::max());
                    Vector3<double> upper_W(std::numeric_limits<double>::min(),
                                            std::numeric_limits<double>::min(),
                                            std::numeric_limits<double>::min());
                    for (uint32_t local_primitive_index = start;
                         local_primitive_index < end; local_primitive_index++) {
                      uint32_t unsorted_local_primitive_index =
                          indicesAll[local_primitive_index +
                                     global_element_offset];
                      Vector3<double> lower_W_i =
                          element_aabb_min_W[unsorted_local_primitive_index +
                                             global_element_offset];
                      Vector3<double> upper_W_i =
                          element_aabb_max_W[unsorted_local_primitive_index +
                                             global_element_offset];
                      lower_W = ComponentwiseMin(lower_W, lower_W_i);
                      upper_W = ComponentwiseMax(upper_W, upper_W_i);
                    }
                    parent_lower.x = lower_W[0];
                    parent_lower.y = lower_W[1];
                    parent_lower.z = lower_W[2];
                    parent_upper.x = upper_W[0];
                    parent_upper.y = upper_W[1];
                    parent_upper.z = upper_W[2];
                  } else if (!bvh.node_lowers[parent_parent].b) {
                    const uint32_t start = parent_lower.i;
                    const uint32_t end = parent_upper.i;
                    Vector3<double> lower_W(std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::max());
                    Vector3<double> upper_W(std::numeric_limits<double>::min(),
                                            std::numeric_limits<double>::min(),
                                            std::numeric_limits<double>::min());
                    for (uint32_t local_primitive_index = start;
                         local_primitive_index < end; local_primitive_index++) {
                      uint32_t unsorted_local_primitive_index =
                          indicesAll[local_primitive_index +
                                     global_element_offset];
                      Vector3<double> lower_W_i =
                          element_aabb_min_W[unsorted_local_primitive_index +
                                             global_element_offset];
                      Vector3<double> upper_W_i =
                          element_aabb_max_W[unsorted_local_primitive_index +
                                             global_element_offset];
                      lower_W = ComponentwiseMin(lower_W, lower_W_i);
                      upper_W = ComponentwiseMax(upper_W, upper_W_i);
                    }
                    parent_lower.x = lower_W[0];
                    parent_lower.y = lower_W[1];
                    parent_lower.z = lower_W[2];
                    parent_upper.x = upper_W[0];
                    parent_upper.y = upper_W[1];
                    parent_upper.z = upper_W[2];
                  }
                } else {
                  // Parent is not a leaf so we recompute its bounds
                  // from its left and right children
                  const uint32_t left = parent_lower.i;
                  const uint32_t right = parent_upper.i;

                  Vector3<double> left_lower(bvh.node_lowers[left].x,
                                             bvh.node_lowers[left].y,
                                             bvh.node_lowers[left].z);
                  Vector3<double> left_upper(bvh.node_uppers[left].x,
                                             bvh.node_uppers[left].y,
                                             bvh.node_uppers[left].z);
                  Vector3<double> right_lower(bvh.node_lowers[right].x,
                                              bvh.node_lowers[right].y,
                                              bvh.node_lowers[right].z);
                  Vector3<double> right_upper(bvh.node_uppers[right].x,
                                              bvh.node_uppers[right].y,
                                              bvh.node_uppers[right].z);
                  Vector3<double> lower_W =
                      ComponentwiseMin(left_lower, right_lower);
                  Vector3<double> upper_W =
                      ComponentwiseMax(left_upper, right_upper);

                  // Set the new bounds for the parent
                  parent_lower.x = lower_W[0];
                  parent_lower.y = lower_W[1];
                  parent_lower.z = lower_W[2];
                  parent_upper.x = upper_W[0];
                  parent_upper.y = upper_W[1];
                  parent_upper.z = upper_W[2];
                }
                // Move up to process the parent
                parent = bvh.node_parents[parent];
              } else {
                break;
              }
            }
          }
        });
  });
  return refit_event;
}

sycl::event BVHBroadPhase::ComputeCollisionCounts(
    const uint32_t mesh_a, const uint32_t mesh_b, const DeviceBVHData& bvh_data,
    const DeviceMeshData& mesh_data, DeviceMeshACollisionCounters& cc,
    sycl::event& event_to_depend_on, sycl::queue& q_device) {
  // Number of primitives in mesh A
  uint32_t num_primitives_a = mesh_data.element_counts[mesh_a];
  // BVH of mesh B
  BVH& bvh_b = bvh_data.bvhAll[mesh_b];

  auto compute_collision_counts_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({event_to_depend_on});
    const uint32_t work_group_size = 64;
    const uint32_t elements_a =
        RoundUpToWorkGroupSize(num_primitives_a, work_group_size);
    h.parallel_for<ComputeCollisionCountsKernel>(
        sycl::nd_range<1>(sycl::range<1>(elements_a),
                          sycl::range<1>(work_group_size)),
        [=, node_lowers = bvh_b.node_lowers, node_uppers = bvh_b.node_uppers,
         num_primitives_a = num_primitives_a, mesh_a = mesh_a, mesh_b = mesh_b,
         element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         collision_counts = cc.collision_counts,
         max_pressures = mesh_data.max_pressures,
         min_pressures =
             mesh_data.min_pressures] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t elementId_A = item.get_global_id(0);
          if (elementId_A >= num_primitives_a) {
            return;
          }
          uint32_t element_offset_A = element_offsets[mesh_a];
          uint32_t element_offset_B = element_offsets[mesh_b];
          uint32_t global_elementId_A = elementId_A + element_offset_A;
          // We process in the order of morton code's.
          // This is because two elements that have similar morton codes
          // and are thus closer to each other would tend to traverse the tree
          // to similar depths.
          uint32_t local_tetId_A = indicesAll[global_elementId_A];
          // TODO (Huzaifa): This access will be uncoalesced since we are
          // accessing my morton order code. Evaluate trade off between thread
          // divergence if we don't do morton order code and memory access
          // pattern if we do.
          uint32_t global_primitive_index_A = local_tetId_A + element_offset_A;
          const Vector3<double> lower_W_Ai =
              element_aabb_min_W[global_primitive_index_A];
          const Vector3<double> upper_W_Ai =
              element_aabb_max_W[global_primitive_index_A];
          const double max_pressure_A = max_pressures[global_primitive_index_A];
          const double min_pressure_A = min_pressures[global_primitive_index_A];

          uint32_t query_stack[static_cast<uint32_t>(BVHParams::kMaxDepth)];
          // Start at root node
          query_stack[0] = *bvh_b.root;  // mesh local index
          uint32_t stack_nc = 1;         // num of nodes in stack
          uint32_t num_collisions = 0;

          while (stack_nc) {
            uint32_t node_index = query_stack[--stack_nc];
            BVHPackedNodeHalf& node_lower = node_lowers[node_index];
            BVHPackedNodeHalf& node_upper = node_uppers[node_index];

            if (!sycl_impl::AABBsIntersect(
                    lower_W_Ai, upper_W_Ai,
                    Vector3<double>(node_lower.x, node_lower.y, node_lower.z),
                    Vector3<double>(node_upper.x, node_upper.y,
                                    node_upper.z))) {
              continue;
            }
            const uint32_t left = node_lower.i;
            const uint32_t right = node_upper.i;

            if (node_lower.b) {
              // Leaf node can have more than 1 primitive - serially check each
              // collision
              for (uint32_t local_primitive_index = left;
                   local_primitive_index < right; local_primitive_index++) {
                uint32_t unsorted_local_primitive_index =
                    indicesAll[local_primitive_index + element_offset_B];
                uint32_t global_primitive_index_B =
                    unsorted_local_primitive_index + element_offset_B;
                const Vector3<double> lower_W_i =
                    element_aabb_min_W[global_primitive_index_B];
                const Vector3<double> upper_W_i =
                    element_aabb_max_W[global_primitive_index_B];
                const double max_pressure_B =
                    max_pressures[global_primitive_index_B];
                const double min_pressure_B =
                    min_pressures[global_primitive_index_B];
                if (sycl_impl::AABBsIntersect(lower_W_Ai, upper_W_Ai, lower_W_i,
                                              upper_W_i) &&
                    (sycl_impl::PressuresIntersect(
                        min_pressure_A, max_pressure_A, min_pressure_B,
                        max_pressure_B))) {
                  num_collisions++;
                }
              }
            } else {
              // Continue traversal
              query_stack[stack_nc++] = left;
              query_stack[stack_nc++] = right;
            }
          }

          // Store the counts
          // Storing in order of morton code
          collision_counts[elementId_A] = num_collisions;
        });
  });

  return compute_collision_counts_event;
}

sycl::event BVHBroadPhase::ComputeCollisionCountsAll(
    const uint32_t* meshAs, const uint32_t* meshBs,
    const DeviceBVHData& bvh_data, const DeviceMeshData& mesh_data,
    DeviceCollisionCountersMemoryChunk& counters_chunk,
    DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk,
    sycl::event& event_to_depend_on, sycl::queue& q_device) {
  auto compute_collision_counts_all_event = q_device.submit([&](sycl::handler&
                                                                    h) {
    h.depends_on({event_to_depend_on});
    const uint32_t work_group_size = 512;
    const uint32_t total_threads =
        RoundUpToWorkGroupSize(counters_chunk.size_, work_group_size);
    h.parallel_for<ComputeCollisionCountsAllKernel>(
        sycl::nd_range<1>(sycl::range<1>(total_threads),
                          sycl::range<1>(work_group_size)),
        [=, bvhAll = bvh_data.bvhAll,
         element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         global_collision_counts = counters_chunk.collision_counts,
         global_counters_offsets = counters_offsets_chunk.mesh_a_offsets,
         total_offsets = counters_offsets_chunk.size_,
         max_pressures = mesh_data.max_pressures,
         min_pressures =
             mesh_data.min_pressures] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_thread_id = item.get_global_id(0);
          if (global_thread_id >= counters_chunk.size_) {
            return;
          }

          // Get candidate pair number
          size_t candidate_pair_number =
              upper_bound_device(global_counters_offsets,
                                 global_counters_offsets + total_offsets,
                                 global_thread_id) -
              global_counters_offsets - 1;
          uint32_t mesh_a = meshAs[candidate_pair_number];
          uint32_t mesh_b = meshBs[candidate_pair_number];
          BVH& bvh_b = bvhAll[mesh_b];
          BVHPackedNodeHalf* node_lowers = bvh_b.node_lowers;
          BVHPackedNodeHalf* node_uppers = bvh_b.node_uppers;

          uint32_t element_offset_A = element_offsets[mesh_a];
          uint32_t element_offset_B = element_offsets[mesh_b];
          uint32_t elementId_A =
              global_thread_id - global_counters_offsets[candidate_pair_number];
          uint32_t global_elementId_A = elementId_A + element_offset_A;

          // We process in the order of morton code's.
          // This is because two elements that have similar morton codes
          // and are thus closer to each other would tend to traverse the
          // tree to similar depths.
          uint32_t local_tetId_A = indicesAll[global_elementId_A];
          // TODO (Huzaifa): This access will be uncoalesced since we are
          // accessing my morton order code. Evaluate trade off between
          // divergence if we don't do morton order code and memory
          // access pattern if we do.
          uint32_t global_primitive_index_A = local_tetId_A + element_offset_A;
          const Vector3<double> lower_W_Ai =
              element_aabb_min_W[global_primitive_index_A];
          const Vector3<double> upper_W_Ai =
              element_aabb_max_W[global_primitive_index_A];
          const double max_pressure_A = max_pressures[global_primitive_index_A];
          const double min_pressure_A = min_pressures[global_primitive_index_A];

          uint32_t query_stack[static_cast<uint32_t>(BVHParams::kMaxDepth)];
          // Start at root node
          query_stack[0] = *bvh_b.root;  // mesh local index
          uint32_t stack_nc = 1;         // num of nodes in stack
          uint32_t num_collisions = 0;

          while (stack_nc) {
            uint32_t node_index = query_stack[--stack_nc];
            BVHPackedNodeHalf& node_lower = node_lowers[node_index];
            BVHPackedNodeHalf& node_upper = node_uppers[node_index];

            if (!sycl_impl::AABBsIntersect(
                    lower_W_Ai, upper_W_Ai,
                    Vector3<double>(node_lower.x, node_lower.y, node_lower.z),
                    Vector3<double>(node_upper.x, node_upper.y,
                                    node_upper.z))) {
              continue;
            }
            const uint32_t left = node_lower.i;
            const uint32_t right = node_upper.i;

            if (node_lower.b) {
              // Leaf node can have more than 1 primitive - serially check each
              // collision
              for (uint32_t local_primitive_index = left;
                   local_primitive_index < right; local_primitive_index++) {
                uint32_t unsorted_local_primitive_index =
                    indicesAll[local_primitive_index + element_offset_B];
                uint32_t global_primitive_index_B =
                    unsorted_local_primitive_index + element_offset_B;
                const Vector3<double> lower_W_i =
                    element_aabb_min_W[global_primitive_index_B];
                const Vector3<double> upper_W_i =
                    element_aabb_max_W[global_primitive_index_B];
                const double max_pressure_B =
                    max_pressures[global_primitive_index_B];
                const double min_pressure_B =
                    min_pressures[global_primitive_index_B];
                if (sycl_impl::AABBsIntersect(lower_W_Ai, upper_W_Ai, lower_W_i,
                                              upper_W_i) &&
                    (sycl_impl::PressuresIntersect(
                        min_pressure_A, max_pressure_A, min_pressure_B,
                        max_pressure_B))) {
                  num_collisions++;
                }
              }
            } else {
              // Continue traversal
              query_stack[stack_nc++] = left;
              query_stack[stack_nc++] = right;
            }
          }

          // Store the counts
          // Storing in order of morton code
          global_collision_counts[global_thread_id] = num_collisions;
        });
  });
  return compute_collision_counts_all_event;
}

sycl::event BVHBroadPhase::ComputeCollisionPairs(
    const uint32_t mesh_a, const uint32_t mesh_b, const DeviceBVHData& bvh_data,
    const DeviceMeshData& mesh_data, DeviceMeshACollisionCounters& cc,
    DeviceMeshPairCollidingIndices& ci, sycl::queue& q_device) {
  // Number of primitives in mesh A
  uint32_t num_primitives_a = mesh_data.element_counts[mesh_a];
  // BVH of mesh B
  BVH& bvh_b = bvh_data.bvhAll[mesh_b];

  auto compute_collision_pairs_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 64;
    const uint32_t elements_a =
        RoundUpToWorkGroupSize(num_primitives_a, work_group_size);
    h.parallel_for<ComputeCollisionPairsKernel>(
        sycl::nd_range<1>(sycl::range<1>(elements_a),
                          sycl::range<1>(work_group_size)),
        [=, node_lowers = bvh_b.node_lowers, node_uppers = bvh_b.node_uppers,
         num_primitives_a = num_primitives_a, mesh_a = mesh_a, mesh_b = mesh_b,
         element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         collision_counts = cc.collision_counts,
         collision_indices_B = ci.collision_indices_B,
         collision_indices_A = ci.collision_indices_A,
         max_pressures = mesh_data.max_pressures,
         min_pressures =
             mesh_data.min_pressures] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t elementId_A = item.get_global_id(0);
          if (elementId_A >= num_primitives_a) {
            return;
          }
          uint32_t element_offset_A = element_offsets[mesh_a];
          uint32_t element_offset_B = element_offsets[mesh_b];
          uint32_t global_elementId_A = elementId_A + element_offset_A;
          // We process in the order of morton code's.
          // This is because two elements that have similar morton codes
          // and are thus closer to each other would tend to traverse the tree
          // to similar depths.
          uint32_t local_tetId_A = indicesAll[global_elementId_A];
          uint32_t global_primitive_index_A = local_tetId_A + element_offset_A;
          // Get the offset that this AABB writes to
          uint32_t write_offset = collision_counts[elementId_A];
          // TODO (Huzaifa): This access will be uncoalesced since we are
          // accessing my morton order code. Evaluate trade off between thread
          // divergence if we don't do morton order code and memory access
          // pattern if we do.
          Vector3<double> lower_W_Ai =
              element_aabb_min_W[local_tetId_A + element_offset_A];
          Vector3<double> upper_W_Ai =
              element_aabb_max_W[local_tetId_A + element_offset_A];
          const double max_pressure_A = max_pressures[global_primitive_index_A];
          const double min_pressure_A = min_pressures[global_primitive_index_A];

          uint32_t query_stack[static_cast<uint32_t>(BVHParams::kMaxDepth)];
          // Start at root node
          query_stack[0] = *bvh_b.root;  // mesh local index
          uint32_t stack_nc = 1;         // num of nodes in stack
          // number of collision indices already written
          uint32_t num_collisions = 0;

          while (stack_nc) {
            uint32_t node_index = query_stack[--stack_nc];
            BVHPackedNodeHalf& node_lower = node_lowers[node_index];
            BVHPackedNodeHalf& node_upper = node_uppers[node_index];

            if (!sycl_impl::AABBsIntersect(
                    lower_W_Ai, upper_W_Ai,
                    Vector3<double>(node_lower.x, node_lower.y, node_lower.z),
                    Vector3<double>(node_upper.x, node_upper.y,
                                    node_upper.z))) {
              continue;
            }
            const uint32_t left = node_lower.i;
            const uint32_t right = node_upper.i;

            if (node_lower.b) {
              // Leaf node can have more than 1 primitive - serially check
              // each collision
              for (uint32_t local_primitive_index = left;
                   local_primitive_index < right; local_primitive_index++) {
                uint32_t unsorted_local_primitive_index =
                    indicesAll[local_primitive_index + element_offset_B];
                uint32_t global_primitive_index_B =
                    unsorted_local_primitive_index + element_offset_B;
                const Vector3<double> lower_W_i =
                    element_aabb_min_W[global_primitive_index_B];
                const Vector3<double> upper_W_i =
                    element_aabb_max_W[global_primitive_index_B];
                const double max_pressure_B =
                    max_pressures[global_primitive_index_B];
                const double min_pressure_B =
                    min_pressures[global_primitive_index_B];
                if (sycl_impl::AABBsIntersect(lower_W_Ai, upper_W_Ai, lower_W_i,
                                              upper_W_i) &&
                    (sycl_impl::PressuresIntersect(
                        min_pressure_A, max_pressure_A, min_pressure_B,
                        max_pressure_B))) {
                  // Where to write - this element offset + number of
                  // collisions this node has processed What to write - global
                  // index of element from mesh B. This can directly index
                  // into our global arrays where data is stored
                  collision_indices_B[write_offset + num_collisions] =
                      global_primitive_index_B;
                  collision_indices_A[write_offset + num_collisions] =
                      global_primitive_index_A;
                  num_collisions++;
                }
              }
            } else {
              // Continue traversal
              query_stack[stack_nc++] = left;
              query_stack[stack_nc++] = right;
            }
          }
        });
  });

  return compute_collision_pairs_event;
}

sycl::event BVHBroadPhase::ComputeCollisionPairsAll(
    const uint32_t* meshAs, const uint32_t* meshBs,
    const DeviceBVHData& bvh_data, const DeviceMeshData& mesh_data,
    DeviceCollisionCountersMemoryChunk& counters_chunk,
    DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk,
    DeviceCollidingIndicesMemoryChunk& pair_chunk, sycl::queue& q_device) {
  auto compute_collision_pairs_all_event = q_device.submit([&](sycl::handler&
                                                                   h) {
    const uint32_t work_group_size = 512;
    const uint32_t total_threads =
        RoundUpToWorkGroupSize(counters_chunk.size_, work_group_size);
    h.parallel_for<ComputeCollisionPairsAllKernel>(
        sycl::nd_range<1>(sycl::range<1>(total_threads),
                          sycl::range<1>(work_group_size)),
        [=, bvhAll = bvh_data.bvhAll,
         element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         global_collision_counts = counters_chunk.collision_counts,
         global_counters_offsets = counters_offsets_chunk.mesh_a_offsets,
         collision_indices_B = pair_chunk.collision_indices_B,
         collision_indices_A = pair_chunk.collision_indices_A,
         total_offsets = counters_offsets_chunk.size_,
         max_pressures = mesh_data.max_pressures,
         min_pressures =
             mesh_data.min_pressures] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_thread_id = item.get_global_id(0);
          if (global_thread_id >= counters_chunk.size_) {
            return;
          }

          // Get candidate pair number
          size_t candidate_pair_number =
              upper_bound_device(global_counters_offsets,
                                 global_counters_offsets + total_offsets,
                                 global_thread_id) -
              global_counters_offsets - 1;
          uint32_t mesh_a = meshAs[candidate_pair_number];
          uint32_t mesh_b = meshBs[candidate_pair_number];
          BVH& bvh_b = bvhAll[mesh_b];
          BVHPackedNodeHalf* node_lowers = bvh_b.node_lowers;
          BVHPackedNodeHalf* node_uppers = bvh_b.node_uppers;

          uint32_t element_offset_A = element_offsets[mesh_a];
          uint32_t element_offset_B = element_offsets[mesh_b];
          uint32_t elementId_A =
              global_thread_id - global_counters_offsets[candidate_pair_number];
          uint32_t global_elementId_A = elementId_A + element_offset_A;

          // We process in the order of morton code's.
          // This is because two elements that have similar morton codes
          // and are thus closer to each other would tend to traverse the
          // tree to similar depths.
          uint32_t local_tetId_A = indicesAll[global_elementId_A];
          // TODO (Huzaifa): This access will be uncoalesced since we are
          // accessing my morton order code. Evaluate trade off between
          // divergence if we don't do morton order code and memory
          // access pattern if we do.
          uint32_t global_primitive_index_A = local_tetId_A + element_offset_A;

          uint32_t write_offset = global_collision_counts[global_thread_id];
          // TODO (Huzaifa): This access will be uncoalesced since we are
          // accessing my morton order code. Evaluate trade off between thread
          // divergence if we don't do morton order code and memory access
          // pattern if we do.
          Vector3<double> lower_W_Ai =
              element_aabb_min_W[global_primitive_index_A];
          Vector3<double> upper_W_Ai =
              element_aabb_max_W[global_primitive_index_A];
          const double max_pressure_A = max_pressures[global_primitive_index_A];
          const double min_pressure_A = min_pressures[global_primitive_index_A];

          uint32_t query_stack[static_cast<uint32_t>(BVHParams::kMaxDepth)];
          // Start at root node
          query_stack[0] = *bvh_b.root;  // mesh local index
          uint32_t stack_nc = 1;         // num of nodes in stack
          // number of collision indices already written
          uint32_t num_collisions = 0;

          while (stack_nc) {
            uint32_t node_index = query_stack[--stack_nc];
            BVHPackedNodeHalf& node_lower = node_lowers[node_index];
            BVHPackedNodeHalf& node_upper = node_uppers[node_index];

            if (!sycl_impl::AABBsIntersect(
                    lower_W_Ai, upper_W_Ai,
                    Vector3<double>(node_lower.x, node_lower.y, node_lower.z),
                    Vector3<double>(node_upper.x, node_upper.y,
                                    node_upper.z))) {
              continue;
            }
            const uint32_t left = node_lower.i;
            const uint32_t right = node_upper.i;

            if (node_lower.b) {
              // Leaf node can have more than 1 primitive - serially check
              // each collision
              for (uint32_t local_primitive_index = left;
                   local_primitive_index < right; local_primitive_index++) {
                uint32_t unsorted_local_primitive_index =
                    indicesAll[local_primitive_index + element_offset_B];
                uint32_t global_primitive_index_B =
                    unsorted_local_primitive_index + element_offset_B;
                const Vector3<double> lower_W_i =
                    element_aabb_min_W[global_primitive_index_B];
                const Vector3<double> upper_W_i =
                    element_aabb_max_W[global_primitive_index_B];
                const double max_pressure_B =
                    max_pressures[global_primitive_index_B];
                const double min_pressure_B =
                    min_pressures[global_primitive_index_B];
                if (sycl_impl::AABBsIntersect(lower_W_Ai, upper_W_Ai, lower_W_i,
                                              upper_W_i) &&
                    (sycl_impl::PressuresIntersect(
                        min_pressure_A, max_pressure_A, min_pressure_B,
                        max_pressure_B))) {
                  // Where to write - this element offset + number of
                  // collisions this node has processed What to write - global
                  // index of element from mesh B. This can directly index
                  // into our global arrays where data is stored
                  collision_indices_B[write_offset + num_collisions] =
                      global_primitive_index_B;
                  collision_indices_A[write_offset + num_collisions] =
                      global_primitive_index_A;
                  num_collisions++;
                }
              }
            } else {
              // Continue traversal
              query_stack[stack_nc++] = left;
              query_stack[stack_nc++] = right;
            }
          }
        });
  });
  return compute_collision_pairs_all_event;
}

void BVHBroadPhase::BroadPhase(
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
    sycl::queue& q_device) {
  auto policy = oneapi::dpl::execution::make_device_policy(q_device);
  // Run a refit with the new AABBs - If the BVH is just built, then we don't
  // need to refit on the very first time step
  // By default, we refit every single mesh every time step because otherwise
  // the number of nodes will be too little for the GPU
  // TODO(Huzaifa): Refit only colliding meshes and compare the performance
  sycl::event refit_event;
  // if (!IsBVHRefitted()) {
  refit_event =
      refit(mesh_data, bvh_data, element_aabb_event, memory_manager, q_device);
  // }

  // Process the collision pairs
  // First the number of collisions for each element of mesh A is computed.
  // sycl::event event_to_depend_on =
  //     IsBVHRefitted() ? element_aabb_event : refit_event;
  // sycl::event event_to_depend_on = refit_event;
  // std::unordered_map<uint64_t, sycl::event> pair_events_map;
  // for (auto& [key, value] : collision_candidates_to_data) {
  //   auto& [cc, ci] = value;
  //   auto [mesh_a, mesh_b] = key_to_pair(key);
  //   pair_events_map[key] = ComputeCollisionCounts(
  //       mesh_a, mesh_b, bvh_data, mesh_data, cc, event_to_depend_on,
  //       q_device);
  // }
  sycl::event event_to_depend_on = refit_event;
  sycl::event compute_collision_counts_all_event = ComputeCollisionCountsAll(
      mesh_pair_ids.meshAs, mesh_pair_ids.meshBs, bvh_data, mesh_data,
      counters_chunk_, counters_offsets_chunk_, event_to_depend_on, q_device);
  compute_collision_counts_all_event.wait_and_throw();

  // for (int i = 0; i < num_mesh_collisions; i++) {
  //   uint32_t mesh_a = mesh_pair_ids.meshAs[i];
  //   uint32_t mesh_b = mesh_pair_ids.meshBs[i];
  //   uint64_t col_key = key(mesh_a, mesh_b);
  //   auto& [cc, ci] = collision_candidates_to_data[col_key];
  //   // pair_events_map[key].wait();
  //   // Store the last element's collision count
  //   uint32_t last_element_collision_count = 0;
  //   q_device
  //       .memcpy(&last_element_collision_count,
  //               cc.collision_counts + cc.size_ - 1, sizeof(uint32_t))
  //       .wait();
  //   cc.last_element_collision_count = last_element_collision_count;
  //   // Scan and get total collision count
  //   oneapi::dpl::exclusive_scan(policy, cc.collision_counts,
  //                               cc.collision_counts + cc.size_,
  //                               cc.collision_counts,
  //                               static_cast<uint32_t>(0));
  // }
  // Wait for all the scans to complete

  // Get the last element count of last mesh
  uint32_t last_element_collision_count = 0;
  q_device
      .memcpy(&last_element_collision_count,
              counters_chunk_.collision_counts + counters_chunk_.size_ - 1,
              sizeof(uint32_t))
      .wait();
  counters_chunk_.last_element_collision_count = last_element_collision_count;
  // Scan the entire collision count chunk
  oneapi::dpl::exclusive_scan(
      policy, counters_chunk_.collision_counts,
      counters_chunk_.collision_counts + counters_chunk_.size_,
      counters_chunk_.collision_counts, static_cast<uint32_t>(0));
  q_device.wait_and_throw();
  // Get the total collisions
  uint32_t total_collisions = 0;
  q_device
      .memcpy(&total_collisions,
              counters_chunk_.collision_counts + counters_chunk_.size_ - 1,
              sizeof(uint32_t))
      .wait();
  total_collisions += last_element_collision_count;
  counters_chunk_.total_collisions = total_collisions;
  pair_chunk_.size_ = total_collisions;

  // // Compute the total collisions and assign the memory required to compute
  // // the collision pairs
  // for (int i = 0; i < num_mesh_collisions; i++) {
  //   uint32_t mesh_a = mesh_pair_ids.meshAs[i];
  //   uint32_t mesh_b = mesh_pair_ids.meshBs[i];
  //   uint64_t col_key = key(mesh_a, mesh_b);
  //   auto& [cc, ci] = collision_candidates_to_data[col_key];
  //   uint32_t total_collisions = 0;
  //   q_device
  //       .memcpy(&total_collisions, cc.collision_counts + cc.size_ - 1,
  //               sizeof(uint32_t))
  //       .wait();
  //   cc.total_collisions = total_collisions +
  //   cc.last_element_collision_count;
  // }

  // Get pointers to the chunk of collision pair indices memory
  // that we need to write to
  // uint32_t offset = 0;
  // for (int i = 0; i < num_mesh_collisions; i++) {
  //   uint32_t mesh_a = mesh_pair_ids.meshAs[i];
  //   uint32_t mesh_b = mesh_pair_ids.meshBs[i];
  //   uint64_t col_key = key(mesh_a, mesh_b);
  //   auto& [cc, ci] = collision_candidates_to_data[col_key];
  //   SyclMemoryHelper::ResizeDeviceMeshPairCollidingIndicesMemory(
  //       memory_manager, ci, pair_chunk_, cc.total_collisions, offset);
  //   offset += cc.total_collisions;
  // }
  // pair_chunk_.size_ = offset;

  // Compute the actual collision pairs
  // std::vector<sycl::event> pair_events_vec;
  // for (int i = 0; i < num_mesh_collisions; i++) {
  //   uint32_t mesh_a = mesh_pair_ids.meshAs[i];
  //   uint32_t mesh_b = mesh_pair_ids.meshBs[i];
  //   uint64_t col_key = key(mesh_a, mesh_b);
  //   auto& [cc, ci] = collision_candidates_to_data[col_key];
  //   pair_events_vec.push_back(ComputeCollisionPairs(
  //       mesh_a, mesh_b, bvh_data, mesh_data, cc, ci, q_device));
  // }
  sycl::event compute_collision_pairs_all_event = ComputeCollisionPairsAll(
      mesh_pair_ids.meshAs, mesh_pair_ids.meshBs, bvh_data, mesh_data,
      counters_chunk_, counters_offsets_chunk_, pair_chunk_, q_device);
  compute_collision_pairs_all_event.wait();

  // sycl::event::wait_and_throw(pair_events_vec);
  // We need to refit every timestep except for the first one
  SetBVHRefitted(false);
}
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake