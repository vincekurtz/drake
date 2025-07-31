#pragma once

#include <array>

#include "geometry/proximity/sycl/utils/sycl_kernel_utils.h"
#include <sycl/sycl.hpp>

#include "drake/geometry/proximity/sycl/utils/sycl_memory_manager.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

class GenerateCollisionFilterKernel;

/**
 * @brief Performs naive broad phase collision detection using AABB and pressure
 * field intersection
 *
 * This function implements a naive broad phase collision detection algorithm
 * that:
 * 1. Computes AABBs for all tetrahedral elements
 * 2. Generates collision filter based on AABB overlap and pressure field
 * intersection
 *
 * @param q_device SYCL queue for device execution
 * @param mesh_data Device mesh data containing vertices, elements, and pressure
 * information
 * @param collision_data Device collision data for storing collision filter
 * results
 * @param total_elements Total number of tetrahedral elements across all
 * geometries
 * @param total_checks Total number of collision checks to perform
 * @param transform_vertices_event Event to wait for before starting AABB
 * computation
 * @param collision_filter_memset_event Event to wait for before starting
 * collision filter generation
 * @return std::pair<sycl::event, sycl::event> Pair of events for AABB
 * computation and collision filter generation
 */
inline sycl::event NaiveBroadPhase(sycl::queue& q_device,
                                   const DeviceMeshData& mesh_data,
                                   const DeviceCollisionData& collision_data,
                                   uint32_t total_elements,
                                   uint32_t total_checks,
                                   sycl::event element_aabb_event,
                                   sycl::event collision_filter_memset_event) {
  // =========================================
  // Generate collision filter with the AABBs that we computed
  // =========================================
  auto generate_collision_filter_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({element_aabb_event, collision_filter_memset_event});

    const uint32_t work_group_size = 1024;
    const uint32_t global_checks =
        RoundUpToWorkGroupSize(total_checks, work_group_size);

    h.parallel_for<GenerateCollisionFilterKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_checks),
                          sycl::range<1>(work_group_size)),
        [=, collision_filter = collision_data.collision_filter,
         collision_filter_host_body_index =
             collision_data.collision_filter_host_body_index,
         geom_collision_filter_check_offsets =
             collision_data.geom_collision_filter_check_offsets,
         geom_collision_filter_num_cols =
             collision_data.geom_collision_filter_num_cols,
         element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         min_pressures = mesh_data.min_pressures,
         max_pressures = mesh_data.max_pressures,
         total_checks_ = total_checks] [[intel::kernel_args_restrict]]

#ifdef __NVPTX__
        [[sycl::reqd_work_group_size(1024)]]
#endif
        (sycl::nd_item<1> item) {
          const uint32_t check_index = item.get_global_id(0);
          if (check_index >= total_checks_) return;

          const uint32_t host_body_index =
              collision_filter_host_body_index[check_index];

          // What elements is this check_index checking?
          // host_body_index is the geometry index that element A belongs to
          uint32_t num_of_checks_offset =
              geom_collision_filter_check_offsets[host_body_index];
          const uint32_t geom_local_check_number =
              check_index - num_of_checks_offset;

          const uint32_t A_element_index =
              element_offsets[host_body_index] +
              geom_local_check_number /
                  geom_collision_filter_num_cols[host_body_index];
          const uint32_t B_element_index =
              element_offsets[host_body_index + 1] +
              geom_local_check_number %
                  geom_collision_filter_num_cols[host_body_index];

          // Default to not colliding.
          // collision_filter[check_index] = 0;

          // First check if the pressure fields of the elements intersect
          if (max_pressures[B_element_index] < min_pressures[A_element_index] ||
              max_pressures[A_element_index] < min_pressures[B_element_index]) {
            return;
          }

          // We have two element index, now just check their AABB
          // A element AABB
          // min
          for (int i = 0; i < 3; ++i) {
            if (element_aabb_max_W[B_element_index][i] <
                element_aabb_min_W[A_element_index][i])
              return;
            if (element_aabb_max_W[A_element_index][i] <
                element_aabb_min_W[B_element_index][i])
              return;
          }

          collision_filter[check_index] = 1;
        });
  });

  return generate_collision_filter_event;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake