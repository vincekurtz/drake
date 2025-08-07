#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"

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

#include "geometry/proximity/sycl/bvh/sycl_bvh.h"
#include <oneapi/dpl/execution>  // For execution policies
#include <oneapi/dpl/numeric>    // For exclusive_scan
#include <sycl/sycl.hpp>

#include "drake/common/problem_size_logger.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/geometry/proximity/sycl/utils/sycl_contact_polygon.h"
#include "drake/geometry/proximity/sycl/utils/sycl_equilibrium_plane.h"
#include "drake/geometry/proximity/sycl/utils/sycl_hydroelastic_surface_creator.h"
#include "drake/geometry/proximity/sycl/utils/sycl_naive_broad_phase.h"
#include "drake/geometry/proximity/sycl/utils/sycl_tetrahedron_slicing.h"
#include "drake/geometry/proximity/sycl/utils/sycl_timing_logger.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

#ifdef __SYCL_DEVICE_ONLY__
#define DRAKE_SYCL_DEVICE_INLINE [[sycl::device]]
#else
#define DRAKE_SYCL_DEVICE_INLINE
#endif

// Forward declarations for kernel names
class TransformVerticesKernel;
class TransformInwardNormalsKernel;
class TransformPressureGradientsKernel;
class ComputeElementAABBKernel;
class GenerateCollisionFilterKernel;
class FillNarrowPhaseCheckIndicesKernel;
class FillValidPolygonIndicesKernel;
class CompactPolygonDataKernel;
class TransformInwardNormalsAndPressureKernel;

// Implementation class for SyclProximityEngine that contains all SYCL-specific
// code
class SyclProximityEngine::Impl {
 public:
  // Default constructor
  Impl()
      : q_device_(InitializeQueue()), mem_mgr_(q_device_), timing_logger_() {}

  // Constructor that initializes with soft geometries
  Impl(const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
           soft_geometries,
       const std::unordered_map<GeometryId, Vector3<double>>& total_lower_map,
       const std::unordered_map<GeometryId, Vector3<double>>& total_upper_map)
      : q_device_(InitializeQueue()), mem_mgr_(q_device_), timing_logger_() {
    DRAKE_THROW_UNLESS(soft_geometries.size() > 0);

    // #ifdef DRAKE_SYCL_TIMING_ENABLED
    //     timing_logger_.SetEnabled(true);
    // #endif

    // Extract and sort geometry IDs for deterministic ordering
    sorted_ids_.reserve(soft_geometries.size());
    for (const auto& [id, _] : soft_geometries) {
      sorted_ids_.push_back(id);
    }
    std::sort(sorted_ids_.begin(), sorted_ids_.end());

    sorted_total_lower_.reserve(sorted_ids_.size());
    sorted_total_upper_.reserve(sorted_ids_.size());

    for (const auto& id : sorted_ids_) {
      sorted_total_lower_.push_back(total_lower_map.at(id));
      sorted_total_upper_.push_back(total_upper_map.at(id));
    }

    // Get number of geometries
    num_geometries_ = soft_geometries.size();

    SyclMemoryHelper::AllocateMeshMemory(mem_mgr_, mesh_data_, num_geometries_);
    SyclMemoryHelper::AllocateBVHAllMeshMemory(mem_mgr_, bvh_data_,
                                               num_geometries_);
    SyclMemoryHelper::AllocateDeviceCollisionCountersOffsetsMemoryChunk(
        mem_mgr_, counters_offsets_chunk_, num_geometries_);
    bvh_data_.num_meshes = num_geometries_;

    // First compute totals and build lookup data
    total_elements_ = 0;
    total_vertices_ = 0;
    total_nodes_ = 0;

    // Use the sorted IDs to ensure deterministic ordering
    for (uint32_t id_index = 0; id_index < sorted_ids_.size(); ++id_index) {
      const GeometryId& id = sorted_ids_[id_index];
      const hydroelastic::SoftGeometry& soft_geometry = soft_geometries.at(id);
      const hydroelastic::SoftMesh& soft_mesh = soft_geometry.soft_mesh();
      const VolumeMesh<double>& mesh = soft_mesh.mesh();

      // Store the geometry's ID
      mesh_data_.geometry_ids[id_index] = id;

      // Store offsets and counts directly (no memcpy needed with shared memory)
      mesh_data_.element_offsets[id_index] = total_elements_;
      mesh_data_.vertex_offsets[id_index] = total_vertices_;
      bvh_data_.node_offsets[id_index] = total_nodes_;

      const uint32_t num_elements = mesh.num_elements();
      const uint32_t num_vertices = mesh.num_vertices();
      mesh_data_.element_counts[id_index] = num_elements;
      mesh_data_.vertex_counts[id_index] = num_vertices;
      bvh_data_.node_counts_per_mesh[id_index] = 2 * num_elements - 1;
      bvh_data_.bvhAll[id_index].max_nodes = 2 * num_elements - 1;

      // Update totals
      total_elements_ += num_elements;
      total_vertices_ += num_vertices;
      total_nodes_ += 2 * num_elements - 1;
    }

    // TODO (Huzaifa) - Just use this everywhere and remove total_elements_ and
    // total_vertices_ from SyclProximityEngine::Impl
    mesh_data_.total_elements = total_elements_;
    mesh_data_.total_vertices = total_vertices_;
    bvh_data_.total_nodes = total_nodes_;

    SyclMemoryHelper::AllocateMeshElementVerticesMemory(
        mem_mgr_, mesh_data_, total_elements_, total_vertices_);
    SyclMemoryHelper::AllocateBVHAllMeshNodeCountsMemory(mem_mgr_, bvh_data_);
    SyclMemoryHelper::AllocateBVHAllMeshTempMemory(mem_mgr_, bvh_data_,
                                                   total_elements_);

    // Create host vectors to pack all data before transferring to device
    std::vector<std::array<int, 4>> host_elements(total_elements_);
    std::vector<uint32_t> host_element_mesh_ids(total_elements_);
    std::vector<uint32_t> host_node_mesh_ids(total_nodes_);
    std::vector<Vector3<double>> host_vertices_M(total_vertices_);
    std::vector<double> host_pressures(total_vertices_);
    std::vector<uint32_t> host_vertex_mesh_ids(total_vertices_);
    std::vector<std::array<Vector3<double>, 4>> host_inward_normals_M(
        total_elements_);
    std::vector<double> host_min_pressures(total_elements_);
    std::vector<double> host_max_pressures(total_elements_);
    std::vector<Vector4<double>> host_gradient_M_pressure_at_Mo(
        total_elements_);

    // Use the sorted IDs for deterministic ordering
    for (uint32_t id_index = 0; id_index < sorted_ids_.size(); ++id_index) {
      const GeometryId& id = sorted_ids_[id_index];
      const hydroelastic::SoftGeometry& soft_geometry = soft_geometries.at(id);
      const hydroelastic::SoftMesh& soft_mesh = soft_geometry.soft_mesh();
      const VolumeMesh<double>& mesh = soft_mesh.mesh();
      const VolumeMeshFieldLinear<double, double>& pressure_field =
          soft_mesh.pressure();

      // Direct access to shared memory values
      uint32_t element_offset = mesh_data_.element_offsets[id_index];
      uint32_t node_offset = bvh_data_.node_offsets[id_index];
      uint32_t vertex_offset = mesh_data_.vertex_offsets[id_index];
      uint32_t num_elements = mesh_data_.element_counts[id_index];
      uint32_t num_nodes = bvh_data_.node_counts_per_mesh[id_index];
      uint32_t num_vertices = mesh_data_.vertex_counts[id_index];

      // Pack elements
      const auto& mesh_elements = mesh.tetrahedra();
      for (uint32_t i = 0; i < num_elements; ++i) {
        const std::array<int, 4>& vertices = mesh_elements[i].getAllVertices();
        host_elements[element_offset + i] = vertices;
      }

      // Pack element mesh IDs
      for (uint32_t i = 0; i < num_elements; ++i) {
        host_element_mesh_ids[element_offset + i] = id_index;
      }

      // Pack node mesh IDs
      for (uint32_t i = 0; i < num_nodes; ++i) {
        host_node_mesh_ids[node_offset + i] = id_index;
      }

      // Pack vertices
      for (uint32_t i = 0; i < num_vertices; ++i) {
        host_vertices_M[vertex_offset + i] = mesh.vertices()[i];
      }

      // Pack pressures
      for (uint32_t i = 0; i < num_vertices; ++i) {
        host_pressures[vertex_offset + i] = pressure_field.values()[i];
      }

      // Pack vertex mesh IDs
      for (uint32_t i = 0; i < num_vertices; ++i) {
        host_vertex_mesh_ids[vertex_offset + i] = id_index;
      }

      // Pack inward normals
      for (uint32_t i = 0; i < num_elements; ++i) {
        host_inward_normals_M[element_offset + i] = mesh.inward_normals()[i];
      }

      // Pack min pressures
      for (uint32_t i = 0; i < num_elements; ++i) {
        host_min_pressures[element_offset + i] = pressure_field.min_values()[i];
      }

      // Pack max pressures
      for (uint32_t i = 0; i < num_elements; ++i) {
        host_max_pressures[element_offset + i] = pressure_field.max_values()[i];
      }

      // Pack gradient and pressure data
      const auto& gradients = pressure_field.gradients();
      const auto& pressuresat_Mo = pressure_field.values_at_Mo();

      for (uint32_t i = 0; i < num_elements; ++i) {
        host_gradient_M_pressure_at_Mo[i + element_offset][0] =
            gradients[i][0];  // x component
        host_gradient_M_pressure_at_Mo[i + element_offset][1] =
            gradients[i][1];  // y component
        host_gradient_M_pressure_at_Mo[i + element_offset][2] =
            gradients[i][2];  // z component
        host_gradient_M_pressure_at_Mo[i + element_offset][3] =
            pressuresat_Mo[i];  // pressure at Mo
      }
    }

    // Transfer all data to device at once
    std::vector<sycl::event> transfer_events;

    transfer_events.push_back(
        q_device_.memcpy(mesh_data_.elements, host_elements.data(),
                         total_elements_ * sizeof(std::array<int, 4>)));

    transfer_events.push_back(q_device_.memcpy(
        mesh_data_.element_mesh_ids, host_element_mesh_ids.data(),
        total_elements_ * sizeof(uint32_t)));

    transfer_events.push_back(
        q_device_.memcpy(bvh_data_.node_mesh_ids, host_node_mesh_ids.data(),
                         total_nodes_ * sizeof(uint32_t)));

    transfer_events.push_back(
        q_device_.memcpy(mesh_data_.vertices_M, host_vertices_M.data(),
                         total_vertices_ * sizeof(Vector3<double>)));

    transfer_events.push_back(
        q_device_.memcpy(mesh_data_.pressures, host_pressures.data(),
                         total_vertices_ * sizeof(double)));

    transfer_events.push_back(q_device_.memcpy(
        mesh_data_.vertex_mesh_ids, host_vertex_mesh_ids.data(),
        total_vertices_ * sizeof(uint32_t)));

    transfer_events.push_back(q_device_.memcpy(
        mesh_data_.inward_normals_M, host_inward_normals_M.data(),
        total_elements_ * sizeof(std::array<Vector3<double>, 4>)));

    transfer_events.push_back(
        q_device_.memcpy(mesh_data_.min_pressures, host_min_pressures.data(),
                         total_elements_ * sizeof(double)));

    transfer_events.push_back(
        q_device_.memcpy(mesh_data_.max_pressures, host_max_pressures.data(),
                         total_elements_ * sizeof(double)));

    transfer_events.push_back(
        q_device_.memcpy(mesh_data_.gradient_M_pressure_at_Mo,
                         host_gradient_M_pressure_at_Mo.data(),
                         total_elements_ * sizeof(Vector4<double>)));

    // We try and heuristically estimate the number of polygons as 1% of the
    // total checks possible
    uint32_t max_checks = 0;
    uint32_t num_elements_in_last_geometry =
        mesh_data_.element_counts[num_geometries_ - 1];
    for (size_t i = 0; i < num_geometries_ - 1; ++i) {
      const uint32_t num_elements_in_geometry = mesh_data_.element_counts[i];
      const uint32_t num_elements_in_rest_of_geometries =
          (mesh_data_.element_offsets[num_geometries_ - 1] +
           num_elements_in_last_geometry) -
          mesh_data_.element_offsets[i + 1];
      const uint32_t total_checks_per_geometry =
          num_elements_in_rest_of_geometries * num_elements_in_geometry;
      max_checks += total_checks_per_geometry;
    }

    // We estimate 1% of the total checks remaining after broad phase
    estimated_narrow_phase_checks_ = std::max(1u, max_checks / 100);
    current_polygon_areas_size_ = estimated_narrow_phase_checks_;
    // Used to identify which of the broad phase checks actually resulted in
    // polygons
    SyclMemoryHelper::AllocateNarrowPhaseChecksCollisionMemory(
        mem_mgr_, collision_data_, estimated_narrow_phase_checks_);
    // We estimate 1% of the narrow phase checks remaining as actual polygons
    estimated_polygons_ = std::max(1u, estimated_narrow_phase_checks_ / 100);
    current_polygon_indices_size_ = estimated_polygons_;

    SyclMemoryHelper::AllocateFullPolygonMemory(mem_mgr_, polygon_data_,
                                                estimated_narrow_phase_checks_);

    SyclMemoryHelper::AllocateCompactPolygonMemory(mem_mgr_, polygon_data_,
                                                   estimated_polygons_);

    // Wait for all transfers to complete before returning
    sycl::event::wait_and_throw(transfer_events);
  }

  // Copy constructor
  Impl(const Impl& other) : q_device_(other.q_device_), mem_mgr_(q_device_) {
    // TODO(huzaifa): Implement deep copy of SYCL resources
    // For now, we'll just create a shallow copy which isn't ideal
    collision_candidates_ = other.collision_candidates_;
    num_geometries_ = other.num_geometries_;
  }

  // Copy assignment operator
  Impl& operator=(const Impl& other) {
    if (this != &other) {
      // TODO(huzaifa): Implement deep copy of SYCL resources
      // For now, we'll just create a shallow copy which isn't ideal
      q_device_ = other.q_device_;
      collision_candidates_ = other.collision_candidates_;
      num_geometries_ = other.num_geometries_;
    }
    return *this;
  }

  // Destructor
  ~Impl() {
    // Free device memory
    if (num_geometries_ > 0) {
      SyclMemoryHelper::FreeMeshMemory(mem_mgr_, mesh_data_);
      SyclMemoryHelper::FreeCollisionMemory(mem_mgr_, collision_data_);
      SyclMemoryHelper::FreeFullPolygonMemory(mem_mgr_, polygon_data_);
      SyclMemoryHelper::FreeCompactPolygonMemory(mem_mgr_, polygon_data_);
      SyclMemoryHelper::FreeBVHSingleMeshAndAllMeshMemory(mem_mgr_, bvh_data_);
    }
    SyclMemoryHelper::FreeDeviceCollidingIndicesMemoryChunk(mem_mgr_,
                                                            pair_chunk_);
    SyclMemoryHelper::FreeDeviceCollisionCountersMemoryChunk(mem_mgr_,
                                                             counters_chunk_);
    SyclMemoryHelper::FreeDeviceCollisionCountersOffsetsMemoryChunk(
        mem_mgr_, counters_offsets_chunk_);
    if (mesh_pair_ids_.meshAs) mem_mgr_.Free(mesh_pair_ids_.meshAs);
    if (mesh_pair_ids_.meshBs) mem_mgr_.Free(mesh_pair_ids_.meshBs);
  }

  // Check if SYCL is available
  static bool is_available() {
    try {
      // Attempt to construct a default queue. If a SYCL runtime is available
      // and a default device can be selected, this construction will succeed.
      sycl::queue q;
      // Suppress unused variable warning
      (void)q;
      return true;
    } catch (const sycl::exception& /* e */) {
      // An exception during queue construction indicates that a default SYCL
      // device/queue is not available.
      return false;
    } catch (...) {
      // Catch any other unexpected exceptions.
      return false;
    }
  }

  // Update collision candidates
  void UpdateCollisionCandidates(
      const std::vector<SortedPair<GeometryId>>& collision_candidates) {
    collision_candidates_.clear();
    // Get a vector of sorted ids that we need to check collision for
    for (const auto& pair : collision_candidates) {
      auto itA = std::lower_bound(sorted_ids_.begin(), sorted_ids_.end(),
                                  pair.first());
      auto itB = std::lower_bound(sorted_ids_.begin(), sorted_ids_.end(),
                                  pair.second());
      // Demand that the found ID is valid and what is being looked for
      DRAKE_DEMAND(itA != sorted_ids_.end() && *itA == pair.first());
      DRAKE_DEMAND(itB != sorted_ids_.end() && *itB == pair.second());
      uint32_t indexA =
          static_cast<uint32_t>(std::distance(sorted_ids_.begin(), itA));
      uint32_t indexB =
          static_cast<uint32_t>(std::distance(sorted_ids_.begin(), itB));

      // Can't have same ID
      DRAKE_DEMAND(indexA != indexB);
      double workA = mesh_data_.element_counts[indexA] *
                     std::log(mesh_data_.element_counts[indexB]);
      double workB = mesh_data_.element_counts[indexB] *
                     std::log(mesh_data_.element_counts[indexA]);
      if (std::abs(workA - workB) < 1e-6) {
        if (indexA < indexB) {
          collision_candidates_.push_back(std::make_pair(indexA, indexB));
        } else {
          collision_candidates_.push_back(std::make_pair(indexB, indexA));
        }
      } else if (workA < workB) {
        collision_candidates_.push_back(std::make_pair(indexA, indexB));
      } else {
        collision_candidates_.push_back(std::make_pair(indexB, indexA));
      }
    }

    // Sort collision_candidates in lexicographical order
    std::sort(collision_candidates_.begin(), collision_candidates_.end(),
              [](const std::pair<uint32_t, uint32_t>& a,
                 const std::pair<uint32_t, uint32_t>& b) {
                return a.first < b.first;
              });

    num_mesh_collisions_ = collision_candidates_.size();

    // Allocates memory for the chunk of memory that holds all the collision
    // indices (global) and collision counters for each MeshA in the
    // collision_candidates_
    SyclMemoryHelper::AllocateDeviceCollidingIndicesMemoryChunk(
        mem_mgr_, pair_chunk_, counters_chunk_, collision_candidates_to_data_,
        collision_candidates_, mesh_data_, mesh_pair_ids_);
    // Set the offsets for the collision counters
    uint32_t running_offset = 0;
    counters_offsets_chunk_.size_ = 0;
    std::vector<uint32_t> offsets_scan(num_mesh_collisions_);
    for (uint32_t i = 0; i < num_mesh_collisions_; i++) {
      uint32_t mesh_a = mesh_pair_ids_.meshAs[i];
      offsets_scan[i] = running_offset;
      running_offset += mesh_data_.element_counts[mesh_a];
      counters_offsets_chunk_.size_++;
    }
    q_device_
        .memcpy(counters_offsets_chunk_.mesh_a_offsets, offsets_scan.data(),
                num_mesh_collisions_ * sizeof(uint32_t))
        .wait();
  }

  // Compute hydroelastic surfaces
  std::vector<SYCLHydroelasticSurface> ComputeSYCLHydroelasticSurface(
      const std::unordered_map<GeometryId, math::RigidTransform<double>>&
          X_WGs) {
    if (num_geometries_ < 2) {
      return {};
    }

    // Build the BVH for each mesh with the untransformed

#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.StartKernel("unpack_transforms");
#endif

    // Get transfomers in host
    for (uint32_t geom_index = 0; geom_index < num_geometries_; ++geom_index) {
      GeometryId geometry_id = mesh_data_.geometry_ids[geom_index];
      const auto& X_WG = X_WGs.at(geometry_id);
      const auto& transform = X_WG.GetAsMatrix34();
#pragma unroll
      for (uint32_t i = 0; i < 12; ++i) {
        uint32_t row = i / 4;
        uint32_t col = i % 4;
        // Store transforms in row major order
        // transforms = [R_00, R_01, R_02, p_0, R_10, R_11, R_12, p_1, ...]
        mesh_data_.transforms[geom_index * 12 + i] = transform(row, col);
      }
    }
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("unpack_transforms");
    timing_logger_.StartKernel("transform_and_broad_phase");
#endif
    // ========================================
    // Command group 1: Transform quantities to world frame
    // ========================================

    // Combine all transformation kernels into a single command group
    auto transform_vertices_event = q_device_.submit([&](sycl::handler& h) {
      // Transform vertices
      const uint32_t work_group_size = 64;
      const uint32_t global_vertices =
          RoundUpToWorkGroupSize(total_vertices_, work_group_size);
      h.parallel_for<TransformVerticesKernel>(
          sycl::nd_range<1>(sycl::range<1>(global_vertices),
                            sycl::range<1>(work_group_size)),
          [=, vertices_M = mesh_data_.vertices_M,
           vertices_W = mesh_data_.vertices_W,
           vertex_mesh_ids = mesh_data_.vertex_mesh_ids,
           transforms = mesh_data_.transforms,
           total_vertices_ = total_vertices_] [[intel::kernel_args_restrict]]
#ifdef __NVPTX__
          [[sycl::reqd_work_group_size(64)]]
#endif
          (sycl::nd_item<1> item) {
            const uint32_t vertex_index = item.get_global_id(0);
            if (vertex_index >= total_vertices_) return;

            const uint32_t mesh_index = vertex_mesh_ids[vertex_index];

            const double x = vertices_M[vertex_index][0];
            const double y = vertices_M[vertex_index][1];
            const double z = vertices_M[vertex_index][2];
            double T[12];
#pragma unroll
            for (uint32_t i = 0; i < 12; ++i) {
              T[i] = transforms[mesh_index * 12 + i];
            }
            double new_x = T[0] * x + T[1] * y + T[2] * z + T[3];
            double new_y = T[4] * x + T[5] * y + T[6] * z + T[7];
            double new_z = T[8] * x + T[9] * y + T[10] * z + T[11];

            vertices_W[vertex_index][0] = new_x;
            vertices_W[vertex_index][1] = new_y;
            vertices_W[vertex_index][2] = new_z;
          });
    });

    // Transform inward normals
    auto transform_elem_quantities_event1 =
        q_device_.submit([&](sycl::handler& h) {
          const uint32_t work_group_size = 256;
          const uint32_t global_elements =
              RoundUpToWorkGroupSize(total_elements_, work_group_size);
          h.parallel_for<TransformInwardNormalsAndPressureKernel>(
              sycl::nd_range<1>(sycl::range<1>(global_elements),
                                sycl::range<1>(work_group_size)),
              [=, inward_normals_M = mesh_data_.inward_normals_M,
               inward_normals_W = mesh_data_.inward_normals_W,
               element_mesh_ids = mesh_data_.element_mesh_ids,
               transforms = mesh_data_.transforms,
               gradient_M_pressure_at_Mo = mesh_data_.gradient_M_pressure_at_Mo,
               gradient_W_pressure_at_Wo = mesh_data_.gradient_W_pressure_at_Wo,
               total_elements_ =
                   total_elements_] [[intel::kernel_args_restrict]]
#ifdef __NVPTX__
              [[sycl::reqd_work_group_size(256)]]
#endif
              (sycl::nd_item<1> item) {
                const uint32_t element_index = item.get_global_id(0);
                if (element_index >= total_elements_) return;

                const uint32_t mesh_index = element_mesh_ids[element_index];

                double T[12];
#pragma unroll
                for (uint32_t i = 0; i < 12; ++i) {
                  T[i] = transforms[mesh_index * 12 + i];
                }

                // Each element has 4 inward normals
                for (uint32_t j = 0; j < 4; ++j) {
                  const double nx = inward_normals_M[element_index][j][0];
                  const double ny = inward_normals_M[element_index][j][1];
                  const double nz = inward_normals_M[element_index][j][2];

                  // Only rotation
                  inward_normals_W[element_index][j][0] =
                      T[0] * nx + T[1] * ny + T[2] * nz;
                  inward_normals_W[element_index][j][1] =
                      T[4] * nx + T[5] * ny + T[6] * nz;
                  inward_normals_W[element_index][j][2] =
                      T[8] * nx + T[9] * ny + T[10] * nz;
                }

                // Each element has 1 pressure gradient
                const double gp_mx =
                    gradient_M_pressure_at_Mo[element_index][0];
                const double gp_my =
                    gradient_M_pressure_at_Mo[element_index][1];
                const double gp_mz =
                    gradient_M_pressure_at_Mo[element_index][2];
                const double p_mo = gradient_M_pressure_at_Mo[element_index][3];

                // Only rotation for the gradient pressures
                const double gp_wx = T[0] * gp_mx + T[1] * gp_my + T[2] * gp_mz;
                const double gp_wy = T[4] * gp_mx + T[5] * gp_my + T[6] * gp_mz;
                const double gp_wz =
                    T[8] * gp_mx + T[9] * gp_my + T[10] * gp_mz;

                const double p_wo =
                    p_mo - (gp_wx * T[3] + gp_wy * T[7] + gp_wz * T[11]);
                gradient_W_pressure_at_Wo[element_index][0] = gp_wx;
                gradient_W_pressure_at_Wo[element_index][1] = gp_wy;
                gradient_W_pressure_at_Wo[element_index][2] = gp_wz;
                gradient_W_pressure_at_Wo[element_index][3] = p_wo;
              });
        });

    // Compute AABBs of all the elements in all the meshes
    auto element_aabb_event = q_device_.submit([&](sycl::handler& h) {
      h.depends_on(transform_vertices_event);
      const uint32_t work_group_size = 256;
      const uint32_t global_elements =
          RoundUpToWorkGroupSize(total_elements_, work_group_size);

      h.parallel_for<ComputeElementAABBKernel>(
          sycl::nd_range<1>(sycl::range<1>(global_elements),
                            sycl::range<1>(work_group_size)),
          [=, elements = mesh_data_.elements,
           vertices_W = mesh_data_.vertices_W,
           element_mesh_ids = mesh_data_.element_mesh_ids,
           element_aabb_min_W = mesh_data_.element_aabb_min_W,
           element_aabb_max_W = mesh_data_.element_aabb_max_W,
           vertex_offsets = mesh_data_.vertex_offsets,
           total_elements_ = total_elements_] [[intel::kernel_args_restrict]]

#ifdef __NVPTX__
          [[sycl::reqd_work_group_size(256)]]
#endif
          (sycl::nd_item<1> item) {
            const uint32_t element_index = item.get_global_id(0);
            if (element_index >= total_elements_) return;

            const uint32_t geom_index = element_mesh_ids[element_index];
            // Get the four vertex indices for this tetrahedron
            const std::array<int, 4>& tet_vertices = elements[element_index];
            const uint32_t vertex_mesh_offset = vertex_offsets[geom_index];

            // Initialize min/max to first vertex
            double min_x = vertices_W[vertex_mesh_offset + tet_vertices[0]][0];
            double min_y = vertices_W[vertex_mesh_offset + tet_vertices[0]][1];
            double min_z = vertices_W[vertex_mesh_offset + tet_vertices[0]][2];

            double max_x = min_x;
            double max_y = min_y;
            double max_z = min_z;

            // Find min/max across all four vertices
            for (int i = 1; i < 4; ++i) {
              const uint32_t vertex_idx = vertex_mesh_offset + tet_vertices[i];

              // Update min coordinates
              min_x = sycl::min(min_x, vertices_W[vertex_idx][0]);
              min_y = sycl::min(min_y, vertices_W[vertex_idx][1]);
              min_z = sycl::min(min_z, vertices_W[vertex_idx][2]);

              // Update max coordinates
              max_x = sycl::max(max_x, vertices_W[vertex_idx][0]);
              max_y = sycl::max(max_y, vertices_W[vertex_idx][1]);
              max_z = sycl::max(max_z, vertices_W[vertex_idx][2]);
            }

            // Store the results
            element_aabb_min_W[element_index][0] = min_x;
            element_aabb_min_W[element_index][1] = min_y;
            element_aabb_min_W[element_index][2] = min_z;

            element_aabb_max_W[element_index][0] = max_x;
            element_aabb_max_W[element_index][1] = max_y;
            element_aabb_max_W[element_index][2] = max_z;
          });
    });
    if (!bvh_broad_phase_.IsBVHBuilt()) {
      // Build BVH if not built
      bvh_broad_phase_.build(mesh_data_, sorted_total_lower_,
                             sorted_total_upper_, bvh_data_, element_aabb_event,
                             mem_mgr_, q_device_);
      // Building refits the BVH
      bvh_broad_phase_.SetBVHRefitted(true);
    }

    // Blocking event call
    bvh_broad_phase_.BroadPhase(
        mesh_data_, sorted_total_lower_, sorted_total_upper_, bvh_data_,
        element_aabb_event, collision_candidates_to_data_, num_mesh_collisions_,
        pair_chunk_, counters_chunk_, counters_offsets_chunk_, mesh_pair_ids_,
        mem_mgr_, q_device_);

    // Chunk size gives us all the element checks that we need to perform
    total_narrow_phase_checks_ = pair_chunk_.size_;
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("transform_and_broad_phase");
#endif
    if (total_narrow_phase_checks_ == 0) {
      return {};
    }
    drake::common::ProblemSizeLogger::GetInstance().AddCount(
        "SYCLCandidateTets", total_narrow_phase_checks_);

    if (total_narrow_phase_checks_ > current_polygon_areas_size_) {
      // Give a 10 % bigger size
      uint32_t new_size =
          static_cast<uint32_t>(1.1 * total_narrow_phase_checks_);

      // Free old memory
      SyclMemoryHelper::FreeFullPolygonMemory(mem_mgr_, polygon_data_);
      SyclMemoryHelper::FreeNarrowPhaseChecksCollisionMemory(mem_mgr_,
                                                             collision_data_);
      // Allocate new memory with larger size
      SyclMemoryHelper::AllocateFullPolygonMemory(mem_mgr_, polygon_data_,
                                                  new_size);
      SyclMemoryHelper::AllocateNarrowPhaseChecksCollisionMemory(
          mem_mgr_, collision_data_, new_size);

      current_polygon_areas_size_ = new_size;
    }
    // Initialize all the narrow phase check validity to 1
    q_device_
        .fill(collision_data_.narrow_phase_check_validity,
              static_cast<uint8_t>(1), current_polygon_areas_size_)
        .wait();

    // Create dependency vector
    std::vector<sycl::event> dependencies = {transform_elem_quantities_event1};
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.StartKernel("compute_contact_polygons");
#endif
    sycl::event compute_contact_polygon_event;
    if (q_device_.get_device().get_info<sycl::info::device::device_type>() ==
        sycl::info::device_type::gpu) {
      compute_contact_polygon_event =
          LaunchContactPolygonComputation<DeviceCollidingIndicesMemoryChunk,
                                          DeviceCollisionData, DeviceMeshData,
                                          DevicePolygonData, DeviceType::GPU>(
              q_device_, dependencies, total_narrow_phase_checks_, pair_chunk_,
              collision_data_, mesh_data_, polygon_data_);
    } else {
      compute_contact_polygon_event =
          LaunchContactPolygonComputation<DeviceCollidingIndicesMemoryChunk,
                                          DeviceCollisionData, DeviceMeshData,
                                          DevicePolygonData, DeviceType::CPU>(
              q_device_, dependencies, total_narrow_phase_checks_, pair_chunk_,
              collision_data_, mesh_data_, polygon_data_);
    }
    compute_contact_polygon_event.wait_and_throw();

    auto policy = oneapi::dpl::execution::make_device_policy(q_device_);
    // Exclusive scan to compact data into only the valid polygons found by
    // SYCL
    oneapi::dpl::transform_exclusive_scan(
        policy, collision_data_.narrow_phase_check_validity,
        collision_data_.narrow_phase_check_validity +
            total_narrow_phase_checks_,
        collision_data_.prefix_sum_narrow_phase_checks,  // output
        static_cast<uint32_t>(0),                        // initial value
        sycl::plus<uint32_t>(),                          // binary operation
        [](uint8_t x) {
          return static_cast<uint32_t>(x);
        });  // transform uint8_t to uint32_t
    q_device_.wait_and_throw();

    total_polygons_ = 0;
    q_device_
        .memcpy(&total_polygons_,
                collision_data_.prefix_sum_narrow_phase_checks +
                    total_narrow_phase_checks_ - 1,
                sizeof(uint32_t))
        .wait();
    // Last element check or not?
    uint8_t last_check_flag = 0;
    q_device_
        .memcpy(&last_check_flag,
                collision_data_.narrow_phase_check_validity +
                    total_narrow_phase_checks_ - 1,
                sizeof(uint8_t))
        .wait();
    // If last check is 1, then we need to add one more check
    total_polygons_ += static_cast<uint32_t>(last_check_flag);
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("compute_contact_polygons");
#endif
    if (total_polygons_ == 0) {
      return {};
    }
    drake::common::ProblemSizeLogger::GetInstance().AddCount("SYCFacesInserted",
                                                             total_polygons_);
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.StartKernel("compact_polygon_data");
#endif
    if (total_polygons_ > current_polygon_indices_size_) {
      // Give a 10 % bigger size
      uint32_t new_size = static_cast<uint32_t>(1.1 * total_polygons_);

      SyclMemoryHelper::FreeCompactPolygonMemory(mem_mgr_, polygon_data_);

      // Allocate new memory with larger size
      SyclMemoryHelper::AllocateCompactPolygonMemory(mem_mgr_, polygon_data_,
                                                     new_size);
      current_polygon_indices_size_ = new_size;
    }

    auto memset_event =
        q_device_.memset(polygon_data_.valid_polygon_indices, 0,
                         current_polygon_indices_size_ * sizeof(uint32_t));
    memset_event.wait_and_throw();
    auto fill_valid_polygon_indicesevent =
        q_device_.submit([&](sycl::handler& h) {
          h.depends_on(compute_contact_polygon_event);
          h.parallel_for<FillValidPolygonIndicesKernel>(
              sycl::range<1>(total_narrow_phase_checks_),
              [=, valid_polygon_indices = polygon_data_.valid_polygon_indices,
               prefix_sum_narrow_phase_checks =
                   collision_data_.prefix_sum_narrow_phase_checks,
               narrow_phase_check_validity =
                   collision_data_.narrow_phase_check_validity](
                  sycl::id<1> idx) {
                const uint32_t check_index = idx[0];
                if (narrow_phase_check_validity[check_index] == 1) {
                  uint32_t valid_polygon_index =
                      prefix_sum_narrow_phase_checks[check_index];
                  valid_polygon_indices[valid_polygon_index] = check_index;
                }
              });
        });
    fill_valid_polygon_indicesevent.wait_and_throw();

    // Compact all the data to data only with valid polygons
    auto compact_event = q_device_.submit([&](sycl::handler& h) {
      h.depends_on({fill_valid_polygon_indicesevent});
      h.parallel_for<CompactPolygonDataKernel>(
          sycl::range<1>(total_polygons_),
          [=, compacted_polygon_areas = polygon_data_.compacted_polygon_areas,
           compacted_polygon_centroids =
               polygon_data_.compacted_polygon_centroids,
           compacted_polygon_normals = polygon_data_.compacted_polygon_normals,
           compacted_polygon_g_M = polygon_data_.compacted_polygon_g_M,
           compacted_polygon_g_N = polygon_data_.compacted_polygon_g_N,
           compacted_polygon_pressure_W =
               polygon_data_.compacted_polygon_pressure_W,
           compacted_polygon_geom_index_A =
               polygon_data_.compacted_polygon_geom_index_A,
           compacted_polygon_geom_index_B =
               polygon_data_.compacted_polygon_geom_index_B,
           valid_polygon_indices = polygon_data_.valid_polygon_indices,
           polygon_areas = polygon_data_.polygon_areas,
           polygon_centroids = polygon_data_.polygon_centroids,
           polygon_normals = polygon_data_.polygon_normals,
           polygon_g_M = polygon_data_.polygon_g_M,
           polygon_g_N = polygon_data_.polygon_g_N,
           polygon_pressure_W = polygon_data_.polygon_pressure_W,
           polygon_geom_index_A = polygon_data_.polygon_geom_index_A,
           polygon_geom_index_B =
               polygon_data_.polygon_geom_index_B](sycl::id<1> idx) {
            const uint32_t valid_polygon_index = idx[0];
            const uint32_t check_index =
                valid_polygon_indices[valid_polygon_index];
            compacted_polygon_areas[valid_polygon_index] =
                polygon_areas[check_index];
            compacted_polygon_centroids[valid_polygon_index] =
                polygon_centroids[check_index];
            compacted_polygon_normals[valid_polygon_index] =
                polygon_normals[check_index];
            compacted_polygon_g_M[valid_polygon_index] =
                polygon_g_M[check_index];
            compacted_polygon_g_N[valid_polygon_index] =
                polygon_g_N[check_index];
            compacted_polygon_pressure_W[valid_polygon_index] =
                polygon_pressure_W[check_index];
            compacted_polygon_geom_index_A[valid_polygon_index] =
                polygon_geom_index_A[check_index];
            compacted_polygon_geom_index_B[valid_polygon_index] =
                polygon_geom_index_B[check_index];
          });
    });
    compact_event.wait_and_throw();

#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("compact_polygon_data");
#endif

    // For now return a vector
    return {CreateHydroelasticSurface(
        q_device_, polygon_data_.compacted_polygon_centroids,
        polygon_data_.compacted_polygon_areas,
        polygon_data_.compacted_polygon_pressure_W,
        polygon_data_.compacted_polygon_normals,
        polygon_data_.compacted_polygon_g_M,
        polygon_data_.compacted_polygon_g_N,
        polygon_data_.compacted_polygon_geom_index_A,
        polygon_data_.compacted_polygon_geom_index_B, total_polygons_)};
  }  // namespace sycl_impl

 private:
  // Helper method to initialize SYCL queue
  static sycl::queue InitializeQueue() {
    try {
#ifdef DRAKE_SYCL_TIMING_ENABLED
      //   sycl::queue q(sycl::gpu_selector_v,
      //                 sycl::property::queue::enable_profiling());
      sycl::queue q(sycl::gpu_selector_v);
#else
      sycl::queue q(sycl::gpu_selector_v);
#endif
      std::cout << "Using "
                << q.get_device().get_info<sycl::info::device::name>()
                << std::endl;
      return q;
    } catch (sycl::exception const& e) {
      std::cout << "Cannot select a GPU\n" << e.what() << std::endl;
      std::cout << "Using a CPU device" << std::endl;
#ifdef DRAKE_SYCL_TIMING_ENABLED
      //   sycl::queue q(sycl::cpu_selector_v,
      //                 sycl::property::queue::enable_profiling());
      sycl::queue q(sycl::cpu_selector_v);
#else
      sycl::queue q(sycl::cpu_selector_v);
#endif
      std::cout << "Using "
                << q.get_device().get_info<sycl::info::device::name>()
                << std::endl;
      return q;
    }
  }

  // We have a CPU queue for operations beneficial to perform on the host
  // and a device queue for operations beneficial to perform on the
  // Accelerator. Note: q_device_ HAS TO BE declared before mem_mgr_ since
  // it needs to be initialized first.
  sycl::queue q_device_;

  SyclMemoryManager mem_mgr_;
  DeviceMeshData mesh_data_;
  DeviceBVHData bvh_data_;
  DeviceCollisionData collision_data_;
  DevicePolygonData polygon_data_;
  BVHBroadPhase bvh_broad_phase_;

  // Timing logger for kernel performance analysis
  SyclTimingLogger timing_logger_;

  // The collision candidates.
  std::vector<std::pair<uint32_t, uint32_t>> collision_candidates_;
  std::vector<GeometryId> sorted_ids_;
  std::unordered_map<uint64_t, std::pair<DeviceMeshACollisionCounters,
                                         DeviceMeshPairCollidingIndices>>
      collision_candidates_to_data_;
  DeviceCollidingIndicesMemoryChunk pair_chunk_;
  DeviceCollisionCountersMemoryChunk counters_chunk_;
  DeviceCollisionCountersOffsetsMemoryChunk counters_offsets_chunk_;
  DeviceMeshPairIds mesh_pair_ids_;

  std::vector<Vector3<double>> sorted_total_lower_;
  std::vector<Vector3<double>> sorted_total_upper_;

  // Number of geometries
  uint32_t num_geometries_ = 0;

  uint32_t num_mesh_collisions_ = 0;

  uint32_t total_vertices_ = 0;
  uint32_t total_elements_ = 0;
  uint32_t total_nodes_ = 0;

  uint32_t current_polygon_areas_size_ =
      0;  // Current size of polygon_areas to prevent constant reallocation

  uint32_t current_polygon_indices_size_ =
      0;  // Current size of valid_polygon_indices to prevent constant
          // reallocation

  uint32_t current_debug_polygon_vertices_size_ = 0;

  uint32_t estimated_narrow_phase_checks_ =
      0;  // Estimated number of narrow phase checks (set to be 5% of total
          // element checks and used to size polygon_areas and
          // polygon_centroids)
  uint32_t total_narrow_phase_checks_ =
      0;  // Total number of narrow phase checks in the current time step
          // (updated in ComputeSYCLHydroelasticSurface)

  uint32_t total_polygons_ = 0;  // Total number of valid polygons found by SYCL
  uint32_t estimated_polygons_ = 0;  // Estimated number of polygons (set to
                                     // be 1% of the narrow phase checks)

  friend class SyclProximityEngineAttorney;
};  // namespace sycl_impl

bool SyclProximityEngine::is_available() {
  return Impl::is_available();
}

SyclProximityEngine::SyclProximityEngine(
    const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
        soft_geometries,
    const std::unordered_map<GeometryId, Vector3<double>>& total_lower,
    const std::unordered_map<GeometryId, Vector3<double>>& total_upper)
    : impl_(std::make_unique<Impl>(soft_geometries, total_lower, total_upper)) {
}

SyclProximityEngine::SyclProximityEngine() : impl_(std::make_unique<Impl>()) {}

SyclProximityEngine::~SyclProximityEngine() = default;

SyclProximityEngine::SyclProximityEngine(const SyclProximityEngine& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

SyclProximityEngine& SyclProximityEngine::operator=(
    const SyclProximityEngine& other) {
  if (this != &other) {
    *impl_ = *other.impl_;
  }
  return *this;
}

void SyclProximityEngine::UpdateCollisionCandidates(
    const std::vector<SortedPair<GeometryId>>& collision_candidates) {
  impl_->UpdateCollisionCandidates(collision_candidates);
}

std::vector<SYCLHydroelasticSurface>
SyclProximityEngine::ComputeSYCLHydroelasticSurface(
    const std::unordered_map<GeometryId, math::RigidTransform<double>>& X_WGs) {
  return impl_->ComputeSYCLHydroelasticSurface(X_WGs);
}

void SyclProximityEngine::PrintTimingStats() const {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  SyclProximityEngineAttorney::PrintTimingStats(impl_.get());
#endif
}

void SyclProximityEngine::PrintTimingStatsJson(const std::string& path) const {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  SyclProximityEngineAttorney::PrintTimingStatsJson(impl_.get(), path);
#endif
}

// SyclProximityEngineAttorney class definition
SyclProximityEngine::Impl* SyclProximityEngineAttorney::get_impl(
    SyclProximityEngine& engine) {
  return engine.impl_.get();
}
const SyclProximityEngine::Impl* SyclProximityEngineAttorney::get_impl(
    const SyclProximityEngine& engine) {
  return engine.impl_.get();
}

uint32_t SyclProximityEngineAttorney::get_total_checks(
    SyclProximityEngine::Impl* impl) {
  return 0;
}

std::vector<uint8_t> SyclProximityEngineAttorney::get_collision_filter(
    SyclProximityEngine::Impl* impl) {
  uint32_t total_checks = SyclProximityEngineAttorney::get_total_checks(impl);
  std::vector<uint8_t> collision_filterhost(total_checks);
  auto q = impl->q_device_;
  auto collision_filter = impl->collision_data_.collision_filter;
  q.memcpy(collision_filterhost.data(), collision_filter,
           total_checks * sizeof(uint8_t))
      .wait();
  return collision_filterhost;
}

std::vector<uint32_t> SyclProximityEngineAttorney::get_prefix_sum(
    SyclProximityEngine::Impl* impl) {
  uint32_t total_checks = SyclProximityEngineAttorney::get_total_checks(impl);
  std::vector<uint32_t> prefix_sum_total_checkshost(total_checks);
  auto q = impl->q_device_;
  auto prefix_sum = impl->collision_data_.prefix_sum_total_checks;
  q.memcpy(prefix_sum_total_checkshost.data(), prefix_sum,
           total_checks * sizeof(uint32_t))
      .wait();
  return prefix_sum_total_checkshost;
}

std::vector<Vector3<double>> SyclProximityEngineAttorney::get_vertices_M(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto vertices_M = impl->mesh_data_.vertices_M;
  auto total_vertices = impl->total_vertices_;
  std::vector<Vector3<double>> vertices_Mhost(total_vertices);
  q.memcpy(vertices_Mhost.data(), vertices_M,
           total_vertices * sizeof(Vector3<double>))
      .wait();
  return vertices_Mhost;
}
std::vector<Vector3<double>> SyclProximityEngineAttorney::get_vertices_W(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto vertices_W = impl->mesh_data_.vertices_W;
  auto total_vertices = impl->total_vertices_;
  std::vector<Vector3<double>> vertices_Whost(total_vertices);
  q.memcpy(vertices_Whost.data(), vertices_W,
           total_vertices * sizeof(Vector3<double>))
      .wait();
  return vertices_Whost;
}
std::vector<std::array<int, 4>> SyclProximityEngineAttorney::get_elements(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto elements = impl->mesh_data_.elements;
  auto total_elements_ = impl->total_elements_;
  std::vector<std::array<int, 4>> elementshost(total_elements_);
  q.memcpy(elementshost.data(), elements,
           total_elements_ * sizeof(std::array<int, 4>))
      .wait();
  return elementshost;
}
double* SyclProximityEngineAttorney::get_pressures(
    SyclProximityEngine::Impl* impl) {
  return impl->mesh_data_.pressures;
}
Vector4<double>* SyclProximityEngineAttorney::get_gradient_M_pressure_at_Mo(
    SyclProximityEngine::Impl* impl) {
  return impl->mesh_data_.gradient_M_pressure_at_Mo;
}
Vector4<double>* SyclProximityEngineAttorney::get_gradient_W_pressure_at_Wo(
    SyclProximityEngine::Impl* impl) {
  return impl->mesh_data_.gradient_W_pressure_at_Wo;
}
uint32_t* SyclProximityEngineAttorney::get_collision_filter_host_body_index(
    SyclProximityEngine::Impl* impl) {
  return impl->collision_data_.collision_filter_host_body_index;
}

uint32_t SyclProximityEngineAttorney::get_total_narrow_phase_checks(
    SyclProximityEngine::Impl* impl) {
  return impl->total_narrow_phase_checks_;
}

uint32_t SyclProximityEngineAttorney::get_total_polygons(
    SyclProximityEngine::Impl* impl) {
  return impl->total_polygons_;
}

std::vector<uint32_t>
SyclProximityEngineAttorney::get_narrow_phase_check_indices(
    SyclProximityEngine::Impl* impl) {
  uint32_t total_narrow_phase_checks =
      SyclProximityEngineAttorney::get_total_narrow_phase_checks(impl);
  std::vector<uint32_t> narrow_phase_check_indiceshost(
      total_narrow_phase_checks);
  auto q = impl->q_device_;
  auto narrow_phase_check_indices =
      impl->collision_data_.narrow_phase_check_indices;
  q.memcpy(narrow_phase_check_indiceshost.data(), narrow_phase_check_indices,
           total_narrow_phase_checks * sizeof(uint32_t))
      .wait();
  return narrow_phase_check_indiceshost;
}

std::vector<uint32_t> SyclProximityEngineAttorney::get_valid_polygon_indices(
    SyclProximityEngine::Impl* impl) {
  uint32_t total_polygons =
      SyclProximityEngineAttorney::get_total_polygons(impl);
  std::vector<uint32_t> valid_polygon_indiceshost(total_polygons);
  auto q = impl->q_device_;
  auto valid_polygon_indices = impl->polygon_data_.valid_polygon_indices;
  q.memcpy(valid_polygon_indiceshost.data(), valid_polygon_indices,
           total_polygons * sizeof(uint32_t))
      .wait();
  return valid_polygon_indiceshost;
}

std::vector<double> SyclProximityEngineAttorney::get_polygon_areas(
    SyclProximityEngine::Impl* impl) {
  uint32_t total_narrow_phase_checks =
      SyclProximityEngineAttorney::get_total_narrow_phase_checks(impl);
  std::vector<double> polygon_areashost(total_narrow_phase_checks);
  auto q = impl->q_device_;
  auto polygon_areas = impl->polygon_data_.polygon_areas;
  q.memcpy(polygon_areashost.data(), polygon_areas,
           total_narrow_phase_checks * sizeof(double))
      .wait();
  return polygon_areashost;
}

std::vector<Vector3<double>> SyclProximityEngineAttorney::get_polygon_centroids(
    SyclProximityEngine::Impl* impl) {
  uint32_t total_narrow_phase_checks =
      SyclProximityEngineAttorney::get_total_narrow_phase_checks(impl);
  std::vector<Vector3<double>> polygon_centroidshost(total_narrow_phase_checks);
  auto q = impl->q_device_;
  auto polygon_centroids = impl->polygon_data_.polygon_centroids;
  q.memcpy(polygon_centroidshost.data(), polygon_centroids,
           total_narrow_phase_checks * sizeof(Vector3<double>))
      .wait();
  return polygon_centroidshost;
}

std::vector<double> SyclProximityEngineAttorney::get_debug_polygon_vertices(
    SyclProximityEngine::Impl* impl) {
  std::vector<double> debug_polygon_vertices_host(
      impl->current_debug_polygon_vertices_size_);
  auto q = impl->q_device_;
  auto debug_polygon_vertices = impl->polygon_data_.debug_polygon_vertices;
  q.memcpy(debug_polygon_vertices_host.data(), debug_polygon_vertices,
           impl->current_debug_polygon_vertices_size_ * sizeof(double))
      .wait();
  return debug_polygon_vertices_host;
}

std::unordered_map<
    SortedPair<GeometryId>,
    std::pair<HostMeshACollisionCounters, HostMeshPairCollidingIndices>>
SyclProximityEngineAttorney::get_collision_candidates_to_data(
    SyclProximityEngine::Impl* impl) {
  // Create new map from the impl's map and copy over data to host
  std::unordered_map<
      SortedPair<GeometryId>,
      std::pair<HostMeshACollisionCounters, HostMeshPairCollidingIndices>>
      host_collision_candidates_to_data;
  auto q = impl->q_device_;

  // Copy over chunked collision counts.
  // This array is scanned for all the mesh collision candidates.
  std::vector<uint32_t> collision_counts_chunk_host(
      impl->counters_chunk_.size_);
  q.memcpy(collision_counts_chunk_host.data(),
           impl->counters_chunk_.collision_counts,
           impl->counters_chunk_.size_ * sizeof(uint32_t))
      .wait();
  // For the tests we need to figure out how many collisions each mesh pair has.
  // We will thus use the scanned array to extract the total collisions for each
  // pair.
  uint32_t collisions_counted = 0;
  uint32_t size_passed = 0;
  for (int i = 0; i < impl->num_mesh_collisions_; i++) {
    uint32_t mesh_a = impl->mesh_pair_ids_.meshAs[i];
    uint32_t mesh_b = impl->mesh_pair_ids_.meshBs[i];
    uint64_t col_key = key(mesh_a, mesh_b);
    auto& [cc, ci] = impl->collision_candidates_to_data_[col_key];
    // If we are in the last mesh collision pair, then direcly used the complete
    // total collisions of the scene.
    size_passed += cc.size_;
    if (i == impl->num_mesh_collisions_ - 1) {
      cc.total_collisions =
          impl->counters_chunk_.total_collisions - collisions_counted;
    } else {
      cc.total_collisions =
          collision_counts_chunk_host[size_passed] - collisions_counted;
    }
    collisions_counted += cc.total_collisions;
  }
  // With the total collisions set for each mesh pair set, we can find the right
  // pointers to each pairs collision indices
  uint32_t running_offset = 0;
  for (int i = 0; i < impl->num_mesh_collisions_; i++) {
    uint32_t mesh_a = impl->mesh_pair_ids_.meshAs[i];
    uint32_t mesh_b = impl->mesh_pair_ids_.meshBs[i];
    uint64_t col_key = key(mesh_a, mesh_b);
    auto& [cc, ci] = impl->collision_candidates_to_data_[col_key];
    ci.size_ = cc.total_collisions;
    ci.collision_indices_A =
        impl->pair_chunk_.collision_indices_A + running_offset;
    ci.collision_indices_B =
        impl->pair_chunk_.collision_indices_B + running_offset;
    running_offset += cc.total_collisions;
  }

  for (auto& [key, value] : impl->collision_candidates_to_data_) {
    auto& [cc, ci] = value;
    HostMeshACollisionCounters host_cc;
    // host_cc.last_element_collision_count = cc.last_element_collision_count;
    host_cc.total_collisions = cc.total_collisions;
    host_cc.collision_counts.resize(cc.size_);

    q.memcpy(host_cc.collision_counts.data(), cc.collision_counts,
             cc.size_ * sizeof(uint32_t))
        .wait();
    HostMeshPairCollidingIndices host_ci;
    host_ci.collision_indices_A.resize(ci.size_);
    host_ci.collision_indices_B.resize(ci.size_);
    q.memcpy(host_ci.collision_indices_A.data(), ci.collision_indices_A,
             ci.size_ * sizeof(uint32_t))
        .wait();
    q.memcpy(host_ci.collision_indices_B.data(), ci.collision_indices_B,
             ci.size_ * sizeof(uint32_t))
        .wait();

    auto [mesh_a, mesh_b] = key_to_pair(key);
    GeometryId idA = impl->sorted_ids_[mesh_a];
    GeometryId idB = impl->sorted_ids_[mesh_b];
    SortedPair<GeometryId> sorted_pair(idA, idB);
    host_collision_candidates_to_data[sorted_pair] = {host_cc, host_ci};
  }
  return host_collision_candidates_to_data;
}

HostMeshData SyclProximityEngineAttorney::get_mesh_data(
    SyclProximityEngine::Impl* impl) {
  uint32_t total_meshes = impl->num_geometries_;
  uint32_t total_elements = impl->total_elements_;
  HostMeshData host_mesh_data;
  host_mesh_data.element_offsets.resize(total_meshes);
  host_mesh_data.element_aabb_min_W.resize(total_elements);
  host_mesh_data.element_aabb_max_W.resize(total_elements);
  auto q = impl->q_device_;
  q.memcpy(host_mesh_data.element_offsets.data(),
           impl->mesh_data_.element_offsets, total_meshes * sizeof(uint32_t))
      .wait();
  q.memcpy(host_mesh_data.element_aabb_min_W.data(),
           impl->mesh_data_.element_aabb_min_W,
           total_elements * sizeof(Vector3<double>))
      .wait();

  q.memcpy(host_mesh_data.element_aabb_max_W.data(),
           impl->mesh_data_.element_aabb_max_W,
           total_elements * sizeof(Vector3<double>))
      .wait();
  host_mesh_data.total_elements = total_elements;

  return host_mesh_data;
}

uint32_t SyclProximityEngineAttorney::get_num_meshes(
    SyclProximityEngine::Impl* impl) {
  return impl->num_geometries_;
}

void SyclProximityEngineAttorney::PrintTimingStats(
    SyclProximityEngine::Impl* impl) {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  impl->timing_logger_.PrintStats();
#endif
}

void SyclProximityEngineAttorney::PrintTimingStatsJson(
    SyclProximityEngine::Impl* impl, const std::string& path) {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  impl->timing_logger_.PrintStatsJson(path);
#endif
}

HostBVH SyclProximityEngineAttorney::get_host_bvh(
    SyclProximityEngine::Impl* impl, const int sorted_mesh_id) {
  if (sorted_mesh_id < 0 ||
      static_cast<uint32_t>(sorted_mesh_id) >= impl->bvh_data_.num_meshes) {
    throw std::runtime_error("Invalid mesh_id: " +
                             std::to_string(sorted_mesh_id));
  }

  const BVH& device_bvh = impl->bvh_data_.bvhAll[sorted_mesh_id];
  HostBVH host_bvh;
  host_bvh.max_nodes = device_bvh.max_nodes;

  // TODO - Set these post build in GPU BVH
  // host_bvh.num_nodes =
  //     device_bvh.num_nodes;  // Assuming this is set post-build.
  // host_bvh.num_leaf_nodes =
  //     device_bvh.num_leaf_nodes;  // Assuming set post-build.

  // Copy arrays to host vectors.
  host_bvh.node_lowers.resize(device_bvh.max_nodes);
  impl->mem_mgr_.CopyToHost(host_bvh.node_lowers.data(), device_bvh.node_lowers,
                            device_bvh.max_nodes);

  host_bvh.node_uppers.resize(device_bvh.max_nodes);
  impl->mem_mgr_.CopyToHost(host_bvh.node_uppers.data(), device_bvh.node_uppers,
                            device_bvh.max_nodes);

  host_bvh.node_parents.resize(device_bvh.max_nodes);
  impl->mem_mgr_.CopyToHost(host_bvh.node_parents.data(),
                            device_bvh.node_parents, device_bvh.max_nodes);

  // Copy root index (single int).
  int device_root;
  impl->mem_mgr_.CopyToHost(&device_root, device_bvh.root, 1);
  host_bvh.root_index = device_root;

  // Wait for all copies to complete.
  impl->q_device_.wait_and_throw();

  return host_bvh;
}

HostIndicesAll SyclProximityEngineAttorney::get_host_indices_all(
    SyclProximityEngine::Impl* impl) {
  HostIndicesAll host_indices_all;
  host_indices_all.indicesAll.resize(impl->total_elements_);
  impl->mem_mgr_.CopyToHost(host_indices_all.indicesAll.data(),
                            impl->bvh_data_.indicesAll, impl->total_elements_);
  impl->q_device_.wait_and_throw();
  return host_indices_all;
}

int SyclProximityEngineAttorney::ComputeBVHTreeHeight(const HostBVH& host_bvh) {
  std::vector<bool> visited(host_bvh.max_nodes, false);
  return ComputeSubtreeHeight(host_bvh, host_bvh.root_index, visited);
}

int SyclProximityEngineAttorney::CountBVHLeaves(const HostBVH& host_bvh) {
  int count = 0;
  std::vector<bool> visited(host_bvh.max_nodes, false);
  CountLeavesRecursive(host_bvh, host_bvh.root_index, visited, &count);
  return count;
}

int SyclProximityEngineAttorney::ComputeBVHBalanceFactor(
    const HostBVH& host_bvh) {
  int max_imbalance = 0;
  std::vector<bool> visited(host_bvh.max_nodes, false);
  ComputeBalanceRecursive(host_bvh, host_bvh.root_index, visited,
                          &max_imbalance);
  return max_imbalance;
}

// Computes average depth across all leaves
double SyclProximityEngineAttorney::ComputeBVHAverageLeafDepth(
    const HostBVH& host_bvh) {
  int total_depth = 0;
  int leaf_count = 0;
  std::vector<bool> visited(host_bvh.max_nodes, false);
  ComputeDepthsRecursive(host_bvh, host_bvh.root_index, visited, 0,
                         &total_depth, &leaf_count);
  return leaf_count > 0 ? static_cast<double>(total_depth) / leaf_count : 0.0;
}

bool SyclProximityEngineAttorney::VerifyBVHBounds(const HostBVH& host_bvh) {
  std::vector<bool> visited(host_bvh.max_nodes, false);
  return VerifyBoundsRecursive(host_bvh, host_bvh.root_index, visited);
}

void SyclProximityEngineAttorney::ComputeAndPrintBVHImbalanceHistogram(
    const HostBVH& host_bvh, const std::string& filepath) {
  auto heights = ComputeAllHeights(host_bvh);

  int max_diff = 0;
  std::vector<int> diffs;
  for (int i = 0; i < host_bvh.max_nodes; ++i) {
    if (heights[i] == -1) continue;
    const auto& lower = host_bvh.node_lowers[i];
    if (lower.b == 1) continue;  // Leaf
    int left_h = heights[lower.i];
    int right_h = heights[host_bvh.node_uppers[i].i];
    int diff = std::abs(left_h - right_h);
    diffs.push_back(diff);
    max_diff = std::max(max_diff, diff);
  }

  std::vector<int> histogram(max_diff + 1, 0);
  for (int d : diffs) {
    ++histogram[d];
  }
  PrintHistogramJSON(histogram, filepath);
}

void SyclProximityEngineAttorney::PrintBVHNodeBoundingBoxes(
    const HostBVH& host_bvh, const HostIndicesAll& host_indices_all,
    const HostMeshData& mesh_data, const int sorted_mesh_id) {
  PrintNodeBoundingBoxesBFS(host_bvh, host_bvh.root_index, host_indices_all,
                            mesh_data, sorted_mesh_id);
}

// Private helpers
// Helper to compute subtree height with cycle detection.
int SyclProximityEngineAttorney::ComputeSubtreeHeight(
    const HostBVH& host_bvh, int node_index, std::vector<bool>& visited) {
  if (node_index < 0 || node_index >= host_bvh.max_nodes ||
      visited[node_index]) {
    return -1;  // Invalid or cycle.
  }
  visited[node_index] = true;

  const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
  if (lower.b == 1) {  // Leaf (based on your BVH: b=1 for leaves).
    return 0;
  }

  // Assume binary tree: left child in lower.i, right in upper.i.
  const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
  int left_height = ComputeSubtreeHeight(host_bvh, lower.i, visited);
  int right_height = ComputeSubtreeHeight(host_bvh, upper.i, visited);
  if (left_height == -1 || right_height == -1) return -1;
  return 1 + std::max(left_height, right_height);
}

// Recursive helper for counting leaves with visited set.
void SyclProximityEngineAttorney::CountLeavesRecursive(
    const HostBVH& host_bvh, int node_index, std::vector<bool>& visited,
    int* count) {
  if (node_index < 0 || node_index >= host_bvh.max_nodes ||
      visited[node_index]) {
    return;
  }
  visited[node_index] = true;
  const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
  if (lower.b == 1) {  // Leaf.
    ++(*count);
    return;
  }
  const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
  CountLeavesRecursive(host_bvh, lower.i, visited, count);
  CountLeavesRecursive(host_bvh, upper.i, visited, count);
}

// Recursive helper for balance factor.
void SyclProximityEngineAttorney::ComputeBalanceRecursive(
    const HostBVH& host_bvh, int node_index, std::vector<bool>& visited,
    int* max_imbalance) {
  if (node_index < 0 || node_index >= host_bvh.max_nodes ||
      visited[node_index]) {
    return;
  }
  visited[node_index] = true;
  const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
  if (lower.b == 1) return;  // Leaf.

  const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
  std::vector<bool> left_visited = visited;  // Copy to avoid interference.
  std::vector<bool> right_visited = visited;
  int left_height = ComputeSubtreeHeight(host_bvh, lower.i, left_visited);
  int right_height = ComputeSubtreeHeight(host_bvh, upper.i, right_visited);
  if (left_height != -1 && right_height != -1) {
    *max_imbalance =
        std::max(*max_imbalance, std::abs(left_height - right_height));
  }
  ComputeBalanceRecursive(host_bvh, lower.i, visited, max_imbalance);
  ComputeBalanceRecursive(host_bvh, upper.i, visited, max_imbalance);
}

// Recursive helper for average leaf depth.
void SyclProximityEngineAttorney::ComputeDepthsRecursive(
    const HostBVH& host_bvh, int node_index, std::vector<bool>& visited,
    int current_depth, int* total_depth, int* leaf_count) {
  if (node_index < 0 || node_index >= host_bvh.max_nodes ||
      visited[node_index]) {
    return;
  }
  visited[node_index] = true;
  const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
  if (lower.b == 1) {  // Leaf.
    *total_depth += current_depth;
    ++(*leaf_count);
    return;
  }
  const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
  ComputeDepthsRecursive(host_bvh, lower.i, visited, current_depth + 1,
                         total_depth, leaf_count);
  ComputeDepthsRecursive(host_bvh, upper.i, visited, current_depth + 1,
                         total_depth, leaf_count);
}

// Recursive helper to verify bounds (e.g., parent bounds == union of
// children). Assumes Vector3<double> has cwiseMin/cwiseMax.
bool SyclProximityEngineAttorney::VerifyBoundsRecursive(
    const HostBVH& host_bvh, int node_index, std::vector<bool>& visited) {
  if (node_index < 0 || node_index >= host_bvh.max_nodes ||
      visited[node_index]) {
    return false;
  }
  visited[node_index] = true;
  const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
  const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];

  Vector3<double> parent_lower(lower.x, lower.y, lower.z);
  Vector3<double> parent_upper(upper.x, upper.y, upper.z);

  if (lower.b == 1) {
    // leafs : Check the maximum primitves per leaf
    // get their AABBs and test if the node is the union of all the
    // primitive AABBs
    const unsigned int start_index = lower.i;
    const unsigned int end_index = upper.i;
    const unsigned int num_primitives = end_index - start_index;
    Vector3<double> obtained_lower(lower.x, lower.y, lower.z);
    Vector3<double> obtained_upper(upper.x, upper.y, upper.z);

    for (unsigned int i = start_index; i < end_index; ++i) {
      const BVHPackedNodeHalf& primitive_lower = host_bvh.node_lowers[i];
      const BVHPackedNodeHalf& primitive_upper = host_bvh.node_uppers[i];
      Vector3<double> current_lower(primitive_lower.x, primitive_lower.y,
                                    primitive_lower.z);
      Vector3<double> current_upper(primitive_upper.x, primitive_upper.y,
                                    primitive_upper.z);
      obtained_lower = obtained_lower.cwiseMin(current_lower);
      obtained_upper = obtained_upper.cwiseMax(current_upper);
    }
    // Check if parent bounds properly enclose the obtained bounds
    if (obtained_lower[0] < parent_lower[0] ||
        obtained_lower[1] < parent_lower[1] ||
        obtained_lower[2] < parent_lower[2] ||
        obtained_upper[0] > parent_upper[0] ||
        obtained_upper[1] > parent_upper[1] ||
        obtained_upper[2] > parent_upper[2]) {
      return false;
    }
    return true;
  }

  // Check left child.
  const BVHPackedNodeHalf& left_lower = host_bvh.node_lowers[lower.i];
  const BVHPackedNodeHalf& left_upper = host_bvh.node_uppers[lower.i];
  Vector3<double> left_min(left_lower.x, left_lower.y, left_lower.z);
  Vector3<double> left_max(left_upper.x, left_upper.y, left_upper.z);

  // Check right child.
  const BVHPackedNodeHalf& right_lower = host_bvh.node_lowers[upper.i];
  const BVHPackedNodeHalf& right_upper = host_bvh.node_uppers[upper.i];
  Vector3<double> right_min(right_lower.x, right_lower.y, right_lower.z);
  Vector3<double> right_max(right_upper.x, right_upper.y, right_upper.z);

  // Verify parent is union.
  Vector3<double> expected_min = left_min.cwiseMin(right_min);
  Vector3<double> expected_max = left_max.cwiseMax(right_max);
  const double kEpsilon = 1e-8;
  bool bounds_valid = (parent_lower[0] <= expected_min[0] + kEpsilon) &&
                      (parent_lower[1] <= expected_min[1] + kEpsilon) &&
                      (parent_lower[2] <= expected_min[2] + kEpsilon) &&
                      (parent_upper[0] >= expected_max[0] - kEpsilon) &&
                      (parent_upper[1] >= expected_max[1] - kEpsilon) &&
                      (parent_upper[2] >= expected_max[2] - kEpsilon);

  return bounds_valid && VerifyBoundsRecursive(host_bvh, lower.i, visited) &&
         VerifyBoundsRecursive(host_bvh, upper.i, visited);
}

// Computes heights for all nodes using memoization.
// Returns vector of heights, or empty if invalid.
std::vector<int> SyclProximityEngineAttorney::ComputeAllHeights(
    const HostBVH& host_bvh) {
  std::vector<int> heights(host_bvh.max_nodes, -1);
  if (!ComputeHeightMemo(host_bvh, host_bvh.root_index, heights)) {
    return {};
  }
  return heights;
}

// Recursive memoized height computation. Returns true if valid.
bool SyclProximityEngineAttorney::ComputeHeightMemo(const HostBVH& host_bvh,
                                                    int node,
                                                    std::vector<int>& heights) {
  if (node < 0 || node >= host_bvh.max_nodes) return false;
  if (heights[node] != -1) return true;

  const auto& lower = host_bvh.node_lowers[node];
  if (lower.b == 1) {  // Leaf
    heights[node] = 0;
    return true;
  }

  if (!ComputeHeightMemo(host_bvh, lower.i, heights)) return false;
  if (!ComputeHeightMemo(host_bvh, host_bvh.node_uppers[node].i, heights))
    return false;

  int left_h = heights[lower.i];
  int right_h = heights[host_bvh.node_uppers[node].i];
  heights[node] = 1 + std::max(left_h, right_h);
  return true;
}

void SyclProximityEngineAttorney::PrintHistogramJSON(
    const std::vector<int>& histogram, const std::string& filepath) {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open histogram.json" << std::endl;
    return;
  }
  file << "{" << std::endl;
  file << "  \"histogram\": [" << std::endl;
  for (size_t i = 0; i < histogram.size(); ++i) {
    file << "    " << histogram[i];
    if (i < histogram.size() - 1) {
      file << ",";
    }
    file << std::endl;
  }
  file << "  ]" << std::endl;
  file << "}" << std::endl;
  file.close();
}

void SyclProximityEngineAttorney::PrintNodeBoundingBoxesBFS(
    const HostBVH& host_bvh, int root_index,
    const HostIndicesAll& host_indices_all, const HostMeshData& mesh_data,
    const int sorted_mesh_id) {
  if (root_index < 0 || root_index >= host_bvh.max_nodes) {
    return;
  }
  const int element_offset = mesh_data.element_offsets[sorted_mesh_id];
  std::vector<bool> visited(host_bvh.max_nodes, false);
  std::queue<int> q;
  q.push(root_index);
  visited[root_index] = true;
  while (!q.empty()) {
    int node_index = q.front();
    q.pop();
    const auto& lower = host_bvh.node_lowers[node_index];
    const auto& upper = host_bvh.node_uppers[node_index];
    std::cout << "Node " << node_index << ": "
              << "Lower = (" << lower.x << ", " << lower.y << ", " << lower.z
              << ") "
              << "Upper = (" << upper.x << ", " << upper.y << ", " << upper.z
              << ")\n";
    if (lower.b != 1) {  // Not a leaf
      if (lower.i >= 0 && lower.i < host_bvh.max_nodes && !visited[lower.i]) {
        q.push(lower.i);
        visited[lower.i] = true;
      }
      int right_index = host_bvh.node_uppers[node_index].i;
      if (right_index >= 0 && right_index < host_bvh.max_nodes &&
          !visited[right_index]) {
        q.push(right_index);
        visited[right_index] = true;
      }
    } else {
      // Leaf node
      const unsigned int start_index = lower.i;
      const unsigned int end_index = upper.i;
      for (unsigned int i = start_index; i < end_index; ++i) {
        const unsigned int local_primitive_index =
            host_indices_all.indicesAll[i + element_offset];
        const unsigned int global_primitive_index =
            local_primitive_index + element_offset;
        const Vector3<double> lower_W =
            mesh_data.element_aabb_min_W[global_primitive_index];
        const Vector3<double> upper_W =
            mesh_data.element_aabb_max_W[global_primitive_index];
        std::cout << "Primitive " << global_primitive_index << ": "
                  << "Lower = (" << lower_W[0] << ", " << lower_W[1] << ", "
                  << lower_W[2] << ") "
                  << "Upper = (" << upper_W[0] << ", " << upper_W[1] << ", "
                  << upper_W[2] << ")\n";
      }
    }
  }
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
