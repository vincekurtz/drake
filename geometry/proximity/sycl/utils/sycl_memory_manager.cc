
#include "geometry/proximity/sycl/utils/sycl_memory_manager.h"

#include "geometry/proximity/sycl/utils/sycl_kernel_utils.h"

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Allocate all mesh-related device memory
void SyclMemoryHelper::AllocateMeshMemory(SyclMemoryManager& mem_mgr,
                                          DeviceMeshData& mesh_data,
                                          uint32_t num_geometries) {
  // Allocate lookup arrays (host accessible)
  mesh_data.element_offsets = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.vertex_offsets = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.element_counts = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.vertex_counts = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.geometry_ids = mem_mgr.AllocateHost<GeometryId>(num_geometries);
  mesh_data.transforms = mem_mgr.AllocateHost<double>(num_geometries * 12);
}

void SyclMemoryHelper::AllocateBVHSingleMeshMemory(SyclMemoryManager& mem_mgr,
                                                   BVH& bvh_mesh,
                                                   uint32_t max_nodes) {
  bvh_mesh.node_lowers = mem_mgr.AllocateDevice<BVHPackedNodeHalf>(max_nodes);
  bvh_mesh.node_uppers = mem_mgr.AllocateDevice<BVHPackedNodeHalf>(max_nodes);
  bvh_mesh.node_parents = mem_mgr.AllocateDevice<int>(max_nodes);
  bvh_mesh.root = mem_mgr.AllocateDevice<int>(1);
}

void SyclMemoryHelper::AllocateBVHAllMeshMemory(SyclMemoryManager& mem_mgr,
                                                DeviceBVHData& bvh_data,
                                                uint32_t num_geometries) {
  // The BVH counter data and pointers on Host but the data pointed to by BVH
  // will be on device
  bvh_data.bvhAll = mem_mgr.AllocateHost<BVH>(num_geometries);
  bvh_data.node_counts_per_mesh =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
  bvh_data.node_offsets = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  bvh_data.total_lowerAll =
      mem_mgr.AllocateDevice<Vector3<double>>(num_geometries);
  bvh_data.total_upperAll =
      mem_mgr.AllocateDevice<Vector3<double>>(num_geometries);
  bvh_data.total_inv_edgesAll =
      mem_mgr.AllocateDevice<Vector3<double>>(num_geometries);
}

void SyclMemoryHelper::AllocateBVHAllMeshNodeCountsMemory(
    SyclMemoryManager& mem_mgr, DeviceBVHData& bvh_data) {
  bvh_data.node_mesh_ids =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
  bvh_data.num_childrenAll =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
}

void SyclMemoryHelper::AllocateBVHAllMeshTempMemory(SyclMemoryManager& mem_mgr,
                                                    DeviceBVHData& bvh_data,
                                                    uint32_t total_elements) {
  bvh_data.indicesAll = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  bvh_data.keysAll = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  bvh_data.deltasAll = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  bvh_data.range_leftsAll =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
  bvh_data.range_rightsAll =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
}

void SyclMemoryHelper::AllocateDeviceMeshPairCollidingIndicesMemory(
    SyclMemoryManager& mem_mgr,
    DeviceMeshPairCollidingIndices& mesh_pair_colliding_indices,
    uint32_t new_size) {
  mesh_pair_colliding_indices.collision_indices_A =
      mem_mgr.AllocateDevice<uint32_t>(new_size);
  mesh_pair_colliding_indices.collision_indices_B =
      mem_mgr.AllocateDevice<uint32_t>(new_size);
  mesh_pair_colliding_indices.capacity_ = new_size;
  mesh_pair_colliding_indices.size_ = 0;
}

void SyclMemoryHelper::AllocateDeviceCollisionCountersOffsetsMemoryChunk(
    SyclMemoryManager& mem_mgr,
    DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk,
    uint32_t num_geometries) {
  const uint32_t max_candidate_pairs =
      num_geometries * (num_geometries - 1) / 2;
  counters_offsets_chunk.mesh_a_offsets =
      mem_mgr.AllocateDevice<uint32_t>(max_candidate_pairs);
  counters_offsets_chunk.capacity_ = max_candidate_pairs;
  counters_offsets_chunk.size_ = 0;
}

void SyclMemoryHelper::AllocateDeviceCollidingIndicesMemoryChunk(
    SyclMemoryManager& mem_mgr, DeviceCollidingIndicesMemoryChunk& pair_chunk,
    DeviceCollisionCountersMemoryChunk& counters_chunk,
    std::unordered_map<uint64_t, std::pair<DeviceMeshACollisionCounters,
                                           DeviceMeshPairCollidingIndices>>&
        collision_candidates_to_data,
    const std::vector<std::pair<uint32_t, uint32_t>>& collision_candidates,
    const DeviceMeshData& mesh_data, DeviceMeshPairIds& mesh_pair_ids) {
  // Compute the heuristic total amount of memory needed for all collision
  // pairs of all the meshes in the scene This will be resized based on the
  // number of collisions found
  uint32_t total_size_needed_pairs = 0;
  uint32_t total_size_needed_counters = 0;
  for (const auto& pair : collision_candidates) {
    total_size_needed_counters += mesh_data.element_counts[pair.first];
    total_size_needed_pairs +=
        std::max(1u, mesh_data.element_counts[pair.first] *
                         mesh_data.element_counts[pair.second] / 10);
  }
  // Pairs memory

  // Round such that memory is allocated in whole MBs
  // Always round up
  // Convert count to bytes, round to MB, then convert back to count
  uint32_t bytes_needed = total_size_needed_pairs * sizeof(uint32_t);
  uint32_t mb_needed = std::max(
      1u, static_cast<uint32_t>(
              std::ceil(static_cast<double>(bytes_needed) / (1024 * 1024))));
  uint32_t rounded_size_needed = (mb_needed * 1024 * 1024) / sizeof(uint32_t);

  // Shrink if needed size is less than 50% of current capacity. This prevents
  // thrashing when size fluctuates around the current capacity.
  constexpr uint32_t shrink_threshold_percent = 50;
  constexpr uint32_t min_buffer_size_mb = 1;
  uint32_t min_buffer_size =
      (min_buffer_size_mb * 1024 * 1024) / sizeof(uint32_t);

  // Ensure we don't shrink below minimum size.
  rounded_size_needed = std::max(rounded_size_needed, min_buffer_size);

  uint32_t shrink_threshold =
      (pair_chunk.capacity_ * shrink_threshold_percent) / 100;

  // Check if we need to resize the chunk (grow or shrink).
  bool need_resize = false;
  if (rounded_size_needed > pair_chunk.capacity_) {
    // Need to grow
    need_resize = true;
  } else if (rounded_size_needed < shrink_threshold &&
             pair_chunk.capacity_ > 0) {
    // Need to shrink (only if we have existing capacity and new size is
    // significantly smaller).
    need_resize = true;
  }

  if (need_resize) {
    if (pair_chunk.collision_indices_A)
      mem_mgr.Free(pair_chunk.collision_indices_A);
    if (pair_chunk.collision_indices_B)
      mem_mgr.Free(pair_chunk.collision_indices_B);

    pair_chunk.collision_indices_A =
        mem_mgr.AllocateDevice<uint32_t>(rounded_size_needed);
    pair_chunk.collision_indices_B =
        mem_mgr.AllocateDevice<uint32_t>(rounded_size_needed);
    pair_chunk.capacity_ = rounded_size_needed;
  }
  pair_chunk.size_ = 0;

  // Counters memory
  bytes_needed = total_size_needed_counters * sizeof(uint32_t);
  mb_needed = std::max(
      1u, static_cast<uint32_t>(
              std::ceil(static_cast<double>(bytes_needed) / (1024 * 1024))));
  rounded_size_needed = (mb_needed * 1024 * 1024) / sizeof(uint32_t);

  rounded_size_needed = std::max(rounded_size_needed, min_buffer_size);
  shrink_threshold =
      (counters_chunk.capacity_ * shrink_threshold_percent) / 100;
  need_resize = false;
  if (rounded_size_needed > counters_chunk.capacity_) {
    need_resize = true;
  } else if (rounded_size_needed < shrink_threshold &&
             counters_chunk.capacity_ > 0) {
    need_resize = true;
  }

  if (need_resize) {
    if (counters_chunk.collision_counts)
      mem_mgr.Free(counters_chunk.collision_counts);
    counters_chunk.collision_counts =
        mem_mgr.AllocateDevice<uint32_t>(rounded_size_needed);
    counters_chunk.capacity_ = rounded_size_needed;
  }
  counters_chunk.size_ = 0;

  // Allocate memory for meshAs and meshBs
  if (mesh_pair_ids.meshAs) mem_mgr.Free(mesh_pair_ids.meshAs);
  if (mesh_pair_ids.meshBs) mem_mgr.Free(mesh_pair_ids.meshBs);
  mesh_pair_ids.meshAs =
      mem_mgr.AllocateHost<uint32_t>(collision_candidates.size());
  mesh_pair_ids.meshBs =
      mem_mgr.AllocateHost<uint32_t>(collision_candidates.size());

  // 3) Initialize the map - for collision counters we can set the pointers
  uint32_t running_size = 0;
  collision_candidates_to_data.clear();
  uint32_t counter = 0;
  for (const auto& pair : collision_candidates) {
    // Update the meshAs and meshBs arrays
    mesh_pair_ids.meshAs[counter] = pair.first;
    mesh_pair_ids.meshBs[counter] = pair.second;
    counter++;

    // Update the map - Mainly used for ease of testing
    uint64_t col_key = key(pair.first, pair.second);

    DeviceMeshACollisionCounters cc;
    cc.collision_counts = counters_chunk.collision_counts + running_size;
    cc.total_collisions = 0;
    cc.size_ = mesh_data.element_counts[pair.first];
    cc.last_element_collision_count = 0;
    running_size += cc.size_;

    DeviceMeshPairCollidingIndices ci;
    ci.size_ = 0;
    collision_candidates_to_data[col_key] = std::make_pair(cc, ci);
  }
  counters_chunk.size_ = running_size;
}
void SyclMemoryHelper::ResizeDeviceMeshPairCollidingIndicesMemory(
    SyclMemoryManager& mem_mgr, DeviceMeshPairCollidingIndices& ci,
    const DeviceCollidingIndicesMemoryChunk& pair_chunk,
    const uint32_t new_size, const uint32_t offset) {
  ci.collision_indices_A = pair_chunk.collision_indices_A + offset;
  ci.collision_indices_B = pair_chunk.collision_indices_B + offset;
  // This only needs to be set for the tests to work - the tests directly query
  // the ci pointers for data
  ci.size_ = new_size;
}

void SyclMemoryHelper::AllocateMeshElementVerticesMemory(
    SyclMemoryManager& mem_mgr, DeviceMeshData& mesh_data,
    uint32_t total_elements, uint32_t total_vertices) {
  // Allocate element data
  mesh_data.elements =
      mem_mgr.AllocateDevice<std::array<int, 4>>(total_elements);
  mesh_data.element_mesh_ids = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  mesh_data.inward_normals_M =
      mem_mgr.AllocateDevice<std::array<Vector3<double>, 4>>(total_elements);
  mesh_data.inward_normals_W =
      mem_mgr.AllocateDevice<std::array<Vector3<double>, 4>>(total_elements);
  mesh_data.min_pressures = mem_mgr.AllocateDevice<double>(total_elements);
  mesh_data.max_pressures = mem_mgr.AllocateDevice<double>(total_elements);
  mesh_data.gradient_M_pressure_at_Mo =
      mem_mgr.AllocateDevice<Vector4<double>>(total_elements);
  mesh_data.gradient_W_pressure_at_Wo =
      mem_mgr.AllocateDevice<Vector4<double>>(total_elements);
  mesh_data.element_aabb_min_W =
      mem_mgr.AllocateDevice<Vector3<double>>(total_elements);
  mesh_data.element_aabb_max_W =
      mem_mgr.AllocateDevice<Vector3<double>>(total_elements);

  // Allocate vertex data
  mesh_data.vertices_M =
      mem_mgr.AllocateDevice<Vector3<double>>(total_vertices);
  mesh_data.vertices_W =
      mem_mgr.AllocateDevice<Vector3<double>>(total_vertices);
  mesh_data.pressures = mem_mgr.AllocateDevice<double>(total_vertices);
  mesh_data.vertex_mesh_ids = mem_mgr.AllocateDevice<uint32_t>(total_vertices);
}

// Allocate collision detection memory of arrays based on number of geometries
void SyclMemoryHelper::AllocateGeometryCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
    uint32_t num_geometries) {
  collision_data.total_checks_per_geometry =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
  // geom_collision_filternum_cols[i] is the number of elements that need to
  // be checked with each of the elements of the ith geometry
  // Will be highest for 1st geometry and lowest for the last geometry (due to
  // symmetric nature of collision_filter - we are only consider upper
  // triangle)
  collision_data.geom_collision_filter_num_cols =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
  // Stores the exclusive scan of total checks per geometry
  collision_data.geom_collision_filter_check_offsets =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
}

// Allocate collision detection memory of arrays based on total checks
void SyclMemoryHelper::AllocateTotalChecksCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
    uint32_t total_checks) {
  // Broad phase data
  collision_data.collision_filter =
      mem_mgr.AllocateDevice<uint8_t>(total_checks);
  collision_data.collision_filter_host_body_index =
      mem_mgr.AllocateHost<uint32_t>(total_checks);
  collision_data.prefix_sum_total_checks =
      mem_mgr.AllocateDevice<uint32_t>(total_checks);
}

// Allocate collision detection memory of arrays based on estimated narrow
// phase checks
void SyclMemoryHelper::AllocateNarrowPhaseChecksCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
    uint32_t estimated_narrow_phase_checks) {
  // Narrow phase data
  //   collision_data.narrow_phase_check_indices =
  //       mem_mgr.AllocateDevice<uint32_t>(estimated_narrow_phase_checks);
  collision_data.narrow_phase_check_validity =
      mem_mgr.AllocateDevice<uint8_t>(estimated_narrow_phase_checks);
  collision_data.prefix_sum_narrow_phase_checks =
      mem_mgr.AllocateDevice<uint32_t>(estimated_narrow_phase_checks);
}

// Allocate polygon memory
void SyclMemoryHelper::AllocateFullPolygonMemory(
    SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data,
    uint32_t estimated_narrow_phase_checks) {
  // Raw polygon data
  polygon_data.polygon_areas =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_centroids =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_narrow_phase_checks);
  polygon_data.polygon_normals =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_narrow_phase_checks);
  polygon_data.polygon_g_M =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_g_N =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_pressure_W =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_geom_index_A =
      mem_mgr.AllocateDevice<GeometryId>(estimated_narrow_phase_checks);
  polygon_data.polygon_geom_index_B =
      mem_mgr.AllocateDevice<GeometryId>(estimated_narrow_phase_checks);
}

void SyclMemoryHelper::AllocateCompactPolygonMemory(
    SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data,
    uint32_t estimated_polygons) {
  // Compacted polygon data
  polygon_data.compacted_polygon_areas =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_centroids =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_polygons);
  polygon_data.compacted_polygon_normals =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_polygons);
  polygon_data.compacted_polygon_g_M =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_g_N =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_pressure_W =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_geom_index_A =
      mem_mgr.AllocateDevice<GeometryId>(estimated_polygons);
  polygon_data.compacted_polygon_geom_index_B =
      mem_mgr.AllocateDevice<GeometryId>(estimated_polygons);

  polygon_data.valid_polygon_indices =
      mem_mgr.AllocateDevice<uint32_t>(estimated_polygons);
}

// Free all mesh memory
void SyclMemoryHelper::FreeMeshMemory(SyclMemoryManager& mem_mgr,
                                      DeviceMeshData& mesh_data) {
  // Element data
  mem_mgr.Free(mesh_data.elements);
  mesh_data.elements = nullptr;
  mem_mgr.Free(mesh_data.element_mesh_ids);
  mesh_data.element_mesh_ids = nullptr;
  mem_mgr.Free(mesh_data.inward_normals_M);
  mesh_data.inward_normals_M = nullptr;
  mem_mgr.Free(mesh_data.inward_normals_W);
  mesh_data.inward_normals_W = nullptr;
  mem_mgr.Free(mesh_data.min_pressures);
  mesh_data.min_pressures = nullptr;
  mem_mgr.Free(mesh_data.max_pressures);
  mesh_data.max_pressures = nullptr;
  mem_mgr.Free(mesh_data.gradient_M_pressure_at_Mo);
  mesh_data.gradient_M_pressure_at_Mo = nullptr;
  mem_mgr.Free(mesh_data.gradient_W_pressure_at_Wo);
  mesh_data.gradient_W_pressure_at_Wo = nullptr;
  mem_mgr.Free(mesh_data.element_aabb_min_W);
  mesh_data.element_aabb_min_W = nullptr;
  mem_mgr.Free(mesh_data.element_aabb_max_W);
  mesh_data.element_aabb_max_W = nullptr;

  // Vertex data
  mem_mgr.Free(mesh_data.vertices_M);
  mesh_data.vertices_M = nullptr;
  mem_mgr.Free(mesh_data.vertices_W);
  mesh_data.vertices_W = nullptr;
  mem_mgr.Free(mesh_data.pressures);
  mesh_data.pressures = nullptr;
  mem_mgr.Free(mesh_data.vertex_mesh_ids);
  mesh_data.vertex_mesh_ids = nullptr;

  // Lookup arrays
  mem_mgr.Free(mesh_data.element_offsets);
  mesh_data.element_offsets = nullptr;
  mem_mgr.Free(mesh_data.vertex_offsets);
  mesh_data.vertex_offsets = nullptr;
  mem_mgr.Free(mesh_data.element_counts);
  mesh_data.element_counts = nullptr;
  mem_mgr.Free(mesh_data.vertex_counts);
  mesh_data.vertex_counts = nullptr;
  mem_mgr.Free(mesh_data.geometry_ids);
  mem_mgr.Free(mesh_data.transforms);
}

// Free individual BVH mesh memory
void SyclMemoryHelper::FreeBVHSingleMeshMemory(SyclMemoryManager& mem_mgr,
                                               BVH& bvh_mesh) {
  mem_mgr.Free(bvh_mesh.node_lowers);
  bvh_mesh.node_lowers = nullptr;
  mem_mgr.Free(bvh_mesh.node_uppers);
  bvh_mesh.node_uppers = nullptr;
  mem_mgr.Free(bvh_mesh.node_parents);
  bvh_mesh.node_parents = nullptr;
  mem_mgr.Free(bvh_mesh.root);
  bvh_mesh.root = nullptr;
}
// Free all BVH memory
void SyclMemoryHelper::FreeBVHSingleMeshAndAllMeshMemory(
    SyclMemoryManager& mem_mgr, DeviceBVHData& bvh_data) {
  // Free per-mesh BVH node data first
  if (bvh_data.bvhAll != nullptr) {
    for (uint32_t i = 0; i < bvh_data.num_meshes; ++i) {
      FreeBVHSingleMeshMemory(mem_mgr, bvh_data.bvhAll[i]);
    }
  }

  mem_mgr.Free(bvh_data.bvhAll);
  bvh_data.bvhAll = nullptr;
  mem_mgr.Free(bvh_data.node_counts_per_mesh);
  bvh_data.node_counts_per_mesh = nullptr;
  mem_mgr.Free(bvh_data.node_offsets);
  bvh_data.node_offsets = nullptr;
  mem_mgr.Free(bvh_data.total_lowerAll);
  bvh_data.total_lowerAll = nullptr;
  mem_mgr.Free(bvh_data.total_upperAll);
  bvh_data.total_upperAll = nullptr;
  mem_mgr.Free(bvh_data.total_inv_edgesAll);
  bvh_data.total_inv_edgesAll = nullptr;
  mem_mgr.Free(bvh_data.indicesAll);
  bvh_data.indicesAll = nullptr;
  mem_mgr.Free(bvh_data.node_mesh_ids);
  bvh_data.node_mesh_ids = nullptr;
  mem_mgr.Free(bvh_data.num_childrenAll);
  bvh_data.num_childrenAll = nullptr;
}

void SyclMemoryHelper::FreeBVHAllMeshTempMemory(SyclMemoryManager& mem_mgr,
                                                DeviceBVHData& bvh_data) {
  mem_mgr.Free(bvh_data.keysAll);
  bvh_data.keysAll = nullptr;
  mem_mgr.Free(bvh_data.deltasAll);
  bvh_data.deltasAll = nullptr;
  mem_mgr.Free(bvh_data.range_leftsAll);
  bvh_data.range_leftsAll = nullptr;
  mem_mgr.Free(bvh_data.range_rightsAll);
  bvh_data.range_rightsAll = nullptr;
}

// Free collision memory
void SyclMemoryHelper::FreeCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data) {
  mem_mgr.Free(collision_data.collision_filter);
  collision_data.collision_filter = nullptr;
  mem_mgr.Free(collision_data.collision_filter_host_body_index);
  collision_data.collision_filter_host_body_index = nullptr;
  mem_mgr.Free(collision_data.total_checks_per_geometry);
  collision_data.total_checks_per_geometry = nullptr;
  mem_mgr.Free(collision_data.geom_collision_filter_num_cols);
  collision_data.geom_collision_filter_num_cols = nullptr;
  mem_mgr.Free(collision_data.geom_collision_filter_check_offsets);
  collision_data.geom_collision_filter_check_offsets = nullptr;
  mem_mgr.Free(collision_data.prefix_sum_total_checks);
  collision_data.prefix_sum_total_checks = nullptr;
  mem_mgr.Free(collision_data.narrow_phase_check_indices);
  collision_data.narrow_phase_check_indices = nullptr;
  mem_mgr.Free(collision_data.narrow_phase_check_validity);
  collision_data.narrow_phase_check_validity = nullptr;
  mem_mgr.Free(collision_data.prefix_sum_narrow_phase_checks);
  collision_data.prefix_sum_narrow_phase_checks = nullptr;
}

// Free only the collision detection memory of arrays based on narrow phase
// checks
void SyclMemoryHelper::FreeNarrowPhaseChecksCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data) {
  //   mem_mgr.Free(collision_data.narrow_phase_check_indices);
  mem_mgr.Free(collision_data.narrow_phase_check_validity);
  collision_data.narrow_phase_check_validity = nullptr;
  mem_mgr.Free(collision_data.prefix_sum_narrow_phase_checks);
  collision_data.prefix_sum_narrow_phase_checks = nullptr;
}

// Free polygon memory
void SyclMemoryHelper::FreeFullPolygonMemory(SyclMemoryManager& mem_mgr,
                                             DevicePolygonData& polygon_data) {
  // Raw polygon data
  mem_mgr.Free(polygon_data.polygon_areas);
  polygon_data.polygon_areas = nullptr;
  mem_mgr.Free(polygon_data.polygon_centroids);
  polygon_data.polygon_centroids = nullptr;
  mem_mgr.Free(polygon_data.polygon_normals);
  polygon_data.polygon_normals = nullptr;
  mem_mgr.Free(polygon_data.polygon_g_M);
  polygon_data.polygon_g_M = nullptr;
  mem_mgr.Free(polygon_data.polygon_g_N);
  polygon_data.polygon_g_N = nullptr;
  mem_mgr.Free(polygon_data.polygon_pressure_W);
  polygon_data.polygon_pressure_W = nullptr;
  mem_mgr.Free(polygon_data.polygon_geom_index_A);
  polygon_data.polygon_geom_index_A = nullptr;
  mem_mgr.Free(polygon_data.polygon_geom_index_B);
  polygon_data.polygon_geom_index_B = nullptr;

  // Debug data
  mem_mgr.Free(polygon_data.debug_polygon_vertices);
  polygon_data.debug_polygon_vertices = nullptr;
}

void SyclMemoryHelper::FreeCompactPolygonMemory(
    SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data) {
  // Compacted polygon data
  mem_mgr.Free(polygon_data.compacted_polygon_areas);
  polygon_data.compacted_polygon_areas = nullptr;
  mem_mgr.Free(polygon_data.compacted_polygon_centroids);
  polygon_data.compacted_polygon_centroids = nullptr;
  mem_mgr.Free(polygon_data.compacted_polygon_normals);
  polygon_data.compacted_polygon_normals = nullptr;
  mem_mgr.Free(polygon_data.compacted_polygon_g_M);
  polygon_data.compacted_polygon_g_M = nullptr;
  mem_mgr.Free(polygon_data.compacted_polygon_g_N);
  polygon_data.compacted_polygon_g_N = nullptr;
  mem_mgr.Free(polygon_data.compacted_polygon_pressure_W);
  polygon_data.compacted_polygon_pressure_W = nullptr;
  mem_mgr.Free(polygon_data.compacted_polygon_geom_index_A);
  polygon_data.compacted_polygon_geom_index_A = nullptr;
  mem_mgr.Free(polygon_data.compacted_polygon_geom_index_B);
  polygon_data.compacted_polygon_geom_index_B = nullptr;
  mem_mgr.Free(polygon_data.valid_polygon_indices);
  polygon_data.valid_polygon_indices = nullptr;
}

void SyclMemoryHelper::FreePolygonMemory(SyclMemoryManager& mem_mgr,
                                         DevicePolygonData& polygon_data) {
  FreeFullPolygonMemory(mem_mgr, polygon_data);
  FreeCompactPolygonMemory(mem_mgr, polygon_data);
}

void SyclMemoryHelper::FreeDeviceCollisionCountersOffsetsMemoryChunk(
    SyclMemoryManager& mem_mgr,
    DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk) {
  mem_mgr.Free(counters_offsets_chunk.mesh_a_offsets);
  counters_offsets_chunk.mesh_a_offsets = nullptr;
  counters_offsets_chunk.capacity_ = 0;
  counters_offsets_chunk.size_ = 0;
}

void SyclMemoryHelper::FreeDeviceCollidingIndicesMemoryChunk(
    SyclMemoryManager& mem_mgr, DeviceCollidingIndicesMemoryChunk& pair_chunk) {
  mem_mgr.Free(pair_chunk.collision_indices_A);
  pair_chunk.collision_indices_A = nullptr;
  mem_mgr.Free(pair_chunk.collision_indices_B);
  pair_chunk.collision_indices_B = nullptr;
  pair_chunk.capacity_ = 0;
  pair_chunk.size_ = 0;
}

void SyclMemoryHelper::FreeDeviceCollisionCountersMemoryChunk(
    SyclMemoryManager& mem_mgr,
    DeviceCollisionCountersMemoryChunk& counters_chunk) {
  mem_mgr.Free(counters_chunk.collision_counts);
  counters_chunk.collision_counts = nullptr;
  counters_chunk.capacity_ = 0;
  counters_chunk.size_ = 0;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake