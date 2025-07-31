#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/sycl/utils/sycl_bvh_structs.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Helper class to manage SYCL device memory allocations and transfers
class SyclMemoryManager {
 public:
  explicit SyclMemoryManager(sycl::queue& queue) : queue_(queue) {}

  // Allocate device memory for basic types
  template <typename T>
  inline T* AllocateDevice(uint32_t count) {
    return sycl::malloc_device<T>(count, queue_);
  }

  // Allocate host-accessible memory for basic types
  template <typename T>
  inline T* AllocateHost(uint32_t count) {
    return sycl::malloc_host<T>(count, queue_);
  }

  // Free device/host memory
  template <typename T>
  inline void Free(T* ptr) {
    if (ptr != nullptr) {
      sycl::free(ptr, queue_);
    }
  }

  // Copy data from host to device
  template <typename T>
  inline sycl::event CopyToDevice(T* device_ptr, const T* host_ptr,
                                  uint32_t count) {
    return queue_.memcpy(device_ptr, host_ptr, count * sizeof(T));
  }

  // Copy data from device to host
  template <typename T>
  inline sycl::event CopyToHost(T* host_ptr, const T* device_ptr,
                                uint32_t count) {
    return queue_.memcpy(host_ptr, device_ptr, count * sizeof(T));
  }

  // Fill device memory with a value
  template <typename T>
  inline sycl::event Fill(T* device_ptr, const T& value, uint32_t count) {
    return queue_.fill(device_ptr, value, count);
  }

  // Memset device memory to zero
  template <typename T>
  inline sycl::event Memset(T* device_ptr, uint32_t count) {
    return queue_.memset(device_ptr, 0, count * sizeof(T));
  }

 private:
  sycl::queue& queue_;
};

// Structure to hold all device memory pointers for mesh data
struct DeviceMeshData {
  // Element data
  std::array<int, 4>* elements = nullptr;
  uint32_t* element_mesh_ids = nullptr;
  std::array<Vector3<double>, 4>* inward_normals_M = nullptr;
  std::array<Vector3<double>, 4>* inward_normals_W = nullptr;
  double* min_pressures = nullptr;
  double* max_pressures = nullptr;
  Vector4<double>* gradient_M_pressure_at_Mo = nullptr;
  Vector4<double>* gradient_W_pressure_at_Wo = nullptr;
  Vector3<double>* element_aabb_min_W = nullptr;
  Vector3<double>* element_aabb_max_W = nullptr;

  // Vertex data
  Vector3<double>* vertices_M = nullptr;
  Vector3<double>* vertices_W = nullptr;
  double* pressures = nullptr;
  uint32_t* vertex_mesh_ids = nullptr;

  // Lookup arrays (host accessible)
  uint32_t* element_offsets = nullptr;
  uint32_t* vertex_offsets = nullptr;
  uint32_t* element_counts = nullptr;
  uint32_t* vertex_counts = nullptr;
  GeometryId* geometry_ids = nullptr;
  double* transforms = nullptr;
  uint32_t total_elements;
  uint32_t total_vertices;
};

// Structure to hold collision detection memory
struct DeviceCollisionData {
  // Broad phase data
  uint8_t* collision_filter = nullptr;
  uint32_t* collision_filter_host_body_index = nullptr;
  uint32_t* total_checks_per_geometry = nullptr;
  uint32_t* geom_collision_filter_num_cols = nullptr;
  uint32_t* geom_collision_filter_check_offsets = nullptr;
  uint32_t* prefix_sum_total_checks = nullptr;

  // Narrow phase data
  uint32_t* narrow_phase_check_indices = nullptr;
  uint8_t* narrow_phase_check_validity = nullptr;
  uint32_t* prefix_sum_narrow_phase_checks = nullptr;
};

// Structure to hold polygon data memory
struct DevicePolygonData {
  // Raw polygon data
  double* polygon_areas = nullptr;
  Vector3<double>* polygon_centroids = nullptr;
  Vector3<double>* polygon_normals = nullptr;
  double* polygon_g_M = nullptr;
  double* polygon_g_N = nullptr;
  double* polygon_pressure_W = nullptr;
  GeometryId* polygon_geom_index_A = nullptr;
  GeometryId* polygon_geom_index_B = nullptr;

  // Compacted polygon data
  double* compacted_polygon_areas = nullptr;
  Vector3<double>* compacted_polygon_centroids = nullptr;
  Vector3<double>* compacted_polygon_normals = nullptr;
  double* compacted_polygon_g_M = nullptr;
  double* compacted_polygon_g_N = nullptr;
  double* compacted_polygon_pressure_W = nullptr;
  GeometryId* compacted_polygon_geom_index_A = nullptr;
  GeometryId* compacted_polygon_geom_index_B = nullptr;

  uint32_t* valid_polygon_indices = nullptr;

  // Debug data
  double* debug_polygon_vertices = nullptr;
};

struct DeviceBVHData {
  // Permenant data only deleted with the SYCL proximity engine
  BVH* bvhAll = nullptr;
  uint32_t* node_counts_per_mesh = nullptr;
  uint32_t* node_offsets = nullptr;
  // This is modified in place to point to mesh local primitive index
  // If this is used again to get primitive AABBs from mesh_data, it needs the
  // mesh wise element offset added to it
  uint32_t* indicesAll = nullptr;
  // Mesh ID corresponding to each node
  uint32_t* node_mesh_ids = nullptr;
  uint32_t* num_childrenAll = nullptr;
  Vector3<double>* total_lowerAll = nullptr;
  Vector3<double>* total_upperAll = nullptr;
  Vector3<double>* total_inv_edgesAll = nullptr;

  // Temp data deleted after tree construction
  uint32_t* keysAll = nullptr;  // Morton keys of all elements
  uint32_t* deltasAll =
      nullptr;  // deltasAll[index] is the delta of key index and index+1
  uint32_t* range_leftsAll =
      nullptr;  // Each node stores the range of primitives it covers. This is
                // the left limit of the range
  uint32_t* range_rightsAll = nullptr;  // This is the right limit of the range
  uint32_t num_meshes;
  uint32_t total_nodes;
};

struct DeviceMeshPairCollidingIndices {
  uint32_t capacity_ = 0;
  uint32_t size_ = 0;
  uint32_t* collision_indices_A = nullptr;
  uint32_t* collision_indices_B = nullptr;
};

struct DeviceMeshACollisionCounters {
  uint32_t* collision_counts = nullptr;
  uint32_t size_ = 0;
  uint32_t total_collisions = 0;
  uint32_t last_element_collision_count = 0;
};

struct DeviceCollidingIndicesMemoryChunk {
  uint32_t* collision_indices_A = nullptr;
  uint32_t* collision_indices_B = nullptr;
  uint32_t capacity_ = 0;
  uint32_t size_ = 0;
};

struct DeviceCollisionCountersMemoryChunk {
  uint32_t* collision_counts = nullptr;
  uint32_t capacity_ = 0;
  uint32_t size_ = 0;
  uint32_t last_element_collision_count = 0;
  uint32_t total_collisions = 0;
};

struct DeviceCollisionCountersOffsetsMemoryChunk {
  uint32_t* mesh_a_offsets = nullptr;
  uint32_t capacity_ = 0;
  uint32_t size_ = 0;
};

struct DeviceMeshPairIds {
  uint32_t* meshAs = nullptr;
  uint32_t* meshBs = nullptr;
};

class SyclMemoryHelper {
 public:
  static void AllocateMeshMemory(SyclMemoryManager& mem_mgr,
                                 DeviceMeshData& mesh_data,
                                 uint32_t num_geometries);
  // Memory each BVH holds for itself
  static void AllocateBVHSingleMeshMemory(SyclMemoryManager& mem_mgr,
                                          BVH& bvh_mesh, uint32_t max_nodes);
  // Memory required to construct the BVH of all meshes
  // Note 1: Some memory is temporary and can be freed after construction - this
  // is allocated in the second function (AllocateBVHAllMeshTempMemory) Some
  // memory is permanent and is referenced by the BVH - this is allocated in the
  // first function Note 2: Some memory is node based and is allocated in
  // AllocateBVHAllMeshNodeCountsMemory However all of the non temporary AllMesh
  // memory is freed in FreeBVHSingleMeshAndAllMeshMemory
  static void AllocateBVHAllMeshMemory(SyclMemoryManager& mem_mgr,
                                       DeviceBVHData& bvh_data,
                                       uint32_t num_geometries);
  static void AllocateBVHAllMeshNodeCountsMemory(SyclMemoryManager& mem_mgr,
                                                 DeviceBVHData& bvh_data);
  static void AllocateBVHAllMeshTempMemory(SyclMemoryManager& mem_mgr,
                                           DeviceBVHData& bvh_data,
                                           uint32_t total_elements);
  static void AllocateMeshElementVerticesMemory(SyclMemoryManager& mem_mgr,
                                                DeviceMeshData& mesh_data,
                                                uint32_t total_elements,
                                                uint32_t total_vertices);
  static void AllocateGeometryCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
      uint32_t num_geometries);
  static void AllocateTotalChecksCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
      uint32_t num_geometries);
  static void AllocateNarrowPhaseChecksCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
      uint32_t num_geometries);
  static void AllocateFullPolygonMemory(SyclMemoryManager& mem_mgr,
                                        DevicePolygonData& polygon_data,
                                        uint32_t num_geometries);
  static void AllocateCompactPolygonMemory(SyclMemoryManager& mem_mgr,
                                           DevicePolygonData& polygon_data,
                                           uint32_t num_geometries);

  static void AllocateDeviceMeshPairCollidingIndicesMemory(
      SyclMemoryManager& mem_mgr,
      DeviceMeshPairCollidingIndices& mesh_pair_colliding_indices,
      uint32_t new_size);
  static void AllocateDeviceCollisionCountersOffsetsMemoryChunk(
      SyclMemoryManager& mem_mgr,
      DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk,
      uint32_t num_geometries);

  // Store pointers to location in collision indices chunk memory to store
  // collision pair indices for the mesh pair that holds this
  // mesh_pair_colliding_indices
  static void ResizeDeviceMeshPairCollidingIndicesMemory(
      SyclMemoryManager& mem_mgr, DeviceMeshPairCollidingIndices& ci,
      const DeviceCollidingIndicesMemoryChunk& pair_chunk,
      const uint32_t new_size, const uint32_t offset);
  // Allocates memory for the chunk of memory that holds all the collision
  // indices (global) and collision counters for each MeshA in the
  // collision_candidates_
  static void AllocateDeviceCollidingIndicesMemoryChunk(
      SyclMemoryManager& mem_mgr, DeviceCollidingIndicesMemoryChunk& pair_chunk,
      DeviceCollisionCountersMemoryChunk& counters_chunk,
      std::unordered_map<uint64_t, std::pair<DeviceMeshACollisionCounters,
                                             DeviceMeshPairCollidingIndices>>&
          collision_candidates_to_data,
      const std::vector<std::pair<uint32_t, uint32_t>>& collision_candidates,
      const DeviceMeshData& mesh_data, DeviceMeshPairIds& mesh_pair_ids);

  static void FreeMeshMemory(SyclMemoryManager& mem_mgr,
                             DeviceMeshData& mesh_data);
  static void FreeBVHSingleMeshAndAllMeshMemory(SyclMemoryManager& mem_mgr,
                                                DeviceBVHData& bvh_data);
  static void FreeBVHAllMeshTempMemory(SyclMemoryManager& mem_mgr,
                                       DeviceBVHData& bvh_data);
  static void FreeCollisionMemory(SyclMemoryManager& mem_mgr,
                                  DeviceCollisionData& collision_data);
  static void FreeNarrowPhaseChecksCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data);
  static void FreeFullPolygonMemory(SyclMemoryManager& mem_mgr,
                                    DevicePolygonData& polygon_data);
  static void FreeCompactPolygonMemory(SyclMemoryManager& mem_mgr,
                                       DevicePolygonData& polygon_data);
  static void FreePolygonMemory(SyclMemoryManager& mem_mgr,
                                DevicePolygonData& polygon_data);
  static void FreeDeviceCollisionCountersOffsetsMemoryChunk(
      SyclMemoryManager& mem_mgr,
      DeviceCollisionCountersOffsetsMemoryChunk& counters_offsets_chunk);
  static void FreeDeviceCollidingIndicesMemoryChunk(
      SyclMemoryManager& mem_mgr,
      DeviceCollidingIndicesMemoryChunk& pair_chunk);

  static void FreeDeviceCollisionCountersMemoryChunk(
      SyclMemoryManager& mem_mgr,
      DeviceCollisionCountersMemoryChunk& counters_chunk);

 private:
  static void FreeBVHSingleMeshMemory(SyclMemoryManager& mem_mgr,
                                      BVH& bvh_mesh);
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake