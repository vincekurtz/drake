#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "drake/common/sorted_pair.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/geometry/proximity/sycl/utils/sycl_host_structs.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

/* This class implements a SYCL compatible version for performing geometric
 * _proximity_ queries. It is instantiated with soft geometry instances lazily
 * when the contact surfaces are to be computed (after ALL geometries of the
 * scene have been instantiated).
 *
 * To provide geometric queries on the geometries, it provides a public member
 * function that takes a map of the geometry ID and its pose. This function
 * requires the collision candidates to be updated first and is the
 * responsibility of the client through UpdateCollisionCandidates().
 *
 * This class employs the PIMPL (Pointer to IMPLementation) idiom primarily
 * because it isolates SYCL-specific code and dependencies to the implementation
 * file, preventing SYCL header files from being transitively included in
 * client code.
 */
class SyclProximityEngine {
 public:
  // Explicitly declare copy constructor and assignment operator
  SyclProximityEngine(const SyclProximityEngine&);
  SyclProximityEngine& operator=(const SyclProximityEngine&);
  /* @returns true iff the SYCL implementation is available. */
  static bool is_available();

  /* @param soft_geometries The soft geometries to use for the proximity
   * queries. To be supplied lazily when contact surface is to be computed. */
  SyclProximityEngine(
      const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
          soft_geometries,
      const std::unordered_map<GeometryId, Vector3<double>>& total_lower,
      const std::unordered_map<GeometryId, Vector3<double>>& total_upper);

  /* Default constructor creates an empty engine. */
  SyclProximityEngine();

  ~SyclProximityEngine();

  /* @param collision_candidates New vector of collision candidates after
   * broad phase collision detection. */
  void UpdateCollisionCandidates(
      const std::vector<SortedPair<GeometryId>>& collision_candidates);

  /* @param X_WGs The poses of the geometries to compute the contact surface
   * for.
   * @returns A vector of SYCLHydroelasticSurfaces from each candidate collision
   * pair of geometries. The HydroelasticSurface itself holds the Id's of the
   * geometries that it belongs to.*/
  std::vector<SYCLHydroelasticSurface> ComputeSYCLHydroelasticSurface(
      const std::unordered_map<GeometryId, math::RigidTransform<double>>&
          X_WGs);

  /* Prints timing statistics for all SYCL kernels if timing is enabled.
   * This method has no effect if DRAKE_SYCL_TIMING_ENABLED is not defined. */
  void PrintTimingStats() const;

  /* Prints timing statistics for all SYCL kernels in JSON format if timing is
   * enabled. This method has no effect if DRAKE_SYCL_TIMING_ENABLED is not
   * defined. */
  void PrintTimingStatsJson(const std::string& path) const;

 private:
  // The implementation class
  class Impl;
  std::unique_ptr<Impl> impl_;
  // Add attorney as friend
  friend class SyclProximityEngineAttorney;
};

// Attorney class for accessing private members of SyclProximityEngine and Impl
class SyclProximityEngineAttorney {
 public:
  static SyclProximityEngine::Impl* get_impl(SyclProximityEngine& engine);
  static const SyclProximityEngine::Impl* get_impl(
      const SyclProximityEngine& engine);

  static std::vector<uint8_t> get_collision_filter(
      SyclProximityEngine::Impl* impl);
  static std::vector<uint32_t> get_prefix_sum(SyclProximityEngine::Impl* impl);
  static std::vector<Vector3<double>> get_vertices_M(
      SyclProximityEngine::Impl* impl);
  static std::vector<Vector3<double>> get_vertices_W(
      SyclProximityEngine::Impl* impl);
  static std::vector<std::array<int, 4>> get_elements(
      SyclProximityEngine::Impl* impl);
  static double* get_pressures(SyclProximityEngine::Impl* impl);
  static Vector4<double>* get_gradient_M_pressure_at_Mo(
      SyclProximityEngine::Impl* impl);
  static Vector4<double>* get_gradient_W_pressure_at_Wo(
      SyclProximityEngine::Impl* impl);
  static uint32_t* get_collision_filter_host_body_index(
      SyclProximityEngine::Impl* impl);
  static uint32_t get_total_checks(SyclProximityEngine::Impl* impl);
  static uint32_t get_total_narrow_phase_checks(
      SyclProximityEngine::Impl* impl);
  static uint32_t get_total_polygons(SyclProximityEngine::Impl* impl);
  static std::vector<uint32_t> get_narrow_phase_check_indices(
      SyclProximityEngine::Impl* impl);
  static std::vector<uint32_t> get_valid_polygon_indices(
      SyclProximityEngine::Impl* impl);
  static std::vector<double> get_polygon_areas(SyclProximityEngine::Impl* impl);
  static std::vector<Vector3<double>> get_polygon_centroids(
      SyclProximityEngine::Impl* impl);
  static std::vector<double> get_debug_polygon_vertices(
      SyclProximityEngine::Impl* impl);
  // Required for testing the BVH
  static HostMeshData get_mesh_data(SyclProximityEngine::Impl* impl);
  static uint32_t get_num_meshes(SyclProximityEngine::Impl* impl);

  // Testing BVH structure and properties
  static HostBVH get_host_bvh(SyclProximityEngine::Impl* impl,
                              const int sorted_mesh_id);
  static HostIndicesAll get_host_indices_all(SyclProximityEngine::Impl* impl);
  // Computes the height (max depth) of the tree starting from root.
  // Height is the longest path from root to a leaf (edges).
  // Returns -1 if tree is invalid (e.g., cycles or bad structure).
  static int ComputeBVHTreeHeight(const HostBVH& host_bvh);
  // Counts the number of leaf nodes in the tree.
  static int CountBVHLeaves(const HostBVH& host_bvh);
  // Computes a simple balance factor: max(|left_height - right_height|) over
  // all nodes. Returns 0 for perfectly balanced; higher values indicate
  // imbalance. Returns -1 if tree is invalid.
  static int ComputeBVHBalanceFactor(const HostBVH& host_bvh);
  // Computes average depth across all leaves
  static double ComputeBVHAverageLeafDepth(const HostBVH& host_bvh);
  // Verifies if node bounds are correct (e.g., union of children).
  // Returns true if valid for the whole tree.
  static bool VerifyBVHBounds(const HostBVH& host_bvh);

  // Computes a histogram of imbalance factors (|left_height - right_height|)
  // for all internal nodes. Returns a vector where histogram[i] = number of
  // internal nodes with imbalance i. The size is max_imbalance + 1. Returns
  // empty vector if tree is invalid.
  static void ComputeAndPrintBVHImbalanceHistogram(const HostBVH& host_bvh,
                                                   const std::string& filepath);
  // Prints the bounding boxes of each node in the BVH tree to std::cout,
  // traversing from root downwards using BFS.
  static void PrintBVHNodeBoundingBoxes(const HostBVH& host_bvh,
                                        const HostIndicesAll& host_indices_all,
                                        const HostMeshData& mesh_data,
                                        const int sorted_mesh_id);

  // Timing logger access
  static void PrintTimingStats(SyclProximityEngine::Impl* impl);
  static void PrintTimingStatsJson(SyclProximityEngine::Impl* impl,
                                   const std::string& path);
  static std::unordered_map<
      SortedPair<GeometryId>,
      std::pair<HostMeshACollisionCounters, HostMeshPairCollidingIndices>>
  get_collision_candidates_to_data(SyclProximityEngine::Impl* impl);

 private:
  static int ComputeSubtreeHeight(const HostBVH& host_bvh, int node_index,
                                  std::vector<bool>& visited);
  // Recursive helper for counting leaves with visited set.
  static void CountLeavesRecursive(const HostBVH& host_bvh, int node_index,
                                   std::vector<bool>& visited, int* count);

  // Recursive helper for balance factor.
  static void ComputeBalanceRecursive(const HostBVH& host_bvh, int node_index,
                                      std::vector<bool>& visited,
                                      int* max_imbalance);

  // Recursive helper for average leaf depth.
  static void ComputeDepthsRecursive(const HostBVH& host_bvh, int node_index,
                                     std::vector<bool>& visited,
                                     int current_depth, int* total_depth,
                                     int* leaf_count);
  // Recursive helper to verify bounds (e.g., parent bounds == union of
  // children). Assumes Vector3<double> has cwiseMin/cwiseMax.
  static bool VerifyBoundsRecursive(const HostBVH& host_bvh, int node_index,
                                    std::vector<bool>& visited);

  // Computes heights for all nodes using memoization.
  // Returns vector of heights, or empty if invalid.
  static std::vector<int> ComputeAllHeights(const HostBVH& host_bvh);

  // Recursive memoized height computation. Returns true if valid.
  static bool ComputeHeightMemo(const HostBVH& host_bvh, int node,
                                std::vector<int>& heights);

  static void PrintNodeBoundingBoxesBFS(const HostBVH& host_bvh, int root_index,
                                        const HostIndicesAll& host_indices_all,
                                        const HostMeshData& mesh_data,
                                        const int sorted_mesh_id);
  static void PrintHistogramJSON(const std::vector<int>& histogram,
                                 const std::string& filepath);
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake