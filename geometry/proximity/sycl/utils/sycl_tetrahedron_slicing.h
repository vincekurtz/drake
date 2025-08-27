#pragma once

#include <array>
#include <limits>
#include <utility>

#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

#ifdef __SYCL_DEVICE_ONLY__
#define DRAKE_SYCL_DEVICE_INLINE [[sycl::device]]
#else
#define DRAKE_SYCL_DEVICE_INLINE
#endif

// Tetrahedon slice with EqPlane helper code from mesh_plane_intersection.cc

/* This table essentially assigns an index to each edge in the tetrahedron.
 Each edge is represented by its pair of vertex indexes. */
using TetrahedronEdge = std::pair<int, int>;

// Edge definitions for tetrahedron
constexpr std::array<std::pair<int, int>, 6> kTetEdges = {
    // base formed by vertices 0, 1, 2.
    TetrahedronEdge{0, 1}, TetrahedronEdge{1, 2}, TetrahedronEdge{2, 0},
    // pyramid with top at node 3.
    TetrahedronEdge{0, 3}, TetrahedronEdge{1, 3}, TetrahedronEdge{2, 3}};

/* Marching tetrahedra tables. Each entry in these tables have an index value
 based on a binary encoding of the signs of the plane's signed distance
 function evaluated at all tetrahedron vertices. Therefore, with four
 vertices and two possible signs, we have a total of 16 entries. We encode
 the table indexes in binary so that a "1" and "0" correspond to a vertex
 with positive or negative signed distance, respectively. The least
 significant bit (0) corresponds to vertex 0 in the tetrahedron, and the
 most significant bit (3) is vertex 3. */

/* Each entry of kMarchingTetsEdgeTable stores a vector of edges.
 Based on the signed distance values, these edges are the ones that
 intersect the plane. Edges are numbered according to the table kTetEdges.
 The edges have been ordered such that a polygon formed by visiting the
 listed edge's intersection vertices in the array order has a right-handed
 normal pointing in the direction of the plane's normal. The accompanying
 unit tests verify this.

 A -1 is a sentinel value indicating no edge encoding. The number of
 intersecting edges is equal to the index of the *first* -1 (with an implicit
 logical -1 at index 4).  */
// clang-format off
constexpr std::array<std::array<int, 4>, 16> kMarchingTetsEdgeTable = {
                                /* bits    3210 */
    std::array<int, 4>{-1, -1, -1, -1}, /* 0000 */
    std::array<int, 4>{0, 3, 2, -1},    /* 0001 */
    std::array<int, 4>{0, 1, 4, -1},    /* 0010 */
    std::array<int, 4>{4, 3, 2, 1},     /* 0011 */
    std::array<int, 4>{1, 2, 5, -1},    /* 0100 */
    std::array<int, 4>{0, 3, 5, 1},     /* 0101 */
    std::array<int, 4>{0, 2, 5, 4},     /* 0110 */
    std::array<int, 4>{3, 5, 4, -1},    /* 0111 */
    std::array<int, 4>{3, 4, 5, -1},    /* 1000 */
    std::array<int, 4>{4, 5, 2, 0},     /* 1001 */
    std::array<int, 4>{1, 5, 3, 0},     /* 1010 */
    std::array<int, 4>{1, 5, 2, -1},    /* 1011 */
    std::array<int, 4>{1, 2, 3, 4},     /* 1100 */
    std::array<int, 4>{0, 4, 1, -1},    /* 1101 */
    std::array<int, 4>{0, 2, 3, -1},    /* 1110 */
    std::array<int, 4>{-1, -1, -1, -1}  /* 1111 */};
// clang-format on

/* Constructs and stores polygon in slm and returns polygon size.
 * Slices a tetrahedron with an equilibrium plane and stores the resulting
 * polygon vertices in shared local memory.
 *
 */
SYCL_EXTERNAL inline void SliceTetWithEqPlane(
    sycl::nd_item<1> item, const sycl::local_accessor<double, 1>& slm,
    const uint32_t slm_offset,
    const sycl::local_accessor<double, 1>& slm_polygon,
    const uint32_t slm_polygon_offset,
    const sycl::local_accessor<int, 1>& slm_ints,
    const uint32_t slm_ints_offset, const uint32_t vertex_offset,
    const uint32_t eq_plane_offset, const uint32_t random_scratch_offset,
    const uint32_t polygon_offset, const uint32_t check_local_item_id,
    const uint32_t NUM_THREADS_PER_CHECK) {
  for (uint32_t llid = check_local_item_id; llid < 4;
       llid += NUM_THREADS_PER_CHECK) {
    // Each thread gets 1 vertex of element A in slm
    const double vertex_A_x = slm[slm_offset + vertex_offset + llid * 3 + 0];
    const double vertex_A_y = slm[slm_offset + vertex_offset + llid * 3 + 1];
    const double vertex_A_z = slm[slm_offset + vertex_offset + llid * 3 + 2];
    // Each thread accesses the same Eq plane from slm
    // TODO(huzaifa) - Will we have shMem bank conflict on Nvidia GPUs?
    // Need to know if SYCL backend compiler propertly recognizes that this is a
    // broadcast operation Normals
    const double normal_x = slm[slm_offset + eq_plane_offset];
    const double normal_y = slm[slm_offset + eq_plane_offset + 1];
    const double normal_z = slm[slm_offset + eq_plane_offset + 2];
    // Point on the plane
    const double point_on_plane_x = slm[slm_offset + eq_plane_offset + 3];
    const double point_on_plane_y = slm[slm_offset + eq_plane_offset + 4];
    const double point_on_plane_z = slm[slm_offset + eq_plane_offset + 5];
    // Compute the dispalcement of the plane from the origin of the frame (world
    // in this case) as simple dot product
    const double displacement = normal_x * point_on_plane_x +
                                normal_y * point_on_plane_y +
                                normal_z * point_on_plane_z;

    // Compute signed distance of this vertex with Eq plane
    // +ve height indicates point is above the plane
    // -ve height indicates point is below the plane
    // Store these in our random scratch space
    slm[slm_offset + random_scratch_offset + llid] =
        normal_x * vertex_A_x + normal_y * vertex_A_y + normal_z * vertex_A_z -
        displacement;
  }
  sycl::group_barrier(item.get_sub_group());

  // Let one thread compute intersection code and store this in the shared
  // memory for other threads
  if (check_local_item_id == 0) {
    int intersection_code = 0;
    for (uint32_t llid = 0; llid < 4; llid++) {
      if (slm[slm_offset + random_scratch_offset + llid] > 0) {
        intersection_code |= (1 << llid);
      }
    }
    slm_ints[slm_ints_offset] = intersection_code;
  }
  sycl::group_barrier(item.get_sub_group());

  // Now go back to using NUM_THREADS_PER_CHECK threads to compute the polygon
  // vertices
  for (uint32_t llid = check_local_item_id; llid < 4;
       llid += NUM_THREADS_PER_CHECK) {
    const int edge_index =
        kMarchingTetsEdgeTable[slm_ints[slm_ints_offset]][llid];
    // Only proceed if we are not at the end of edge list
    if (edge_index != -1) {
      // Get the tet edge
      const TetrahedronEdge& tet_edge = kTetEdges[edge_index];
      // Get the heights of these vertices from the scratch space
      const double height_0 =
          slm[slm_offset + random_scratch_offset + tet_edge.first];
      const double height_1 =
          slm[slm_offset + random_scratch_offset + tet_edge.second];

      // Compute the intersection point
      const double t = height_0 / (height_0 - height_1);

// Compute polygon vertices
// Loop is over x,y,z
#pragma unroll
      for (uint32_t i = 0; i < 3; i++) {
        const double vertex_0 =
            slm[slm_offset + vertex_offset + tet_edge.first * 3 + i];
        const double vertex_1 =
            slm[slm_offset + vertex_offset + tet_edge.second * 3 + i];

        const double intersection = vertex_0 + t * (vertex_1 - vertex_0);

        // Store the intersection point in the polygon
        slm_polygon[slm_polygon_offset + polygon_offset + llid * 3 + i] =
            intersection;
      }
    }
  }
  sycl::group_barrier(item.get_sub_group());

  // Compute current polygon size by checking number of max values
  int polygon_size = 0;
  if (check_local_item_id == 0) {
    for (uint32_t i = 0; i < 4; i++) {
      // TODO - Is just checking x enough? Should be I think
      if (slm_polygon[slm_polygon_offset + polygon_offset + i * 3 + 0] !=
          std::numeric_limits<double>::max()) {
        polygon_size++;
      }
    }
    slm_ints[slm_ints_offset] = polygon_size;
  }
}

// Same math as above but features the "no return" version where no threads are
// returned upon invalid geometry calcs (no surfaces, no areas etc)
SYCL_EXTERNAL inline void SliceTetWithEqPlaneNoReturn(
    sycl::nd_item<1> item, const sycl::local_accessor<double, 1>& slm,
    const uint32_t slm_offset,
    const sycl::local_accessor<double, 1>& slm_polygon,
    const uint32_t slm_polygon_offset,
    const sycl::local_accessor<int, 1>& slm_ints,
    const uint32_t slm_ints_offset, const uint32_t vertex_offset,
    const uint32_t eq_plane_offset, const uint32_t random_scratch_offset,
    const uint32_t polygon_offset, const uint32_t check_local_item_id,
    const uint32_t NUM_THREADS_PER_CHECK, bool valid_thread) {
  if (valid_thread) {
    for (uint32_t llid = check_local_item_id; llid < 4;
         llid += NUM_THREADS_PER_CHECK) {
      // Each thread gets 1 vertex of element A in slm
      const double vertex_A_x = slm[slm_offset + vertex_offset + llid * 3 + 0];
      const double vertex_A_y = slm[slm_offset + vertex_offset + llid * 3 + 1];
      const double vertex_A_z = slm[slm_offset + vertex_offset + llid * 3 + 2];
      // Each thread accesses the same Eq plane from slm
      // TODO(huzaifa) - Will we have shMem bank conflict on Nvidia GPUs?
      // Need to know if SYCL backend compiler propertly recognizes that this is
      // a broadcast operation Normals
      const double normal_x = slm[slm_offset + eq_plane_offset];
      const double normal_y = slm[slm_offset + eq_plane_offset + 1];
      const double normal_z = slm[slm_offset + eq_plane_offset + 2];
      // Point on the plane
      const double point_on_plane_x = slm[slm_offset + eq_plane_offset + 3];
      const double point_on_plane_y = slm[slm_offset + eq_plane_offset + 4];
      const double point_on_plane_z = slm[slm_offset + eq_plane_offset + 5];
      // Compute the dispalcement of the plane from the origin of the frame
      // (world in this case) as simple dot product
      const double displacement = normal_x * point_on_plane_x +
                                  normal_y * point_on_plane_y +
                                  normal_z * point_on_plane_z;

      // Compute signed distance of this vertex with Eq plane
      // +ve height indicates point is above the plane
      // -ve height indicates point is below the plane
      // Store these in our random scratch space
      slm[slm_offset + random_scratch_offset + llid] =
          normal_x * vertex_A_x + normal_y * vertex_A_y +
          normal_z * vertex_A_z - displacement;
    }
  }
  sycl::group_barrier(item.get_group());

  // Let one thread compute intersection code and store this in the shared
  // memory for other threads
  if (check_local_item_id == 0 && valid_thread) {
    int intersection_code = 0;
    for (uint32_t llid = 0; llid < 4; llid++) {
      if (slm[slm_offset + random_scratch_offset + llid] > 0) {
        intersection_code |= (1 << llid);
      }
    }
    slm_ints[slm_ints_offset] = intersection_code;
  }
  sycl::group_barrier(item.get_group());

  // Now go back to using NUM_THREADS_PER_CHECK threads to compute the polygon
  // vertices
  if (valid_thread) {
    for (uint32_t llid = check_local_item_id; llid < 4;
         llid += NUM_THREADS_PER_CHECK) {
      const int edge_index =
          kMarchingTetsEdgeTable[slm_ints[slm_ints_offset]][llid];
      // Only proceed if we are not at the end of edge list
      if (edge_index != -1) {
        // Get the tet edge
        const TetrahedronEdge& tet_edge = kTetEdges[edge_index];
        // Get the heights of these vertices from the scratch space
        const double height_0 =
            slm[slm_offset + random_scratch_offset + tet_edge.first];
        const double height_1 =
            slm[slm_offset + random_scratch_offset + tet_edge.second];

        // Compute the intersection point
        const double t = height_0 / (height_0 - height_1);

// Compute polygon vertices
// Loop is over x,y,z
#pragma unroll
        for (uint32_t i = 0; i < 3; i++) {
          const double vertex_0 =
              slm[slm_offset + vertex_offset + tet_edge.first * 3 + i];
          const double vertex_1 =
              slm[slm_offset + vertex_offset + tet_edge.second * 3 + i];

          const double intersection = vertex_0 + t * (vertex_1 - vertex_0);

          // Store the intersection point in the polygon
          slm_polygon[slm_polygon_offset + polygon_offset + llid * 3 + i] =
              intersection;
        }
      }
    }
  }
  sycl::group_barrier(item.get_group());

  if (check_local_item_id == 0 && valid_thread) {
    // Compute current polygon size by checking number of max values
    int polygon_size = 0;
    for (uint32_t i = 0; i < 4; i++) {
      // TODO - Is just checking x enough? Should be I think
      if (slm_polygon[slm_polygon_offset + polygon_offset + i * 3 + 0] !=
          std::numeric_limits<double>::max()) {
        polygon_size++;
      }
    }
    slm_ints[slm_ints_offset] = polygon_size;
  }
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake