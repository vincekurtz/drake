#pragma once

#include <array>
#include <limits>
#include <vector>

#include "geometry/proximity/sycl/utils/sycl_device_types.h"
#include "geometry/proximity/sycl/utils/sycl_equilibrium_plane.h"
#include "geometry/proximity/sycl/utils/sycl_tetrahedron_slicing.h"
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Forward declaration for kernel name template
template <DeviceType device_type>
class ComputeContactPolygonsKernel;

/* Computes contact polygons for narrow phase collision detection.
 *
 * This function performs the entire narrow phase collision detection pipeline:
 * 1. Computes equilibrium planes between tetrahedra pairs
 * 2. Slices tetrahedra with equilibrium planes
 * 3. Clips resulting polygons against tetrahedra faces
 * 4. Computes polygon areas, centroids, and other contact properties
 * This kernel function is called "NoReturn" because it is made so as the
 * threads do not return even for invalid checks. This is becasue the kernel has
 * barriers that need to be reached by ALL threads in the work group.
 *
 */
SYCL_EXTERNAL inline void ComputeContactPolygonsNoReturn(
    sycl::nd_item<1> item, const sycl::local_accessor<double, 1>& slm,
    const sycl::local_accessor<double, 1>& slm_polygon,
    const sycl::local_accessor<int, 1>& slm_ints,
    const uint32_t TOTAL_THREADS_NEEDED, const uint32_t NUM_THREADS_PER_CHECK,
    const uint32_t DOUBLES_PER_CHECK, const uint32_t POLYGON_DOUBLES,
    const uint32_t EQ_PLANE_OFFSET, const uint32_t VERTEX_A_OFFSET,
    const uint32_t VERTEX_B_OFFSET, const uint32_t RANDOM_SCRATCH_OFFSET,
    const uint32_t POLYGON_VERTICES,
    const Vector4<double>* gradient_W_pressure_at_Wo,
    const uint32_t* vertex_offsets, const uint32_t* element_mesh_ids,
    const std::array<int, 4>* elements, const Vector3<double>* vertices_W,
    const std::array<Vector3<double>, 4>* inward_normals_W,
    const uint32_t* collision_indices_A, const uint32_t* collision_indices_B,
    uint8_t* narrow_phase_check_validity, double* polygon_areas,
    Vector3<double>* polygon_centroids, Vector3<double>* polygon_normals,
    double* polygon_g_M, double* polygon_g_N, double* polygon_pressure_W,
    GeometryId* polygon_geom_index_A, GeometryId* polygon_geom_index_B,
    const GeometryId* geometry_ids) {
  uint32_t global_id = item.get_global_id(0);
  bool valid_thread = true;
  sycl::sub_group sg = item.get_sub_group();
  // Early return for extra threads
  if (global_id >= TOTAL_THREADS_NEEDED) {
    valid_thread = false;
  }
  uint32_t local_id = item.get_local_id(0);
  // In a group we have NUM_CHECKS_IN_WORK_GROUP checks
  // This gives us which check number in [0, NUM_CHECKS_IN_WORK_GROUP)
  // this item is working on
  // NUM_THREADS_PER_CHECK threads will have the same
  // group_check_number
  // It ranges from [0, NUM_CHECKS_IN_WORK_GROUP)
  uint32_t group_local_check_number = local_id / NUM_THREADS_PER_CHECK;

  // This offset is used to compute the positions each of the
  // quantities for reading and writing to slm
  uint32_t slm_offset = group_local_check_number * DOUBLES_PER_CHECK;

  // This offset is used to compute the positions each of the
  // quantities for reading and writing to slm_polygon
  uint32_t slm_polygon_offset = group_local_check_number * POLYGON_DOUBLES;

  // Check offset for slm_ints array
  constexpr uint32_t RANDOM_SCRATCH_INTS = 1;
  uint32_t slm_ints_offset = group_local_check_number * RANDOM_SCRATCH_INTS;

  // Each check has NUM_THREADS_PER_CHECK workers.
  // This index helps identify the check local worker id
  // It ranges for [0, NUM_THREADS_PER_CHECK)
  uint32_t check_local_item_id = local_id % NUM_THREADS_PER_CHECK;

  // Need to make sure that invalid threads never participate in
  // any of the computations
  uint32_t narrow_phase_check_index = std::numeric_limits<uint32_t>::max();
  uint32_t A_element_index = std::numeric_limits<uint32_t>::max();
  uint32_t B_element_index = std::numeric_limits<uint32_t>::max();

  // Get global element ids
  if (valid_thread) {
    narrow_phase_check_index = global_id / NUM_THREADS_PER_CHECK;

    A_element_index = collision_indices_A[narrow_phase_check_index];
    B_element_index = collision_indices_B[narrow_phase_check_index];
  }

  // We only need one thread to compute the Equilibrium Plane
  // for each check, however we have potentially multiple threads
  // per check.
  // We cannot use the first "total_narrow_phase_check" items since we
  // need to store the equilibirum planes in shared memory which is
  // work group local Thus, make sure only 1 thread in the check group
  // computes the equilibrium plane and we choose this to be the 1st
  // thread
  if (check_local_item_id == 0 && valid_thread) {
    // Get individual quanities for quick access from registers
    const double gradP_A_Wo_x = gradient_W_pressure_at_Wo[A_element_index][0];
    const double gradP_A_Wo_y = gradient_W_pressure_at_Wo[A_element_index][1];
    const double gradP_A_Wo_z = gradient_W_pressure_at_Wo[A_element_index][2];
    const double p_A_Wo = gradient_W_pressure_at_Wo[A_element_index][3];
    const double gradP_B_Wo_x = gradient_W_pressure_at_Wo[B_element_index][0];
    const double gradP_B_Wo_y = gradient_W_pressure_at_Wo[B_element_index][1];
    const double gradP_B_Wo_z = gradient_W_pressure_at_Wo[B_element_index][2];
    const double p_B_Wo = gradient_W_pressure_at_Wo[B_element_index][3];

    constexpr uint32_t EQ_PLANE_DOUBLES = 8;
    double eq_plane[EQ_PLANE_DOUBLES];
    bool valid_check = ComputeEquilibriumPlane(
        gradP_A_Wo_x, gradP_A_Wo_y, gradP_A_Wo_z, p_A_Wo, gradP_B_Wo_x,
        gradP_B_Wo_y, gradP_B_Wo_z, p_B_Wo, eq_plane);
    if (valid_check) {
// Write for Eq plane to slm
#pragma unroll
      for (int i = 0; i < EQ_PLANE_DOUBLES; ++i) {
        slm[slm_offset + EQ_PLANE_OFFSET + i] = eq_plane[i];
      }
    } else {
      narrow_phase_check_validity[narrow_phase_check_index] = 0;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Return all invalid checks
  if (valid_thread) {
    if (narrow_phase_check_validity[narrow_phase_check_index] == 0) {
      valid_thread = false;
    }
  }

  // Initialize the current polygon offset inside since we need to
  // switch them around later
  constexpr uint32_t POLYGON_CURRENT_DOUBLES = 48;
  uint32_t POLYGON_CURRENT_OFFSET = 0;
  uint32_t POLYGON_CLIPPED_OFFSET =
      POLYGON_CURRENT_OFFSET + POLYGON_CURRENT_DOUBLES;

  // Move vertices and edge vectors to slm
  // Some quantities required for indexing
  uint32_t geom_index_A = std::numeric_limits<uint32_t>::max();
  uint32_t geom_index_B = std::numeric_limits<uint32_t>::max();

  if (valid_thread) {
    geom_index_A = element_mesh_ids[A_element_index];
    const std::array<int, 4> tet_vertices_A = elements[A_element_index];
    const uint32_t vertex_mesh_offset_A = vertex_offsets[geom_index_A];

    // Vertices of element B
    geom_index_B = element_mesh_ids[B_element_index];
    const std::array<int, 4> tet_vertices_B = elements[B_element_index];
    const uint32_t vertex_mesh_offset_B = vertex_offsets[geom_index_B];

    // Loop is over x,y,z

#pragma unroll
    for (uint32_t i = 0; i < 3; i++) {
      // Quantities that we have "4" of
      for (uint32_t llid = check_local_item_id; llid < 4;
           llid += NUM_THREADS_PER_CHECK) {
        // All 4 vertices moved at once by our sub items
        slm[slm_offset + VERTEX_A_OFFSET + llid * 3 + i] =
            vertices_W[vertex_mesh_offset_A + tet_vertices_A[llid]][i];
        slm[slm_offset + VERTEX_B_OFFSET + llid * 3 + i] =
            vertices_W[vertex_mesh_offset_B + tet_vertices_B[llid]][i];
      }
    }
    // Quantity that we have "16" of - Only set 0'th element
    for (uint32_t llid = check_local_item_id; llid < POLYGON_VERTICES;
         llid += NUM_THREADS_PER_CHECK) {
      slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + llid * 3] =
          std::numeric_limits<double>::max();
      slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET + llid * 3] =
          std::numeric_limits<double>::max();
    }
  }
  sycl::group_barrier(item.get_group());

  // =====================================
  // Intersect element A with Eq Plane
  // =====================================

  // Compute signed distance of all vertices of element A with Eq
  // plane Parallelization based on distance computation

  SliceTetWithEqPlaneNoReturn(
      item, slm, slm_offset, slm_polygon, slm_polygon_offset, slm_ints,
      slm_ints_offset, VERTEX_A_OFFSET, EQ_PLANE_OFFSET, RANDOM_SCRATCH_OFFSET,
      POLYGON_CURRENT_OFFSET, check_local_item_id, NUM_THREADS_PER_CHECK,
      valid_thread);

  if (check_local_item_id == 0 && valid_thread) {
    if (slm_ints[slm_ints_offset] < 3) {
      narrow_phase_check_validity[narrow_phase_check_index] = 0;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
  // Return all invalid checks
  if (valid_thread) {
    if (narrow_phase_check_validity[narrow_phase_check_index] == 0) {
      valid_thread = false;
    }
  }
  // Compute the intersection of Polygon Q with the faces of element
  // B We will sequentially loop over the faces but we will use our
  // work items to parallely compute the intersection point over
  // each edge We have 4 faces, so we will have 4 jobs per check
  for (uint32_t face = 0; face < 4; face++) {
    uint32_t num_edges_current_polygon = 0;
    if (valid_thread) {
      // This is the same as the number of points in the polygon
      num_edges_current_polygon = slm_ints[slm_ints_offset];
    }

    // First lets find the height of each of these vertices from the
    // face of interest
    for (uint32_t job = check_local_item_id; job < num_edges_current_polygon;
         job += NUM_THREADS_PER_CHECK) {
      // Get the outward normal of the face, point on face, and
      // polygon vertex
      double outward_normal[3];
      double point_on_face[3];
      double polygon_vertex_coords[3];

      // 'face' corresponds to the triangle formed by {0, 1, 2, 3} -
      // {face} so any of (face+1)%4, (face+2)%4, (face+3)%4 are
      // candidates for a point on the face's plane. We arbitrarily
      // choose (face + 1) % 4.
      const uint32_t face_vertex_index = (face + 1) % 4;
// This loop is over x,y,z
#pragma unroll
      for (uint32_t i = 0; i < 3; i++) {
        outward_normal[i] = -inward_normals_W[B_element_index][face][i];

        // Get a point from the verticies of element B
        point_on_face[i] =
            slm[slm_offset + VERTEX_B_OFFSET + face_vertex_index * 3 + i];

        // Get the polygon vertex -> This has to be from
        // POLYGON_CURRENT_OFFSET
        polygon_vertex_coords[i] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + job * 3 +
                        i];
      }
      // We will store our heights in the random scratch space that
      // we have (we have upto 16 doubles space) They will be
      // ordered in row major order [face_0_vertex_0,
      // face_0_vertex_1, face_0_vertex_2, face_0_vertex_3,
      // face_1_vertex_0, ...]
      const double displacement = outward_normal[0] * point_on_face[0] +
                                  outward_normal[1] * point_on_face[1] +
                                  outward_normal[2] * point_on_face[2];

      // <= 0 is inside, > 0 is outside
      slm[slm_offset + RANDOM_SCRATCH_OFFSET + job] =
          outward_normal[0] * polygon_vertex_coords[0] +
          outward_normal[1] * polygon_vertex_coords[1] +
          outward_normal[2] * polygon_vertex_coords[2] - displacement;
    }
    // Sync shared memory
    item.barrier(sycl::access::fence_space::local_space);

    // Now we will walk the current polygon and construct the
    // clipped polygon
    for (uint32_t vertex_0_index = check_local_item_id;
         vertex_0_index < num_edges_current_polygon;
         vertex_0_index += NUM_THREADS_PER_CHECK) {
      // Get the height of vertex_1
      const uint32_t vertex_1_index =
          (vertex_0_index + 1) % num_edges_current_polygon;

      // Get the height of vertex_0
      double height_0 =
          slm[slm_offset + RANDOM_SCRATCH_OFFSET + vertex_0_index];
      double height_1 =
          slm[slm_offset + RANDOM_SCRATCH_OFFSET + vertex_1_index];

      // Each edge can store upto two vertices
      // We will do a compaction in the end
      if (height_0 <= 0) {
        // If vertex_0 is inside, it is part of the clipped polygon

        // Copy vertex_0 into the clipped polygon
        // The "2" multiplier is because each edge can contribute in
        // one loop upto 2 vertices. The "3" is because of xyz
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 0] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        vertex_0_index * 3 + 0];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 1] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        vertex_0_index * 3 + 1];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 2] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        vertex_0_index * 3 + 2];

        // Now if vertex_1 is outside, we will have an intersection
        // point too
        if (height_1 > 0) {
          // Compute the intersection point
          const double wa = height_1 / (height_1 - height_0);
          const double wb = 1 - wa;

          // Copy the intersection point into the clipped polygon
          slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                      (2 * vertex_0_index + 1) * 3 + 0] =
              wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_0_index * 3 + 0] +
              wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_1_index * 3 + 0];
          slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                      (2 * vertex_0_index + 1) * 3 + 1] =
              wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_0_index * 3 + 1] +
              wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_1_index * 3 + 1];
          slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                      (2 * vertex_0_index + 1) * 3 + 2] =
              wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_0_index * 3 + 2] +
              wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_1_index * 3 + 2];
        }
      } else if (height_1 <= 0) {
        // If vertex_1 is inside and vertex_0 is outside this edge
        // will contribute 1 point (intersection point)
        const double wa = height_1 / (height_1 - height_0);
        const double wb = 1 - wa;

        // Copy the intersection point into the clipped polygon
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 0] =
            wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_0_index * 3 + 0] +
            wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_1_index * 3 + 0];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 1] =
            wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_0_index * 3 + 1] +
            wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_1_index * 3 + 1];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 2] =
            wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_0_index * 3 + 2] +
            wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_1_index * 3 + 2];
      }
    }
    sycl::group_barrier(item.get_group());

    // Flip Current and clipped polygon
    const uint32_t temp_polygon_current_offset = POLYGON_CURRENT_OFFSET;
    POLYGON_CURRENT_OFFSET = POLYGON_CLIPPED_OFFSET;
    POLYGON_CLIPPED_OFFSET = temp_polygon_current_offset;

    // Now clean up the current polygon to remove out the
    // std::numeric_limits<double>::max() vertices
    if (check_local_item_id == 0 && valid_thread) {
      uint32_t write_index = 0;
      // Scan through all potential vertices
      for (uint32_t read_index = 0; read_index < POLYGON_VERTICES;
           ++read_index) {
        // Check if this vertex is valid (not max value)
        if (slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        read_index * 3 + 0] !=
            std::numeric_limits<double>::max()) {
          // Only copy if read and write indices are different
          if (read_index != write_index) {
            // Copy the valid vertex to the write position
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        write_index * 3 + 0] =
                slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                            read_index * 3 + 0];
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        write_index * 3 + 1] =
                slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                            read_index * 3 + 1];
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        write_index * 3 + 2] =
                slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                            read_index * 3 + 2];
          }
          write_index++;
        }
      }

      // Fill remaining positions with max values to mark them as
      // invalid At the same time even fill in the clipped polygon
      // with max values
      for (uint32_t i = write_index; i < POLYGON_VERTICES; ++i) {
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + i * 3] =
            std::numeric_limits<double>::max();
      }

      // Update polygon size
      slm_ints[slm_ints_offset] = write_index;
    }
    sycl::group_barrier(item.get_group());

    // Clear out the clipped polygon
    if (valid_thread) {
      for (uint32_t llid = check_local_item_id; llid < POLYGON_VERTICES;
           llid += NUM_THREADS_PER_CHECK) {
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET + llid * 3] =
            std::numeric_limits<double>::max();
      }
    }
    sycl::group_barrier(item.get_group());
  }

  if (check_local_item_id == 0 && valid_thread) {
    if (slm_ints[slm_ints_offset] < 3) {
      narrow_phase_check_validity[narrow_phase_check_index] = 0;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (valid_thread) {
    if (narrow_phase_check_validity[narrow_phase_check_index] == 0) {
      valid_thread = false;
    }
  }

  // Now we compute the area and the centroid of the polygons
  // Compute mean vertex of the polygon using a reduce
  // We will use the clipped polygon shared memory area for all
  // these intermediate results

  // We use one of the polygon's vertices as our base point to cut
  // the polygon into triangles We will use the first point for this
  uint32_t polygon_size = 0;
  if (valid_thread) {
    polygon_size = slm_ints[slm_ints_offset];
  }
  const uint32_t AREAS_OFFSET = POLYGON_CLIPPED_OFFSET;
  const uint32_t CENTROID_OFFSET = VERTEX_A_OFFSET;
  double thread_area_sum = 0;
  double thread_centroid_x = 0;
  double thread_centroid_y = 0;
  double thread_centroid_z = 0;
  for (uint32_t triangle_index = check_local_item_id;
       triangle_index + 2 < polygon_size;
       triangle_index += NUM_THREADS_PER_CHECK) {
    const double v0_x =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + 0 * 3 + 0];
    const double v0_y =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + 0 * 3 + 1];

    // Compute the thread local cross magnitude

    // First vertex of triangle edge (current polygon vertex)
    const double v1_x =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 1) * 3 + 0];
    const double v1_y =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 1) * 3 + 1];
    // Second vertex of triangle edge (next polygon vertex, wrapping
    // around)
    const double v2_x =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 2) * 3 + 0];
    const double v2_y =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 2) * 3 + 1];

    double cross_magnitude = (v1_x - v0_x) * (v2_y - v0_y);
    cross_magnitude -= (v1_y - v0_y) * (v2_x - v0_x);
    cross_magnitude *= cross_magnitude;

    const double v0_z =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + 0 * 3 + 2];
    const double v1_z =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 1) * 3 + 2];
    const double v2_z =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 2) * 3 + 2];

    double temp = (v1_y - v0_y) * (v2_z - v0_z);
    temp -= (v1_z - v0_z) * (v2_y - v0_y);
    cross_magnitude += temp * temp;

    temp = (v1_z - v0_z) * (v2_x - v0_x);
    temp -= (v1_x - v0_x) * (v2_z - v0_z);
    cross_magnitude += temp * temp;

    cross_magnitude = sycl::sqrt(cross_magnitude);
    thread_area_sum += cross_magnitude;

    // Compute the thread local centroid
    thread_centroid_x += cross_magnitude * v0_x;
    thread_centroid_y += cross_magnitude * v0_y;
    thread_centroid_z += cross_magnitude * v0_z;
    thread_centroid_x += cross_magnitude * v1_x;
    thread_centroid_y += cross_magnitude * v1_y;
    thread_centroid_z += cross_magnitude * v1_z;
    thread_centroid_x += cross_magnitude * v2_x;
    thread_centroid_y += cross_magnitude * v2_y;
    thread_centroid_z += cross_magnitude * v2_z;
  }

  // Now each thread writes its computed values
  if (check_local_item_id + 2 < polygon_size) {
    slm_polygon[slm_polygon_offset + AREAS_OFFSET + check_local_item_id] =
        thread_area_sum;
    slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 0] =
        thread_centroid_x;
    slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 1] =
        thread_centroid_y;
    slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 2] =
        thread_centroid_z;
  }

  item.barrier(sycl::access::fence_space::local_space);

  for (uint32_t stride = NUM_THREADS_PER_CHECK / 2; stride > 0; stride >>= 1) {
    if (check_local_item_id < stride &&
        check_local_item_id + stride + 2 < polygon_size) {
      slm_polygon[slm_polygon_offset + AREAS_OFFSET + check_local_item_id] +=
          slm_polygon[slm_polygon_offset + AREAS_OFFSET +
                      (check_local_item_id + stride)];
      slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 0] +=
          slm[slm_offset + CENTROID_OFFSET +
              (check_local_item_id + stride) * 3 + 0];
      slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 1] +=
          slm[slm_offset + CENTROID_OFFSET +
              (check_local_item_id + stride) * 3 + 1];
      slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 2] +=
          slm[slm_offset + CENTROID_OFFSET +
              (check_local_item_id + stride) * 3 + 2];
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Now write everything to global memory
  if (check_local_item_id == 0 && valid_thread) {
    // Write Polygon Area and Centroid
    const double polygon_area =
        slm_polygon[slm_polygon_offset + AREAS_OFFSET + 0] * 0.5;
    if (polygon_area > 1e-18) {
      polygon_areas[narrow_phase_check_index] = polygon_area;
      const double inv_polygon_area_6 = 1.0 / (polygon_area * 6);
      const double centroid_x =
          slm[slm_offset + CENTROID_OFFSET + 0 * 3 + 0] * inv_polygon_area_6;
      const double centroid_y =
          slm[slm_offset + CENTROID_OFFSET + 0 * 3 + 1] * inv_polygon_area_6;
      const double centroid_z =
          slm[slm_offset + CENTROID_OFFSET + 0 * 3 + 2] * inv_polygon_area_6;
      polygon_centroids[narrow_phase_check_index][0] = centroid_x;
      polygon_centroids[narrow_phase_check_index][1] = centroid_y;
      polygon_centroids[narrow_phase_check_index][2] = centroid_z;

      // Write Polygon Normal -> This is already normalized
      polygon_normals[narrow_phase_check_index][0] =
          slm[slm_offset + EQ_PLANE_OFFSET];
      polygon_normals[narrow_phase_check_index][1] =
          slm[slm_offset + EQ_PLANE_OFFSET + 1];
      polygon_normals[narrow_phase_check_index][2] =
          slm[slm_offset + EQ_PLANE_OFFSET + 2];

      // Write Polygon g_M_
      polygon_g_M[narrow_phase_check_index] =
          slm[slm_offset + EQ_PLANE_OFFSET + 6];
      // Write Polygon g_N_
      polygon_g_N[narrow_phase_check_index] =
          slm[slm_offset + EQ_PLANE_OFFSET + 7];

      // Compute the pressure at the centroid
      // TODO(Huzaifa): Check this for correctness
      const double gradP_A_Wo_x = gradient_W_pressure_at_Wo[A_element_index][0];
      const double gradP_A_Wo_y = gradient_W_pressure_at_Wo[A_element_index][1];
      const double gradP_A_Wo_z = gradient_W_pressure_at_Wo[A_element_index][2];
      const double p_A_Wo = gradient_W_pressure_at_Wo[A_element_index][3];
      polygon_pressure_W[narrow_phase_check_index] =
          gradP_A_Wo_x * centroid_x + gradP_A_Wo_y * centroid_y +
          gradP_A_Wo_z * centroid_z + p_A_Wo;

      // Write Geometry Index A
      const uint32_t geom_index_A = element_mesh_ids[A_element_index];
      const uint32_t geom_index_B = element_mesh_ids[B_element_index];
      polygon_geom_index_A[narrow_phase_check_index] =
          geometry_ids[geom_index_A];
      // Write Geometry Index B
      polygon_geom_index_B[narrow_phase_check_index] =
          geometry_ids[geom_index_B];
    } else {
      narrow_phase_check_validity[narrow_phase_check_index] = 0;
    }
  }
}

/*
 * This kernel function is the same as above but it returns for invalid checks.
 * We found that on the GPU this seems to work fine but not on the CPU. Keeping
 * it around for now since its simpler with lesser branching.
 */
SYCL_EXTERNAL inline void ComputeContactPolygons(
    sycl::nd_item<1> item, const sycl::local_accessor<double, 1>& slm,
    const sycl::local_accessor<double, 1>& slm_polygon,
    const sycl::local_accessor<int, 1>& slm_ints,
    const uint32_t TOTAL_THREADS_NEEDED, const uint32_t NUM_THREADS_PER_CHECK,
    const uint32_t DOUBLES_PER_CHECK, const uint32_t POLYGON_DOUBLES,
    const uint32_t EQ_PLANE_OFFSET, const uint32_t VERTEX_A_OFFSET,
    const uint32_t VERTEX_B_OFFSET, const uint32_t RANDOM_SCRATCH_OFFSET,
    const uint32_t POLYGON_VERTICES,
    const Vector4<double>* gradient_W_pressure_at_Wo,
    const uint32_t* element_offsets, const uint32_t* vertex_offsets,
    const uint32_t* element_mesh_ids, const std::array<int, 4>* elements,
    const Vector3<double>* vertices_W,
    const std::array<Vector3<double>, 4>* inward_normals_W,
    const uint32_t* collision_indices_A, const uint32_t* collision_indices_B,
    uint8_t* narrow_phase_check_validity, double* polygon_areas,
    Vector3<double>* polygon_centroids, Vector3<double>* polygon_normals,
    double* polygon_g_M, double* polygon_g_N, double* polygon_pressure_W,
    GeometryId* polygon_geom_index_A, GeometryId* polygon_geom_index_B,
    const GeometryId* geometry_ids) {
  uint32_t global_id = item.get_global_id(0);
  // Early return for extra threads
  if (global_id >= TOTAL_THREADS_NEEDED) return;
  uint32_t local_id = item.get_local_id(0);
  auto sub_group = item.get_sub_group();
  // In a group we have NUM_CHECKS_IN_WORK_GROUP checks
  // This gives us which check number in [0, NUM_CHECKS_IN_WORK_GROUP)
  // this item is working on
  // NUM_THREADS_PER_CHECK threads will have the same
  // group_check_number
  // It ranges from [0, NUM_CHECKS_IN_WORK_GROUP)
  uint32_t group_local_check_number = local_id / NUM_THREADS_PER_CHECK;

  // This offset is used to compute the positions each of the
  // quantities for reading and writing to slm
  uint32_t slm_offset = group_local_check_number * DOUBLES_PER_CHECK;

  // This offset is used to compute the positions each of the
  // quantities for reading and writing to slm_polygon
  uint32_t slm_polygon_offset = group_local_check_number * POLYGON_DOUBLES;

  // Check offset for slm_ints array
  constexpr uint32_t RANDOM_SCRATCH_INTS = 1;
  uint32_t slm_ints_offset = group_local_check_number * RANDOM_SCRATCH_INTS;

  // Each check has NUM_THREADS_PER_CHECK workers.
  // This index helps identify the check local worker id
  // It ranges for [0, NUM_THREADS_PER_CHECK)
  uint32_t check_local_item_id = local_id % NUM_THREADS_PER_CHECK;

  // Get global element ids
  uint32_t narrow_phase_check_index = global_id / NUM_THREADS_PER_CHECK;

  const uint32_t A_element_index =
      collision_indices_A[narrow_phase_check_index];
  const uint32_t B_element_index =
      collision_indices_B[narrow_phase_check_index];

  // We only need one thread to compute the Equilibrium Plane
  // for each check, however we have potentially multiple threads
  // per check.
  // We cannot use the first "total_narrow_phase_check" items since we
  // need to store the equilibirum planes in shared memory which is
  // work group local Thus, make sure only 1 thread in the check group
  // computes the equilibrium plane and we choose this to be the 1st
  // thread
  if (check_local_item_id == 0) {
    // Get individual quanities for quick access from registers
    const double gradP_A_Wo_x = gradient_W_pressure_at_Wo[A_element_index][0];
    const double gradP_A_Wo_y = gradient_W_pressure_at_Wo[A_element_index][1];
    const double gradP_A_Wo_z = gradient_W_pressure_at_Wo[A_element_index][2];
    const double p_A_Wo = gradient_W_pressure_at_Wo[A_element_index][3];
    const double gradP_B_Wo_x = gradient_W_pressure_at_Wo[B_element_index][0];
    const double gradP_B_Wo_y = gradient_W_pressure_at_Wo[B_element_index][1];
    const double gradP_B_Wo_z = gradient_W_pressure_at_Wo[B_element_index][2];
    const double p_B_Wo = gradient_W_pressure_at_Wo[B_element_index][3];

    constexpr uint32_t EQ_PLANE_DOUBLES = 8;
    double eq_plane[EQ_PLANE_DOUBLES];
    bool valid_check = ComputeEquilibriumPlane(
        gradP_A_Wo_x, gradP_A_Wo_y, gradP_A_Wo_z, p_A_Wo, gradP_B_Wo_x,
        gradP_B_Wo_y, gradP_B_Wo_z, p_B_Wo, eq_plane);
    if (valid_check) {
// Write for Eq plane to slm
#pragma unroll
      for (int i = 0; i < EQ_PLANE_DOUBLES; ++i) {
        slm[slm_offset + EQ_PLANE_OFFSET + i] = eq_plane[i];
      }
    } else {
      narrow_phase_check_validity[narrow_phase_check_index] = 0;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Return all invalid checks
  if (narrow_phase_check_validity[narrow_phase_check_index] == 0) {
    return;
  }

  // Initialize the current polygon offset inside since we need to
  // switch them around later
  constexpr uint32_t POLYGON_CURRENT_DOUBLES = 48;
  uint32_t POLYGON_CURRENT_OFFSET = 0;
  uint32_t POLYGON_CLIPPED_OFFSET =
      POLYGON_CURRENT_OFFSET + POLYGON_CURRENT_DOUBLES;

  // Move vertices and edge vectors to slm
  // Some quantities required for indexing
  const uint32_t geom_index_A = element_mesh_ids[A_element_index];
  const std::array<int, 4>& tet_vertices_A = elements[A_element_index];
  const uint32_t vertex_mesh_offset_A = vertex_offsets[geom_index_A];

  // Vertices of element B
  const uint32_t geom_index_B = element_mesh_ids[B_element_index];
  const std::array<int, 4>& tet_vertices_B = elements[B_element_index];
  const uint32_t vertex_mesh_offset_B = vertex_offsets[geom_index_B];

// Loop is over x,y,z
#pragma unroll
  for (uint32_t i = 0; i < 3; i++) {
    // Quantities that we have "4" of
    for (uint32_t llid = check_local_item_id; llid < 4;
         llid += NUM_THREADS_PER_CHECK) {
      // All 4 vertices moved at once by our sub items
      slm[slm_offset + VERTEX_A_OFFSET + llid * 3 + i] =
          vertices_W[vertex_mesh_offset_A + tet_vertices_A[llid]][i];
      slm[slm_offset + VERTEX_B_OFFSET + llid * 3 + i] =
          vertices_W[vertex_mesh_offset_B + tet_vertices_B[llid]][i];
    }
    // Quantity that we have "16" of - For now set all the verticies
    // of the polygon to double max so that we know all are stale
    for (uint32_t llid = check_local_item_id; llid < POLYGON_VERTICES;
         llid += NUM_THREADS_PER_CHECK) {
      slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + llid * 3 + i] =
          std::numeric_limits<double>::max();
      slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET + llid * 3 + i] =
          std::numeric_limits<double>::max();
    }
  }
  sycl::group_barrier(sub_group);

  // =====================================
  // Intersect element A with Eq Plane
  // =====================================

  // Compute signed distance of all vertices of element A with Eq
  // plane Parallelization based on distance computation

  SliceTetWithEqPlane(
      item, slm, slm_offset, slm_polygon, slm_polygon_offset, slm_ints,
      slm_ints_offset, VERTEX_A_OFFSET, EQ_PLANE_OFFSET, RANDOM_SCRATCH_OFFSET,
      POLYGON_CURRENT_OFFSET, check_local_item_id, NUM_THREADS_PER_CHECK);

  if (check_local_item_id == 0 && slm_ints[slm_ints_offset] < 3) {
    narrow_phase_check_validity[narrow_phase_check_index] = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);
  // Return all invalid checks
  if (narrow_phase_check_validity[narrow_phase_check_index] == 0) {
    return;
  }

  // Compute the intersection of Polygon Q with the faces of element B
  // We will sequentially loop over the faces but we will use our work
  // items to parallely compute the intersection point over each edge
  // We have 4 faces, so we will have 4 jobs per check
  for (uint32_t face = 0; face < 4; face++) {
    // This is the same as the number of points in the polygon
    const uint32_t num_edges_current_polygon = slm_ints[slm_ints_offset];

    // First lets find the height of each of these vertices from the
    // face of interest
    for (uint32_t job = check_local_item_id; job < num_edges_current_polygon;
         job += NUM_THREADS_PER_CHECK) {
      // Get the outward normal of the face, point on face, and
      // polygon vertex
      double outward_normal[3];
      double point_on_face[3];
      double polygon_vertex_coords[3];

      // 'face' corresponds to the triangle formed by {0, 1, 2, 3} -
      // {face} so any of (face+1)%4, (face+2)%4, (face+3)%4 are
      // candidates for a point on the face's plane. We arbitrarily
      // choose (face + 1) % 4.
      const uint32_t face_vertex_index = (face + 1) % 4;
// This loop is over x,y,z
#pragma unroll
      for (uint32_t i = 0; i < 3; i++) {
        outward_normal[i] = -inward_normals_W[B_element_index][face][i];

        // Get a point from the verticies of element B
        point_on_face[i] =
            slm[slm_offset + VERTEX_B_OFFSET + face_vertex_index * 3 + i];

        // Get the polygon vertex -> This has to be from
        // POLYGON_CURRENT_OFFSET
        polygon_vertex_coords[i] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + job * 3 +
                        i];
      }
      // We will store our heights in the random scratch space that we
      // have (we have upto 16 doubles space) They will be ordered in
      // row major order [face_0_vertex_0, face_0_vertex_1,
      // face_0_vertex_2, face_0_vertex_3, face_1_vertex_0, ...]
      const double displacement = outward_normal[0] * point_on_face[0] +
                                  outward_normal[1] * point_on_face[1] +
                                  outward_normal[2] * point_on_face[2];

      // <= 0 is inside, > 0 is outside
      slm[slm_offset + RANDOM_SCRATCH_OFFSET + job] =
          outward_normal[0] * polygon_vertex_coords[0] +
          outward_normal[1] * polygon_vertex_coords[1] +
          outward_normal[2] * polygon_vertex_coords[2] - displacement;
    }
    // Sync shared memory
    sycl::group_barrier(sub_group);

    // Now we will walk the current polygon and construct the clipped
    // polygon
    for (uint32_t vertex_0_index = check_local_item_id;
         vertex_0_index < num_edges_current_polygon;
         vertex_0_index += NUM_THREADS_PER_CHECK) {
      // Get the height of vertex_1
      const uint32_t vertex_1_index =
          (vertex_0_index + 1) % num_edges_current_polygon;

      // Get the height of vertex_0
      double height_0 =
          slm[slm_offset + RANDOM_SCRATCH_OFFSET + vertex_0_index];
      double height_1 =
          slm[slm_offset + RANDOM_SCRATCH_OFFSET + vertex_1_index];

      // Each edge can store upto two vertices
      // We will do a compaction in the end
      if (height_0 <= 0) {
        // If vertex_0 is inside, it is part of the clipped polygon

        // Copy vertex_0 into the clipped polygon
        // The "2" multiplier is because each edge can contribute in
        // one loop upto 2 vertices. The "3" is because of xyz
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 0] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        vertex_0_index * 3 + 0];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 1] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        vertex_0_index * 3 + 1];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 2] =
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        vertex_0_index * 3 + 2];

        // Now if vertex_1 is outside, we will have an intersection
        // point too
        if (height_1 > 0) {
          // Compute the intersection point
          const double wa = height_1 / (height_1 - height_0);
          const double wb = 1 - wa;

          // Copy the intersection point into the clipped polygon
          slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                      (2 * vertex_0_index + 1) * 3 + 0] =
              wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_0_index * 3 + 0] +
              wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_1_index * 3 + 0];
          slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                      (2 * vertex_0_index + 1) * 3 + 1] =
              wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_0_index * 3 + 1] +
              wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_1_index * 3 + 1];
          slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                      (2 * vertex_0_index + 1) * 3 + 2] =
              wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_0_index * 3 + 2] +
              wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                               vertex_1_index * 3 + 2];
        }
      } else if (height_1 <= 0) {
        // If vertex_1 is inside and vertex_0 is outside this edge
        // will contribute 1 point (intersection point)
        const double wa = height_1 / (height_1 - height_0);
        const double wb = 1 - wa;

        // Copy the intersection point into the clipped polygon
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 0] =
            wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_0_index * 3 + 0] +
            wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_1_index * 3 + 0];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 1] =
            wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_0_index * 3 + 1] +
            wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_1_index * 3 + 1];
        slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET +
                    2 * vertex_0_index * 3 + 2] =
            wa * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_0_index * 3 + 2] +
            wb * slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                             vertex_1_index * 3 + 2];
      }
    }
    sycl::group_barrier(sub_group);

    // Flip Current and clipped polygon
    const uint32_t temp_polygon_current_offset = POLYGON_CURRENT_OFFSET;
    POLYGON_CURRENT_OFFSET = POLYGON_CLIPPED_OFFSET;
    POLYGON_CLIPPED_OFFSET = temp_polygon_current_offset;

    // Now clean up the current polygon to remove out the
    // std::numeric_limits<double>::max() vertices
    if (check_local_item_id == 0) {
      uint32_t write_index = 0;
      // Scan through all potential vertices
      for (uint32_t read_index = 0; read_index < POLYGON_VERTICES;
           ++read_index) {
        // Check if this vertex is valid (not max value)
        if (slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        read_index * 3 + 0] !=
            std::numeric_limits<double>::max()) {
          // Only copy if read and write indices are different
          if (read_index != write_index) {
            // Copy the valid vertex to the write position
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        write_index * 3 + 0] =
                slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                            read_index * 3 + 0];
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        write_index * 3 + 1] =
                slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                            read_index * 3 + 1];
            slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                        write_index * 3 + 2] =
                slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                            read_index * 3 + 2];
          }
          write_index++;
        }
      }

      // Fill remaining positions with max values to mark them as
      // invalid At the same time even fill in the clipped polygon
      // with max values
      for (uint32_t i = write_index; i < POLYGON_VERTICES; ++i) {
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + i * 3 + 0] =
            std::numeric_limits<double>::max();
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + i * 3 + 1] =
            std::numeric_limits<double>::max();
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + i * 3 + 2] =
            std::numeric_limits<double>::max();
      }

      // Update polygon size
      slm_ints[slm_ints_offset] = write_index;
    }
    sycl::group_barrier(sub_group);

    // Clear out the clipped polygon
    for (uint32_t llid = check_local_item_id; llid < POLYGON_VERTICES;
         llid += NUM_THREADS_PER_CHECK) {
      slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET + llid * 3 + 0] =
          std::numeric_limits<double>::max();
      slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET + llid * 3 + 1] =
          std::numeric_limits<double>::max();
      slm_polygon[slm_polygon_offset + POLYGON_CLIPPED_OFFSET + llid * 3 + 2] =
          std::numeric_limits<double>::max();
    }
    sycl::group_barrier(sub_group);
  }

  if (check_local_item_id == 0 && slm_ints[slm_ints_offset] < 3) {
    narrow_phase_check_validity[narrow_phase_check_index] = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (narrow_phase_check_validity[narrow_phase_check_index] == 0) {
    return;
  }

  // Now we compute the area and the centroid of the polygons
  // Compute mean vertex of the polygon using a reduce
  // We will use the clipped polygon shared memory area for all these
  // intermediate results

  // We use one of the polygon's vertices as our base point to cut the
  // polygon into triangles We will use the first point for this
  const uint32_t polygon_size = slm_ints[slm_ints_offset];
  const uint32_t AREAS_OFFSET = POLYGON_CLIPPED_OFFSET;
  const uint32_t CENTROID_OFFSET = VERTEX_A_OFFSET;
  double thread_area_sum = 0;
  double thread_centroid_x = 0;
  double thread_centroid_y = 0;
  double thread_centroid_z = 0;
  for (uint32_t triangle_index = check_local_item_id;
       triangle_index + 2 < polygon_size;
       triangle_index += NUM_THREADS_PER_CHECK) {
    const double v0_x =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + 0 * 3 + 0];
    const double v0_y =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + 0 * 3 + 1];
    const double v0_z =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET + 0 * 3 + 2];
    // Compute the thread local cross magnitude

    // First vertex of triangle edge (current polygon vertex)
    const double v1_x =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 1) * 3 + 0];
    const double v1_y =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 1) * 3 + 1];
    const double v1_z =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 1) * 3 + 2];

    // Second vertex of triangle edge (next polygon vertex, wrapping
    // around)
    const double v2_x =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 2) * 3 + 0];
    const double v2_y =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 2) * 3 + 1];
    const double v2_z =
        slm_polygon[slm_polygon_offset + POLYGON_CURRENT_OFFSET +
                    (triangle_index + 2) * 3 + 2];

    const double r_UV_x = v1_x - v0_x;
    const double r_UV_y = v1_y - v0_y;
    const double r_UV_z = v1_z - v0_z;

    const double r_UW_x = v2_x - v0_x;
    const double r_UW_y = v2_y - v0_y;
    const double r_UW_z = v2_z - v0_z;

    const double cross_x = r_UV_y * r_UW_z - r_UV_z * r_UW_y;
    const double cross_y = r_UV_z * r_UW_x - r_UV_x * r_UW_z;
    const double cross_z = r_UV_x * r_UW_y - r_UV_y * r_UW_x;

    const double cross_magnitude =
        sycl::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
    thread_area_sum += cross_magnitude;

    // Compute the thread local centroid
    thread_centroid_x += cross_magnitude * (v1_x + v2_x + v0_x);
    thread_centroid_y += cross_magnitude * (v1_y + v2_y + v0_y);
    thread_centroid_z += cross_magnitude * (v1_z + v2_z + v0_z);
  }

  // Now each thread writes its computed values
  if (check_local_item_id < polygon_size) {
    slm_polygon[slm_polygon_offset + AREAS_OFFSET + check_local_item_id] =
        thread_area_sum;
    slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 0] =
        thread_centroid_x;
    slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 1] =
        thread_centroid_y;
    slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 2] =
        thread_centroid_z;
  }

  sycl::group_barrier(sub_group);

  for (uint32_t stride = NUM_THREADS_PER_CHECK / 2; stride > 0; stride >>= 1) {
    if (check_local_item_id < stride &&
        check_local_item_id + stride < polygon_size) {
      slm_polygon[slm_polygon_offset + AREAS_OFFSET + check_local_item_id] +=
          slm_polygon[slm_polygon_offset + AREAS_OFFSET +
                      (check_local_item_id + stride)];
      slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 0] +=
          slm[slm_offset + CENTROID_OFFSET +
              (check_local_item_id + stride) * 3 + 0];
      slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 1] +=
          slm[slm_offset + CENTROID_OFFSET +
              (check_local_item_id + stride) * 3 + 1];
      slm[slm_offset + CENTROID_OFFSET + check_local_item_id * 3 + 2] +=
          slm[slm_offset + CENTROID_OFFSET +
              (check_local_item_id + stride) * 3 + 2];
    }
  }
  sycl::group_barrier(sub_group);

  // Now write everything to global memory
  if (check_local_item_id == 0) {
    // Write Polygon Area and Centroid
    const double polygon_area =
        slm_polygon[slm_polygon_offset + AREAS_OFFSET + 0] * 0.5;
    if (polygon_area > 1e-15) {
      polygon_areas[narrow_phase_check_index] = polygon_area;
      const double inv_polygon_area_6 = 1.0 / (polygon_area * 6);
      const double centroid_x =
          slm[slm_offset + CENTROID_OFFSET + 0 * 3 + 0] * inv_polygon_area_6;
      const double centroid_y =
          slm[slm_offset + CENTROID_OFFSET + 0 * 3 + 1] * inv_polygon_area_6;
      const double centroid_z =
          slm[slm_offset + CENTROID_OFFSET + 0 * 3 + 2] * inv_polygon_area_6;
      polygon_centroids[narrow_phase_check_index][0] = centroid_x;
      polygon_centroids[narrow_phase_check_index][1] = centroid_y;
      polygon_centroids[narrow_phase_check_index][2] = centroid_z;

      // Write Polygon Normal -> This is already normalized
      polygon_normals[narrow_phase_check_index][0] =
          slm[slm_offset + EQ_PLANE_OFFSET];
      polygon_normals[narrow_phase_check_index][1] =
          slm[slm_offset + EQ_PLANE_OFFSET + 1];
      polygon_normals[narrow_phase_check_index][2] =
          slm[slm_offset + EQ_PLANE_OFFSET + 2];

      // Write Polygon g_M_
      polygon_g_M[narrow_phase_check_index] =
          slm[slm_offset + EQ_PLANE_OFFSET + 6];
      // Write Polygon g_N_
      polygon_g_N[narrow_phase_check_index] =
          slm[slm_offset + EQ_PLANE_OFFSET + 7];

      // Compute the pressure at the centroid
      // TODO(Huzaifa): Check this for correctness
      const double gradP_A_Wo_x = gradient_W_pressure_at_Wo[A_element_index][0];
      const double gradP_A_Wo_y = gradient_W_pressure_at_Wo[A_element_index][1];
      const double gradP_A_Wo_z = gradient_W_pressure_at_Wo[A_element_index][2];
      const double p_A_Wo = gradient_W_pressure_at_Wo[A_element_index][3];
      polygon_pressure_W[narrow_phase_check_index] =
          gradP_A_Wo_x * centroid_x + gradP_A_Wo_y * centroid_y +
          gradP_A_Wo_z * centroid_z + p_A_Wo;

      const uint32_t geom_index_A = element_mesh_ids[A_element_index];
      const uint32_t geom_index_B = element_mesh_ids[B_element_index];
      // Write Geometry Index A
      polygon_geom_index_A[narrow_phase_check_index] =
          geometry_ids[geom_index_A];
      // Write Geometry Index B
      polygon_geom_index_B[narrow_phase_check_index] =
          geometry_ids[geom_index_B];
    } else {
      narrow_phase_check_validity[narrow_phase_check_index] = 0;
    }
  }
}

/* High-level function that launches the entire contact polygon computation
 * pipeline
 *
 * This function encapsulates:
 * - All constant calculations and memory layout
 * - Local memory size validation
 * - Kernel launch with appropriate work group sizing
 * - Device type detection and template instantiation
 *
 * @param q_device SYCL queue for kernel submission
 * @param dependencies Vector of SYCL events this computation depends on
 * @param total_narrow_phase_checks Total number of narrow phase checks
 * @param collision_data Device collision data structures
 * @param mesh_data Device mesh data structures
 * @param polygon_data Device polygon data structures
 * @returns SYCL event for the contact polygon computation
 */

template <typename DeviceCollidingIndicesMemoryChunk,
          typename DeviceCollisionData, typename MeshData, typename PolygonData,
          DeviceType device_type>
sycl::event LaunchContactPolygonComputation(
    sycl::queue& q_device, const std::vector<sycl::event>& dependencies,
    uint32_t total_narrow_phase_checks,
    const DeviceCollidingIndicesMemoryChunk& pair_chunk,
    const DeviceCollisionData& collision_data, const MeshData& mesh_data,
    const PolygonData& polygon_data) {
  //   constexpr uint32_t NUM_THREADS_PER_CHECK =
  //       device_type == DeviceType::GPU ? 4 : 1;
  constexpr uint32_t NUM_THREADS_PER_CHECK = 4;

  // Demand that NUM_THREADS_PER_CHECK is factor of 32 and less than 32
  static_assert(NUM_THREADS_PER_CHECK <= 32,
                "NUM_THREADS_PER_CHECK must be <= 32");
  static_assert(32 % NUM_THREADS_PER_CHECK == 0,
                "NUM_THREADS_PER_CHECK must be factor of 32");
  constexpr uint32_t SUB_GROUP_SIZE = NUM_THREADS_PER_CHECK;

  // Calculate total threads needed (4 threads per check)
  const uint32_t TOTAL_THREADS_NEEDED =
      total_narrow_phase_checks * NUM_THREADS_PER_CHECK;

  // Check device work group size limits for CPU compatibility
  uint32_t max_work_group_size =
      q_device.get_device().get_info<sycl::info::device::max_work_group_size>();
  constexpr uint32_t LOCAL_SIZE = 128;
  DRAKE_DEMAND(LOCAL_SIZE <= max_work_group_size);
  const uint32_t NUM_CHECKS_IN_WORK_GROUP = LOCAL_SIZE / NUM_THREADS_PER_CHECK;
  // Number of work groups
  const uint32_t NUM_GROUPS =
      std::max(static_cast<uint32_t>(1),
               (TOTAL_THREADS_NEEDED + LOCAL_SIZE - 1) / LOCAL_SIZE);

  // Calculation of the number of doubles to be stored in shared memory per
  // check Offsets are required to index the local memory Two extra for gM and
  // gN
  constexpr uint32_t EQ_PLANE_OFFSET = 0;
  constexpr uint32_t EQ_PLANE_DOUBLES = 8;

  constexpr uint32_t VERTEX_A_OFFSET = EQ_PLANE_OFFSET + EQ_PLANE_DOUBLES;
  constexpr uint32_t VERTEX_A_DOUBLES = 12;
  constexpr uint32_t VERTEX_B_OFFSET = VERTEX_A_OFFSET + VERTEX_A_DOUBLES;
  constexpr uint32_t VERTEX_B_DOUBLES = 12;

  // Used varylingly through the kernel to express more parallelism
  constexpr uint32_t RANDOM_SCRATCH_OFFSET = VERTEX_B_OFFSET + VERTEX_B_DOUBLES;
  constexpr uint32_t RANDOM_SCRATCH_DOUBLES = 8;  // 8 heights at max

  // Calculate total doubles for verification
  constexpr uint32_t VERTEX_DOUBLES = VERTEX_A_DOUBLES + VERTEX_B_DOUBLES;

  constexpr uint32_t DOUBLES_PER_CHECK =
      EQ_PLANE_DOUBLES + VERTEX_DOUBLES + RANDOM_SCRATCH_DOUBLES;

  constexpr uint32_t POLYGON_CURRENT_DOUBLES =
      48;  // 16 vertices (although 8 is max, we need 16 because each edge can
           // produce 2 vertices which means for parallelization and indexing
           // we need 16)
  constexpr uint32_t POLYGON_CLIPPED_DOUBLES = 48;  // 16 vertices
  constexpr uint32_t POLYGON_DOUBLES =
      POLYGON_CURRENT_DOUBLES + POLYGON_CLIPPED_DOUBLES;

  constexpr uint32_t POLYGON_VERTICES =
      16;  // Just useful to have this in the kernels

  // Additionally lets have a random scratch space for storing INTS
  // These will also be used varyingly throughout the kernel to express
  // parallelism
  constexpr uint32_t RANDOM_SCRATCH_INTS = 1;
  // Launch the contact polygon computation kernel
  return q_device.submit([&](sycl::handler& h) {
    h.depends_on(dependencies);

    // Check local memory size constraints for CPU compatibility
    uint32_t slm_size = LOCAL_SIZE / NUM_THREADS_PER_CHECK * DOUBLES_PER_CHECK;
    uint32_t slm_polygon_size =
        LOCAL_SIZE / NUM_THREADS_PER_CHECK * POLYGON_DOUBLES;
    uint32_t slm_ints_size =
        LOCAL_SIZE / NUM_THREADS_PER_CHECK * RANDOM_SCRATCH_INTS;

    uint32_t total_local_memory =
        (slm_size + slm_polygon_size) * sizeof(double) +
        slm_ints_size * sizeof(int);
    uint32_t max_local_memory =
        q_device.get_device().get_info<sycl::info::device::local_mem_size>();
    if (total_local_memory > max_local_memory) {
      throw std::runtime_error("Requested local memory (" +
                               std::to_string(total_local_memory) +
                               " bytes) exceeds device limit (" +
                               std::to_string(max_local_memory) + " bytes)");
    }

    // Shared Local Memory (SLM) is stored as
    // [Eq_plane_i, Vertices_A_i, Vertices_B_i, Inward_normals_A_i,
    // Inward_normals_B_i, Eq_plane_i+1, Vertices_A_i+1,
    // Vertices_B_i+1, Inward_normals_A_i+1, Inward_normals_B_i+1]
    // Always Polygon A stored first and then Polygon B (for the
    // quantities which we need both off)
    sycl::local_accessor<double, 1> slm(slm_size, h);
    sycl::local_accessor<double, 1> slm_polygon(slm_polygon_size, h);
    sycl::local_accessor<int, 1> slm_ints(slm_ints_size, h);
    constexpr uint32_t SUB_GROUP_SIZE = NUM_THREADS_PER_CHECK;
    h.parallel_for<ComputeContactPolygonsKernel<device_type>>(
        sycl::nd_range<1>{NUM_GROUPS * LOCAL_SIZE, LOCAL_SIZE},
        [=, gradient_W_pressure_at_Wo = mesh_data.gradient_W_pressure_at_Wo,
         vertex_offsets = mesh_data.vertex_offsets,
         element_mesh_ids = mesh_data.element_mesh_ids,
         elements = mesh_data.elements, vertices_W = mesh_data.vertices_W,
         inward_normals_W = mesh_data.inward_normals_W,
         collision_indices_A = pair_chunk.collision_indices_A,
         collision_indices_B = pair_chunk.collision_indices_B,
         narrow_phase_check_validity =
             collision_data.narrow_phase_check_validity,
         polygon_areas = polygon_data.polygon_areas,
         polygon_centroids = polygon_data.polygon_centroids,
         polygon_normals = polygon_data.polygon_normals,
         polygon_g_M = polygon_data.polygon_g_M,
         polygon_g_N = polygon_data.polygon_g_N,
         polygon_pressure_W = polygon_data.polygon_pressure_W,
         polygon_geom_index_A = polygon_data.polygon_geom_index_A,
         polygon_geom_index_B = polygon_data.polygon_geom_index_B,
         geometry_ids = mesh_data.geometry_ids,
         TOTAL_THREADS_NEEDED = TOTAL_THREADS_NEEDED,
         NUM_THREADS_PER_CHECK = NUM_THREADS_PER_CHECK,
         DOUBLES_PER_CHECK = DOUBLES_PER_CHECK,
         POLYGON_DOUBLES = POLYGON_DOUBLES, EQ_PLANE_OFFSET = EQ_PLANE_OFFSET,
         VERTEX_A_OFFSET = VERTEX_A_OFFSET, VERTEX_B_OFFSET = VERTEX_B_OFFSET,
         RANDOM_SCRATCH_OFFSET = RANDOM_SCRATCH_OFFSET,
         POLYGON_VERTICES = POLYGON_VERTICES] [[intel::kernel_args_restrict]]
#ifndef __NVPTX__
        [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]]
#endif
#ifdef __NVPTX__
        [[sycl::reqd_work_group_size(LOCAL_SIZE)]]
#endif
        (sycl::nd_item<1> item) {
          ComputeContactPolygonsNoReturn(
              item, slm, slm_polygon, slm_ints, TOTAL_THREADS_NEEDED,
              NUM_THREADS_PER_CHECK, DOUBLES_PER_CHECK, POLYGON_DOUBLES,
              EQ_PLANE_OFFSET, VERTEX_A_OFFSET, VERTEX_B_OFFSET,
              RANDOM_SCRATCH_OFFSET, POLYGON_VERTICES,
              gradient_W_pressure_at_Wo, vertex_offsets, element_mesh_ids,
              elements, vertices_W, inward_normals_W, collision_indices_A,
              collision_indices_B, narrow_phase_check_validity, polygon_areas,
              polygon_centroids, polygon_normals, polygon_g_M, polygon_g_N,
              polygon_pressure_W, polygon_geom_index_A, polygon_geom_index_B,
              geometry_ids);
          // ComputeContactPolygons(
          //     item, slm, slm_polygon, slm_ints, TOTAL_THREADS_NEEDED,
          //     NUM_THREADS_PER_CHECK, DOUBLES_PER_CHECK, POLYGON_DOUBLES,
          //     EQ_PLANE_OFFSET, VERTEX_A_OFFSET, VERTEX_B_OFFSET,
          //      RANDOM_SCRATCH_OFFSET, POLYGON_VERTICES,
          //     narrow_phase_check_indices, gradient_W_pressure_at_Wo,
          //     element_offsets, vertex_offsets, element_mesh_ids, elements,
          //     vertices_W, inward_normals_W, geom_collision_filter_num_cols,
          //     geom_collision_filter_check_offsets,
          //     collision_filter_host_body_index, narrow_phase_check_validity,
          //     polygon_areas, polygon_centroids, polygon_normals, polygon_g_M,
          //     polygon_g_N, polygon_pressure_W, polygon_geom_index_A,
          //     polygon_geom_index_B, geometry_ids);
        });
  });
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake