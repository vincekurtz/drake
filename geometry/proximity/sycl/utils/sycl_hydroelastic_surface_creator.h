#pragma once

#include "sycl/sycl.hpp"

#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

inline SYCLHydroelasticSurface CreateHydroelasticSurface(
    sycl::queue& q_device, const Vector3<double>* compacted_polygon_centroids,
    const double* compacted_polygon_areas,
    const double* compacted_polygon_pressure_W,
    const Vector3<double>* compacted_polygon_normals,
    const double* compacted_polygon_g_M, const double* compacted_polygon_g_N,
    const GeometryId* compacted_polygon_geom_index_A,
    const GeometryId* compacted_polygon_geom_index_B,
    const uint32_t total_polygons) {
  // Transfer data from device to host
  std::vector<Vector3<double>> host_centroids(total_polygons);
  std::vector<double> host_areas(total_polygons);
  std::vector<double> host_pressure_W(total_polygons);
  std::vector<Vector3<double>> host_normals(total_polygons);
  std::vector<double> host_g_M(total_polygons);
  std::vector<double> host_g_N(total_polygons);
  std::vector<GeometryId> host_geom_A(total_polygons);
  std::vector<GeometryId> host_geom_B(total_polygons);

  // Perform all memory transfers in parallel
  std::vector<sycl::event> transfer_events;
  transfer_events.push_back(
      q_device.memcpy(host_centroids.data(), compacted_polygon_centroids,
                      total_polygons * sizeof(Vector3<double>)));
  transfer_events.push_back(q_device.memcpy(host_areas.data(),
                                            compacted_polygon_areas,
                                            total_polygons * sizeof(double)));
  transfer_events.push_back(q_device.memcpy(host_pressure_W.data(),
                                            compacted_polygon_pressure_W,
                                            total_polygons * sizeof(double)));
  transfer_events.push_back(
      q_device.memcpy(host_normals.data(), compacted_polygon_normals,
                      total_polygons * sizeof(Vector3<double>)));
  transfer_events.push_back(q_device.memcpy(
      host_g_M.data(), compacted_polygon_g_M, total_polygons * sizeof(double)));
  transfer_events.push_back(q_device.memcpy(
      host_g_N.data(), compacted_polygon_g_N, total_polygons * sizeof(double)));
  transfer_events.push_back(
      q_device.memcpy(host_geom_A.data(), compacted_polygon_geom_index_A,
                      total_polygons * sizeof(GeometryId)));
  transfer_events.push_back(
      q_device.memcpy(host_geom_B.data(), compacted_polygon_geom_index_B,
                      total_polygons * sizeof(GeometryId)));
  // Wait for all transfers to complete
  sycl::event::wait_and_throw(transfer_events);

  return SYCLHydroelasticSurface(
      std::move(host_centroids), std::move(host_areas),
      std::move(host_pressure_W), std::move(host_normals), std::move(host_g_M),
      std::move(host_g_N), std::move(host_geom_A), std::move(host_geom_B));
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake