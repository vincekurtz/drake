#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"

#include <algorithm>
#include <map>
#include <utility>

#include "drake/common/drake_assert.h"
namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

SYCLHydroelasticSurface::SYCLHydroelasticSurface(
    std::vector<Vector3<double>> centroids, std::vector<double> areas,
    std::vector<double> pressure_Ws, std::vector<Vector3<double>> normal_Ws,
    std::vector<double> g_M, std::vector<double> g_N,
    std::vector<GeometryId> geometry_ids_M,
    std::vector<GeometryId> geometry_ids_N)
    : centroid_(std::move(centroids)),
      area_(std::move(areas)),
      pressure_W_(std::move(pressure_Ws)),
      normal_W_(std::move(normal_Ws)),
      g_M_(std::move(g_M)),
      g_N_(std::move(g_N)),
      geometry_ids_M_(std::move(geometry_ids_M)),
      geometry_ids_N_(std::move(geometry_ids_N)) {
  // Verify that all vectors have the same size
  DRAKE_THROW_UNLESS(centroid_.size() == area_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == pressure_W_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == normal_W_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == g_M_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == g_N_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == geometry_ids_M_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == geometry_ids_N_.size());
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
