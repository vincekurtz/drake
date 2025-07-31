#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/core.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include "drake/common/text_logging.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/sycl/bvh/sycl_bvh.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

namespace {

using Eigen::Vector3d;
using math::RigidTransformd;

/*
Creates a simple SoftGeometry with two tets whose faces align and their heights
are in opposite directions, and a simple linear field.
*/
hydroelastic::SoftGeometry MakeSimpleSoftGeometry() {
  // Create mesh
  std::vector<Vector3d> p_MV;
  std::vector<VolumeElement> elements;
  p_MV.push_back(Vector3d(0, 0, -1));
  p_MV.push_back(Vector3d(-1, -1, 0));
  p_MV.push_back(Vector3d(1, -1, 0));
  p_MV.push_back(Vector3d(0, 1, 0));
  p_MV.push_back(Vector3d(0, 0, 1));
  elements.emplace_back(1, 3, 2, 0);
  elements.emplace_back(1, 2, 3, 4);
  auto mesh = std::make_unique<VolumeMesh<double>>(std::move(elements),
                                                   std::move(p_MV));

  // Create field
  std::vector<double> pressure(mesh->num_vertices());
  for (int i = 0; i < mesh->num_vertices(); ++i) {
    pressure[i] = i;
  }
  auto field = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      std::move(pressure), mesh.get());

  // Construct SoftGeometry
  return hydroelastic::SoftGeometry(
      hydroelastic::SoftMesh(std::move(mesh), std::move(field)));
}

std::pair<Vector3<double>, Vector3<double>> ComputeTotalBounds(
    const VolumeMesh<double>& mesh) {
  const auto& mesh_elements = mesh.tetrahedra();
  Vector3<double> min_W = mesh.vertex(0).cast<double>();
  Vector3<double> max_W = min_W;
  for (int i = 1; i < mesh.num_vertices(); ++i) {
    const Vector3<double>& p_W = mesh.vertex(i).cast<double>();
    min_W = min_W.cwiseMin(p_W);
    max_W = max_W.cwiseMax(p_W);
  }
  return {min_W, max_W};
}

VolumeMesh<double> TransformMesh(const hydroelastic::SoftGeometry& geometry,
                                 const RigidTransformd& X_WG) {
  auto mesh = geometry.soft_mesh().mesh();
  mesh.TransformVertices(X_WG);
  return mesh;
}

// Helper to check BVH properties for a given mesh in the SYCL engine.
std::tuple<int, int, int, double, bool> CheckSyclBvhProperties(
    const HostBVH& host_bvh, const std::string& filepath) {
  const int height =
      SyclProximityEngineAttorney::ComputeBVHTreeHeight(host_bvh);
  const int num_leaves = SyclProximityEngineAttorney::CountBVHLeaves(host_bvh);
  const int balance_factor =
      SyclProximityEngineAttorney::ComputeBVHBalanceFactor(host_bvh);
  const double average_depth =
      SyclProximityEngineAttorney::ComputeBVHAverageLeafDepth(host_bvh);
  const bool bounds_valid =
      SyclProximityEngineAttorney::VerifyBVHBounds(host_bvh);
  SyclProximityEngineAttorney::ComputeAndPrintBVHImbalanceHistogram(host_bvh,
                                                                    filepath);
  EXPECT_EQ(bounds_valid, true);

  return {height, num_leaves, balance_factor, average_depth, bounds_valid};
}

GTEST_TEST(SPETest, ZeroMeshes) {
  // Should throw when soft_geometries is empty
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries;
  std::unordered_map<GeometryId, Vector3<double>> total_lower;
  std::unordered_map<GeometryId, Vector3<double>> total_upper;
  EXPECT_THROW(drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
                   soft_geometries, total_lower, total_upper),
               std::runtime_error);
}

GTEST_TEST(SPETest, SingleMesh) {
  GeometryId id = GeometryId::get_new_id();
  auto geometry = MakeSimpleSoftGeometry();

  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {id, geometry}};
  auto [min_W, max_W] = ComputeTotalBounds(geometry.soft_mesh().mesh());
  std::unordered_map<GeometryId, Vector3<double>> total_lower{{id, min_W}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{{id, max_W}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);

  // Should not error out and just return early as num_geometries_ < 2
  std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {id, RigidTransformd::Identity()}};
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);
}

GTEST_TEST(SPETest, TwoMeshesColliding) {
  GeometryId idA = GeometryId::get_new_id();
  GeometryId idB = GeometryId::get_new_id();
  auto geometryA = MakeSimpleSoftGeometry();
  auto geometryB = MakeSimpleSoftGeometry();

  // engine.UpdateCollisionCandidates({SortedPair<GeometryId>(idA, idB)});
  // Move meshes along Z so that they just intersect
  std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {idA, RigidTransformd(Vector3d{0, 0, 0})},
      {idB, RigidTransformd(Vector3d{0, 0, 1.1})}};
  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(geometryA, X_WGs[idA]));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(geometryB, X_WGs[idB]));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {idA, geometryA}, {idB, geometryB}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{{idA, minA},
                                                              {idB, minB}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{{idA, maxA},
                                                              {idB, maxB}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);
  std::vector<SortedPair<GeometryId>> collision_candidates{
      SortedPair<GeometryId>(idA, idB)};
  engine.UpdateCollisionCandidates(collision_candidates);
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  auto impl = SyclProximityEngineAttorney::get_impl(engine);

  auto verticies_from_meshA = geometryA.mesh().vertices();
  auto verticies_from_meshB = geometryB.mesh().vertices();
  std::vector<Vector3d> vertices_of_both_meshes;
  vertices_of_both_meshes.insert(vertices_of_both_meshes.end(),
                                 verticies_from_meshA.begin(),
                                 verticies_from_meshA.end());
  vertices_of_both_meshes.insert(vertices_of_both_meshes.end(),
                                 verticies_from_meshB.begin(),
                                 verticies_from_meshB.end());

  auto vertices_M_host = SyclProximityEngineAttorney::get_vertices_M(impl);
  auto vertices_W_host = SyclProximityEngineAttorney::get_vertices_W(impl);
  EXPECT_EQ(vertices_M_host.size(), vertices_of_both_meshes.size());

  // Compare vertices within machine precision
  for (uint32_t i = 0; i < vertices_M_host.size(); ++i) {
    EXPECT_NEAR(vertices_M_host[i][0], vertices_of_both_meshes[i][0],
                std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(vertices_M_host[i][1], vertices_of_both_meshes[i][1],
                std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(vertices_M_host[i][2], vertices_of_both_meshes[i][2],
                std::numeric_limits<double>::epsilon());
  }
  // Elements stored should be same as elements from mesh
  auto elements_host = SyclProximityEngineAttorney::get_elements(impl);
  auto elements_from_meshA = geometryA.mesh().pack_element_vertices();
  auto elements_from_meshB = geometryB.mesh().pack_element_vertices();
  std::vector<std::array<int, 4>> elements_of_both_meshes;
  elements_of_both_meshes.insert(elements_of_both_meshes.end(),
                                 elements_from_meshA.begin(),
                                 elements_from_meshA.end());
  elements_of_both_meshes.insert(elements_of_both_meshes.end(),
                                 elements_from_meshB.begin(),
                                 elements_from_meshB.end());
  EXPECT_EQ(elements_host.size(), elements_of_both_meshes.size());
  for (uint32_t i = 0; i < elements_host.size(); ++i) {
    EXPECT_EQ(elements_host[i], elements_of_both_meshes[i]);
  }

  auto collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  // loop over CPU collision candidates and just print global collision indices
  for (auto& candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    EXPECT_EQ(cc.total_collisions, 1);
    for (uint32_t i = 0; i < ci.collision_indices_A.size(); ++i) {
      // Mesh 1 2nd element collides with Mesh 2 1st element
      // Indices returned by ci is global indices
      EXPECT_EQ(ci.collision_indices_A[i], 1);
      EXPECT_EQ(ci.collision_indices_B[i], 2);
    }
  }

  // Move geometries closer so that all elements are colliding and check
  // collision filter
  X_WGs[idB] = RigidTransformd(Vector3d{0, 0, 0.3});
  engine.UpdateCollisionCandidates({SortedPair<GeometryId>(idA, idB)});
  surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  for (auto& candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    EXPECT_EQ(cc.total_collisions, 3);
    EXPECT_EQ(ci.collision_indices_A[0], 0);
    EXPECT_EQ(ci.collision_indices_B[0], 2);
    EXPECT_EQ(ci.collision_indices_A[1], 1);
    EXPECT_EQ(ci.collision_indices_B[1], 2);
    EXPECT_EQ(ci.collision_indices_A[2], 1);
    EXPECT_EQ(ci.collision_indices_B[2], 3);
  }
}

GTEST_TEST(SPETest, ThreeMeshesAllColliding) {
  GeometryId idA = GeometryId::get_new_id();
  GeometryId idB = GeometryId::get_new_id();
  GeometryId idC = GeometryId::get_new_id();
  auto geometryA = MakeSimpleSoftGeometry();
  auto geometryB = MakeSimpleSoftGeometry();
  auto geometryC = MakeSimpleSoftGeometry();
  std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {idA, RigidTransformd(Vector3d{0, 0, 0})},
      {idB, RigidTransformd(Vector3d{0, 0, 1.1})},
      {idC, RigidTransformd(Vector3d{0, 0, 2.2})}};
  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(geometryA, X_WGs[idA]));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(geometryB, X_WGs[idB]));
  auto [minC, maxC] = ComputeTotalBounds(TransformMesh(geometryC, X_WGs[idC]));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {idA, geometryA}, {idB, geometryB}, {idC, geometryC}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{
      {idA, minA}, {idB, minB}, {idC, minC}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{
      {idA, maxA}, {idB, maxB}, {idC, maxC}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);
  std::vector<SortedPair<GeometryId>> collision_candidates{
      SortedPair<GeometryId>(idA, idB), SortedPair<GeometryId>(idA, idC),
      SortedPair<GeometryId>(idB, idC)};
  engine.UpdateCollisionCandidates(collision_candidates);
  // Move meshes along Z so that they just intersect

  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  auto impl = SyclProximityEngineAttorney::get_impl(engine);

  auto collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  for (auto& candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    if (candidate == SortedPair<GeometryId>(idA, idB)) {
      EXPECT_EQ(cc.total_collisions, 1);
      EXPECT_EQ(ci.collision_indices_A[0], 1);
      EXPECT_EQ(ci.collision_indices_B[0], 2);
    } else if (candidate == SortedPair<GeometryId>(idB, idC)) {
      EXPECT_EQ(cc.total_collisions, 1);
      EXPECT_EQ(ci.collision_indices_A[0], 3);
      EXPECT_EQ(ci.collision_indices_B[0], 4);
    } else if (candidate == SortedPair<GeometryId>(idA, idC)) {
      EXPECT_EQ(cc.total_collisions, 0);
    }
  }

  // Move meshes closer so all elements collide
  X_WGs[idB] = RigidTransformd(Vector3d{0, 0, 0.3});
  X_WGs[idC] = RigidTransformd(Vector3d{0, 0, 0.6});
  engine.UpdateCollisionCandidates(collision_candidates);
  surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  for (auto& candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    if (candidate == SortedPair<GeometryId>(idA, idB)) {
      EXPECT_EQ(cc.total_collisions, 3);
      EXPECT_EQ(ci.collision_indices_A[0], 0);
      EXPECT_EQ(ci.collision_indices_B[0], 2);
      EXPECT_EQ(ci.collision_indices_A[1], 1);
      EXPECT_EQ(ci.collision_indices_B[1], 2);
      EXPECT_EQ(ci.collision_indices_A[2], 1);
      EXPECT_EQ(ci.collision_indices_B[2], 3);
    } else if (candidate == SortedPair<GeometryId>(idA, idC)) {
      EXPECT_EQ(cc.total_collisions, 3);
      EXPECT_EQ(ci.collision_indices_A[0], 0);
      EXPECT_EQ(ci.collision_indices_B[0], 4);
      EXPECT_EQ(ci.collision_indices_A[1], 1);
      EXPECT_EQ(ci.collision_indices_B[1], 4);
      EXPECT_EQ(ci.collision_indices_A[2], 1);
      EXPECT_EQ(ci.collision_indices_B[2], 5);
    } else if (candidate == SortedPair<GeometryId>(idB, idC)) {
      EXPECT_EQ(cc.total_collisions, 3);
      EXPECT_EQ(ci.collision_indices_A[0], 2);
      EXPECT_EQ(ci.collision_indices_B[0], 4);
      EXPECT_EQ(ci.collision_indices_A[1], 3);
      EXPECT_EQ(ci.collision_indices_B[1], 4);
      EXPECT_EQ(ci.collision_indices_A[2], 3);
      EXPECT_EQ(ci.collision_indices_B[2], 5);
    }
  }
}

GTEST_TEST(SPETest, FourMeshAllColliding) {
  GeometryId idA = GeometryId::get_new_id();
  GeometryId idB = GeometryId::get_new_id();
  GeometryId idC = GeometryId::get_new_id();
  GeometryId idD = GeometryId::get_new_id();
  auto geometryA = MakeSimpleSoftGeometry();
  auto geometryB = MakeSimpleSoftGeometry();
  auto geometryC = MakeSimpleSoftGeometry();
  auto geometryD = MakeSimpleSoftGeometry();
  // Move meshes along Z so that they just intersect
  std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {idA, RigidTransformd(Vector3d{0, 0, 0})},
      {idB, RigidTransformd(Vector3d{0, 0, 1.1})},
      {idC, RigidTransformd(Vector3d{0, 0, 2.2})},
      {idD, RigidTransformd(Vector3d{0, 0, 3.3})}};
  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(geometryA, X_WGs[idA]));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(geometryB, X_WGs[idB]));
  auto [minC, maxC] = ComputeTotalBounds(TransformMesh(geometryC, X_WGs[idC]));
  auto [minD, maxD] = ComputeTotalBounds(TransformMesh(geometryD, X_WGs[idD]));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {idA, geometryA}, {idB, geometryB}, {idC, geometryC}, {idD, geometryD}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{
      {idA, minA}, {idB, minB}, {idC, minC}, {idD, minD}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{
      {idA, maxA}, {idB, maxB}, {idC, maxC}, {idD, maxD}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);
  std::vector<SortedPair<GeometryId>> collision_candidates{
      SortedPair<GeometryId>(idA, idB), SortedPair<GeometryId>(idA, idC),
      SortedPair<GeometryId>(idA, idD), SortedPair<GeometryId>(idB, idC),
      SortedPair<GeometryId>(idB, idD), SortedPair<GeometryId>(idC, idD)};
  engine.UpdateCollisionCandidates(collision_candidates);

  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks
  auto impl = SyclProximityEngineAttorney::get_impl(engine);

  auto collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  for (auto& candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    if (candidate == SortedPair<GeometryId>(idA, idB)) {
      EXPECT_EQ(cc.total_collisions, 1);
      EXPECT_EQ(ci.collision_indices_A[0], 1);
      EXPECT_EQ(ci.collision_indices_B[0], 2);
    } else if (candidate == SortedPair<GeometryId>(idA, idC)) {
      EXPECT_EQ(cc.total_collisions, 0);
    } else if (candidate == SortedPair<GeometryId>(idA, idD)) {
      EXPECT_EQ(cc.total_collisions, 0);
    } else if (candidate == SortedPair<GeometryId>(idB, idC)) {
      EXPECT_EQ(cc.total_collisions, 1);
      EXPECT_EQ(ci.collision_indices_A[0], 3);
      EXPECT_EQ(ci.collision_indices_B[0], 4);
    } else if (candidate == SortedPair<GeometryId>(idB, idD)) {
      EXPECT_EQ(cc.total_collisions, 0);
    } else if (candidate == SortedPair<GeometryId>(idC, idD)) {
      EXPECT_EQ(cc.total_collisions, 1);
      EXPECT_EQ(ci.collision_indices_A[0], 5);
      EXPECT_EQ(ci.collision_indices_B[0], 6);
    }
  }
}

GTEST_TEST(SPETest, TwoSpheresColliding) {
  constexpr double radius = 0.5;
  constexpr double resolution_hint = 0.5 * radius;
  constexpr double hydroelastic_modulus = 1e+7;

  // Sphere A
  const Sphere sphereA(radius);
  auto meshA =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereA, resolution_hint,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureA = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereA, meshA.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereA(*meshA);

  const hydroelastic::SoftGeometry soft_geometryA(
      hydroelastic::SoftMesh(std::move(meshA), std::move(pressureA)));
  const GeometryId sphereA_id = GeometryId::get_new_id();

  // Sphere B
  const Sphere sphereB(radius);
  auto meshB =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereB, resolution_hint,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  const hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  // Compute the candidate tets with the two BVHs
  std::vector<std::pair<int, int>> candidate_tetrahedra;
  const auto callback = [&candidate_tetrahedra, &soft_geometryA,
                         &soft_geometryB](int tet0,
                                          int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet1);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_B || max_B < min_A))
      candidate_tetrahedra.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };

  // Arbitrarily pose the spheres into a colliding configuration.
  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.0 * radius, 0.0 * radius, 0.3 * radius});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{1.0 * radius, 0.0 * radius, 0.3 * radius});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);

  bvhSphereA.Collide(bvhSphereB, X_AB, callback);

  // Expected candidate tets map
  std::unordered_map<SortedPair<GeometryId>, std::vector<std::pair<int, int>>>
      expected_candidate_tetrahedra;
  const int num_elements_A = soft_geometryA.mesh().num_elements();
  for (auto [eA, eB] : candidate_tetrahedra) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereA_id,
                                                         sphereB_id)]
        .push_back(std::make_pair(eA, eB + num_elements_A));
  }

  // Convert cadidate tets to collision_filter_ that can be compared to one
  // from sycl_proximity_engine
  std::vector<uint8_t> expected_filter(soft_geometryA.mesh().num_elements() *
                                           soft_geometryB.mesh().num_elements(),
                                       0);
  for (auto [eA, eB] : candidate_tetrahedra) {
    const int i = eA * soft_geometryB.mesh().num_elements() + eB;
    expected_filter[i] = 1;
  }

  // Create inputs to SyclProximityEngine
  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(soft_geometryA, X_WA));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(soft_geometryB, X_WB));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {sphereA_id, soft_geometryA}, {sphereB_id, soft_geometryB}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{
      {sphereA_id, minA}, {sphereB_id, minB}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{
      {sphereA_id, maxA}, {sphereB_id, maxB}};

  // Instantiate SyclProximityEngine to obtain collision filter
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);

  // Update collision candidates
  std::vector<SortedPair<GeometryId>> collision_candidates{
      SortedPair<GeometryId>(sphereA_id, sphereB_id)};
  engine.UpdateCollisionCandidates(collision_candidates);

  // Move spheres closer so that they collide
  const std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {sphereA_id, X_WA}, {sphereB_id, X_WB}};
  const auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks
  const auto impl = SyclProximityEngineAttorney::get_impl(engine);

  auto collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  // Package obtained collisions as pair vector
  std::vector<std::pair<int, int>> obtained_collisions;
  std::vector<std::pair<int, int>> mismatch_pairs;
  // Compare obtained collision pairs with expected from Drake
  for (auto candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    auto ExpMeshA_indices = expected_candidate_tetrahedra[candidate];
    // For each candidate on the GPU, we look for the corresponding
    // candidate on the CPU. We should expect to find all.
    for (uint32_t i = 0; i < cc.total_collisions; ++i) {
      obtained_collisions.emplace_back(
          std::make_pair(static_cast<int>(ci.collision_indices_A[i]),
                         static_cast<int>(ci.collision_indices_B[i])));
      auto it = std::find(
          ExpMeshA_indices.begin(), ExpMeshA_indices.end(),
          std::make_pair(static_cast<int>(ci.collision_indices_A[i]),
                         static_cast<int>(ci.collision_indices_B[i])));
      EXPECT_NE(it, ExpMeshA_indices.end());
    }
    // Inverse check: Find all the candidates found on the CPU but not on the
    // GPU
    for (auto [eA, eB] : ExpMeshA_indices) {
      auto it = std::find(obtained_collisions.begin(),
                          obtained_collisions.end(), std::make_pair(eA, eB));
      if (it == obtained_collisions.end()) {
        mismatch_pairs.emplace_back(std::make_pair(eA, eB));
      }
    }
  }

  // Verify that each of the mismatches is TRULY a false positive from the CPU
  // broadphase. To do this, we compute the Aabb's of the element pairs in the
  // world frame, and then compute the intersection of those Aabb's and verify
  // that each intersection is empty.
  const auto CalcAabb = [](const Vector3d& a, const Vector3d& b,
                           const Vector3d& c, const Vector3d& d) {
    Vector3d min = a;
    Vector3d max = a;
    min = min.cwiseMin(b);
    max = max.cwiseMax(b);
    min = min.cwiseMin(c);
    max = max.cwiseMax(c);
    min = min.cwiseMin(d);
    max = max.cwiseMax(d);
    return std::make_pair(min, max);
  };

  // Check mismatch pairs for false positives
  for (auto [eA, eB] : mismatch_pairs) {
    // Project back to local coordinates
    eB = eB - num_elements_A;
    const auto [minA, maxA] =
        CalcAabb(X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(0)),
                 X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(1)),
                 X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(2)),
                 X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(3)));
    const auto [minB, maxB] =
        CalcAabb(X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(0)),
                 X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(1)),
                 X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(2)),
                 X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(3)));
    // Compute the bounds of the intersection of the Aabbs. The intersection
    // is empty if at least one of the dimensions has negative width.
    const Vector3d intersection_min = minA.cwiseMax(minB);
    const Vector3d intersection_max = maxA.cwiseMin(maxB);
    const Vector3d intersection_widths = intersection_max - intersection_min;
    EXPECT_LE(intersection_widths.minCoeff(), 0);
  }

  std::unique_ptr<PolygonSurfaceMesh<double>> contact_surface;
  std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>
      contact_pressure;
  VolumeIntersector<PolyMeshBuilder<double>, Aabb> volume_intersector;
  volume_intersector.IntersectFields(
      soft_geometryA.pressure_field(), bvhSphereA,
      soft_geometryB.pressure_field(), bvhSphereB, X_AB, &contact_surface,
      &contact_pressure);

  if (surfaces.empty()) {
    fmt::print("No surfaces found in SYCL implementation\n");
    EXPECT_EQ(nullptr, contact_surface.get());
    return;
  }

  // Get the narrow phase check indices
  const uint32_t total_polygons =
      SyclProximityEngineAttorney::get_total_polygons(impl);
  const std::vector<uint32_t> valid_polygon_indices =
      SyclProximityEngineAttorney::get_valid_polygon_indices(impl);

  fmt::print("ssize(compacted_polygon_areas): {}\n", total_polygons);
  fmt::print("contact surface num_faces: {}\n", contact_surface->num_faces());
  std::vector<int> polygons_found;
  std::vector<int> bad_area;
  std::vector<int> bad_centroid;
  std::vector<int> bad_normal;
  std::vector<int> bad_gM;
  std::vector<int> bad_gN;
  std::vector<int> bad_pressure;
  std::vector<std::pair<int, int>> degenerate_tets;

  for (int i = 0; i < contact_surface->num_faces(); ++i) {
    const double expected_area = contact_surface->area(i);
    const Vector3d expected_centroid_M = contact_surface->element_centroid(i);
    // Transform by transforms of A since the contact surface is posed in
    // frame A.
    const Vector3d expected_normal_M = contact_surface->face_normal(i);

    const Vector3d expected_centroid_W = X_WA * expected_centroid_M;
    const Vector3d expected_normal_W =
        (X_WA.rotation() * expected_normal_M).normalized();

    // Find the correct tet pair
    const int tet0 = volume_intersector.tet0_of_polygon(i);
    const int tet1 = volume_intersector.tet1_of_polygon(i);
    const Vector3d grad_field0_M =
        soft_geometryA.pressure_field().EvaluateGradient(tet0);
    const Vector3d grad_field1_M =
        soft_geometryB.pressure_field().EvaluateGradient(tet1);
    const Vector3d grad_field0_W = X_WA.rotation() * grad_field0_M;
    const Vector3d grad_field1_W = X_WB.rotation() * grad_field1_M;

    const double expected_gM = grad_field0_M.dot(expected_normal_M);
    const double expected_gN = grad_field1_M.dot(expected_normal_M);

    const double expected_pressure =
        contact_pressure->EvaluateCartesian(i, expected_centroid_M);

    // Global offsets for the tets
    const int global_tet0 = tet0;
    const int global_tet1 = tet1 + num_elements_A;
    const std::pair<int, int> global_tet_pair{global_tet0, global_tet1};
    const auto it = std::find(obtained_collisions.begin(),
                              obtained_collisions.end(), global_tet_pair);
    // We expect to find polygons for every polygon in the cpu surface.
    EXPECT_TRUE(it != obtained_collisions.end());

    // Do all the checks
    if (it != obtained_collisions.end()) {
      int global_index = (it - obtained_collisions.begin());
      // Find this check index in the valid polygon index
      int index = std::find(valid_polygon_indices.begin(),
                            valid_polygon_indices.end(), global_index) -
                  valid_polygon_indices.begin();
      polygons_found.push_back(index);
      if (std::abs(surfaces[0].areas()[index] - expected_area) >
          1e2 * std::numeric_limits<double>::epsilon()) {
        bad_area.push_back(index);
        std::cerr << fmt::format(
            "Bad area at index {} for tet pair ({}, {}): expected={}, "
            "got={}\n\n",
            index, tet0, tet1, expected_area, surfaces[0].areas()[index]);
        degenerate_tets.push_back(global_tet_pair);
      }
      const double centroid_error =
          (expected_centroid_W - surfaces[0].centroids()[index]).norm();
      if (centroid_error > 1e2 * std::numeric_limits<double>::epsilon() &&
          expected_area > 1e-15) {
        bad_centroid.push_back(index);
        std::cerr << fmt::format(
            "Bad centroid at index {} for tet pair ({}, {}) error: {} "
            "expected area: {}:, got area {}\n  "
            "expected={}\n  got=     "
            "{}\n\n",
            index, tet0, tet1, centroid_error, expected_area,
            surfaces[0].areas()[index],
            fmt_eigen(expected_centroid_W.transpose()),
            fmt_eigen(surfaces[0].centroids()[index].transpose()));
      }
      const double normal_error =
          (expected_normal_W - surfaces[0].normals()[index]).norm();
      if (normal_error > 1e2 * std::numeric_limits<double>::epsilon() &&
          expected_area > 1e-15) {
        bad_normal.push_back(index);
        std::cerr << fmt::format(
            "Bad normal at index {} for tet pair ({}, {}) error: {} "
            "expected area: {}, got area {}\n  "
            "expected={}\n  got=     "
            "{}\n\n",
            index, tet0, tet1, normal_error, expected_area,
            surfaces[0].areas()[index],
            fmt_eigen(expected_normal_W.transpose()),
            fmt_eigen(surfaces[0].normals()[index].transpose()));
      }
      if (expected_gM - surfaces[0].g_M()[index] > 1e-8) {
        bad_gM.push_back(index);
        std::cerr << fmt::format(
            "Bad gM at index {} for tet pair ({}, {}) error: {} "
            "expected area: {}, got area {}\n  "
            "expected={}\n  got=     "
            "{}\n\n",
            index, tet0, tet1, expected_gM - surfaces[0].g_M()[index],
            expected_area, surfaces[0].areas()[index], expected_gM,
            surfaces[0].g_M()[index]);
      }
      if (expected_gN - surfaces[0].g_N()[index] > 1e-8) {
        bad_gN.push_back(index);
        std::cerr << fmt::format(
            "Bad gN at index {} for tet pair ({}, {}) error: {} "
            "expected area: {}, got area {}\n  "
            "expected={}\n  got=     "
            "{}\n\n",
            index, tet0, tet1, expected_gN - surfaces[0].g_N()[index],
            expected_area, surfaces[0].areas()[index], expected_gN,
            surfaces[0].g_N()[index]);
      }
      if (expected_pressure - surfaces[0].pressures()[index] > 1e-8) {
        bad_pressure.push_back(index);
        std::cerr << fmt::format(
            "Bad pressure at index {} for tet pair ({}, {}) error: {} "
            "expected area: {}, got area {}\n  "
            "expected={}\n  got=     "
            "{}\n\n",
            index, tet0, tet1,
            expected_pressure - surfaces[0].pressures()[index], expected_area,
            surfaces[0].areas()[index], expected_pressure,
            surfaces[0].pressures()[index]);
      }
    }
  }
  fmt::print("Polygons found by SYCL implementation: {}\n",
             ssize(polygons_found));
  fmt::print("Polygons with area difference beyond rounding error: {}\n",
             ssize(bad_area));
  EXPECT_EQ(bad_area.size(), 0);
  fmt::print(
      "Polygons with centroid difference beyond rounding error (in any of x, "
      "y, z): {}\n",
      ssize(bad_centroid));
  EXPECT_EQ(bad_centroid.size(), 0);
  fmt::print("Polygons with normal difference beyond rounding error: {}\n",
             ssize(bad_normal));
  EXPECT_EQ(bad_normal.size(), 0);
  fmt::print("Polygons with gM difference beyond rounding error: {}\n",
             ssize(bad_gM));
  EXPECT_EQ(bad_gM.size(), 0);
  fmt::print("Polygons with gN difference beyond rounding error: {}\n",
             ssize(bad_gN));
  EXPECT_EQ(bad_gN.size(), 0);
  fmt::print("Polygons with pressure difference beyond rounding error: {}\n",
             ssize(bad_pressure));
  EXPECT_EQ(bad_pressure.size(), 0);

  std::sort(polygons_found.begin(), polygons_found.end());
  int counter = 0;
  for (int i = 0; i < static_cast<int>(surfaces[0].num_polygons()); ++i) {
    if (!std::binary_search(polygons_found.begin(), polygons_found.end(), i)) {
      if (surfaces[0].areas()[i] > 1e-15) {
        int index = valid_polygon_indices[i];
        std::cerr << fmt::format(
            "Polygon with index {} and tet pair ({}, {}) has area {} and "
            "centroid {} in SYCL but not found in Drake\n",
            i, obtained_collisions[index].first,
            obtained_collisions[index].second, surfaces[0].areas()[i],
            fmt_eigen(surfaces[0].centroids()[i].transpose()));
        counter++;
      }
    }
  }
  fmt::print("Polygons found by SYCL implementation but NOT in Drake: {}\n",
             counter);
  EXPECT_EQ(counter, 0);
}

GTEST_TEST(SPETest, ThreeSpheresColliding) {
  constexpr double radius = 1.0;
  constexpr double resolution_hint_A = 2 * radius;
  constexpr double resolution_hint_B = 2 * radius;
  constexpr double resolution_hint_C = 1 * radius;
  constexpr double hydroelastic_modulus = 1e+7;

  // Sphere A
  const Sphere sphereA(radius);
  auto meshA =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereA, resolution_hint_A,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureA = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereA, meshA.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereA(*meshA);

  hydroelastic::SoftGeometry soft_geometryA(
      hydroelastic::SoftMesh(std::move(meshA), std::move(pressureA)));
  const GeometryId sphereA_id = GeometryId::get_new_id();

  // Sphere B
  const Sphere sphereB(radius);
  auto meshB =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereB, resolution_hint_B,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  // Sphere C
  const Sphere sphereC(radius);
  auto meshC =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereC, resolution_hint_C,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureC = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereC, meshC.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereC(*meshC);
  hydroelastic::SoftGeometry soft_geometryC(
      hydroelastic::SoftMesh(std::move(meshC), std::move(pressureC)));
  const GeometryId sphereC_id = GeometryId::get_new_id();

  // ARbitrarily pose the spheres into a colliding configuration.
  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.2 * radius, 0.1 * radius, 0.3 * radius});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{0.1 * radius, 0.2 * radius, 0.3 * radius});
  const RigidTransformd X_WC =
      RigidTransformd(Vector3d{0.2 * radius, 0.2 * radius, 0.3 * radius});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const RigidTransformd X_AC = X_WA.InvertAndCompose(X_WC);
  const RigidTransformd X_BC = X_WB.InvertAndCompose(X_WC);

  // Compute the candidate tets.
  std::vector<std::pair<int, int>> candidate_tetrahedra_AB;
  const auto callback_AB = [&candidate_tetrahedra_AB, &soft_geometryA,
                            &soft_geometryB](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet1);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_B || max_B < min_A))
      candidate_tetrahedra_AB.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereA.Collide(bvhSphereB, X_AB, callback_AB);

  std::vector<std::pair<int, int>> candidate_tetrahedra_AC;
  const auto callback_AC = [&candidate_tetrahedra_AC, &soft_geometryA,
                            &soft_geometryC](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_C = soft_geometryC.pressure_field().EvaluateMin(tet1);
    const double max_C = soft_geometryC.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_C || max_C < min_A))
      candidate_tetrahedra_AC.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereA.Collide(bvhSphereC, X_AC, callback_AC);

  std::vector<std::pair<int, int>> candidate_tetrahedra_BC;
  const auto callback_BC = [&candidate_tetrahedra_BC, &soft_geometryB,
                            &soft_geometryC](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet0);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet0);
    const double min_C = soft_geometryC.pressure_field().EvaluateMin(tet1);
    const double max_C = soft_geometryC.pressure_field().EvaluateMax(tet1);
    if (!(max_B < min_C || max_C < min_B))
      candidate_tetrahedra_BC.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereB.Collide(bvhSphereC, X_BC, callback_BC);

  // Convert cadidate tets to collision_filter_ that can be compared to one
  // from sycl_proximity_engine
  const int num_A = soft_geometryA.mesh().num_elements();
  const int num_B = soft_geometryB.mesh().num_elements();
  const int num_C = soft_geometryC.mesh().num_elements();
  // fmt::print("Number of elements in geometry A: {}\n", num_A);
  // fmt::print("Number of elements in geometry B: {}\n", num_B);
  // fmt::print("Number of elements in geometry C: {}\n", num_C);

  // Expected candidate tetrahedra
  std::unordered_map<SortedPair<GeometryId>, std::vector<std::pair<int, int>>>
      expected_candidate_tetrahedra;
  for (auto [eA, eB] : candidate_tetrahedra_AB) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereA_id,
                                                         sphereB_id)]
        .emplace_back(eA, eB + num_A);
  }
  for (auto [eA, eC] : candidate_tetrahedra_AC) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereA_id,
                                                         sphereC_id)]
        .emplace_back(eA, eC + num_A + num_B);
  }
  for (auto [eB, eC] : candidate_tetrahedra_BC) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereB_id,
                                                         sphereC_id)]
        .emplace_back(eB + num_A, eC + num_A + num_B);
  }

  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(soft_geometryA, X_WA));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(soft_geometryB, X_WB));
  auto [minC, maxC] = ComputeTotalBounds(TransformMesh(soft_geometryC, X_WC));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {sphereA_id, soft_geometryA},
      {sphereB_id, soft_geometryB},
      {sphereC_id, soft_geometryC}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{
      {sphereA_id, minA}, {sphereB_id, minB}, {sphereC_id, minC}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{
      {sphereA_id, maxA}, {sphereB_id, maxB}, {sphereC_id, maxC}};
  // Instantiate SyclProximityEngine to obtain collision filter
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);

  // Update collision candidates
  std::vector<SortedPair<GeometryId>> collision_candidates{
      SortedPair<GeometryId>(sphereA_id, sphereB_id),
      SortedPair<GeometryId>(sphereA_id, sphereC_id),
      SortedPair<GeometryId>(sphereB_id, sphereC_id)};
  engine.UpdateCollisionCandidates(collision_candidates);

  // Move spheres closer so that they collide
  const std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {sphereA_id, X_WA}, {sphereB_id, X_WB}, {sphereC_id, X_WC}};
  const auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks
  const auto impl = SyclProximityEngineAttorney::get_impl(engine);

  auto collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  // Package obtained collisions as pair vector
  std::vector<std::pair<int, int>> obtained_collisions;
  std::vector<std::pair<int, int>> mismatch_pairs;
  // Compare obtained collision pairs with expected from Drake
  for (auto candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    auto ExpMeshA_indices = expected_candidate_tetrahedra[candidate];
    fmt::print("Obtained collisions {} for geometry {} and {}\n",
               cc.total_collisions, candidate.first(), candidate.second());
    fmt::print("Expected collisions {} for geometry {} and {}\n",
               ExpMeshA_indices.size(), candidate.first(), candidate.second());
    // For each candidate on the GPU, we look for the corresponding
    // candidate on the CPU. We should expect to find all.
    for (uint32_t i = 0; i < cc.total_collisions; ++i) {
      obtained_collisions.emplace_back(
          std::make_pair(static_cast<int>(ci.collision_indices_A[i]),
                         static_cast<int>(ci.collision_indices_B[i])));
      auto it = std::find(
          ExpMeshA_indices.begin(), ExpMeshA_indices.end(),
          std::make_pair(static_cast<int>(ci.collision_indices_A[i]),
                         static_cast<int>(ci.collision_indices_B[i])));
      EXPECT_NE(it, ExpMeshA_indices.end());
    }
    // Inverse check: Find all the candidates found on the CPU but not on the
    // GPU
    for (auto [eA, eB] : ExpMeshA_indices) {
      auto it = std::find(obtained_collisions.begin(),
                          obtained_collisions.end(), std::make_pair(eA, eB));
      if (it == obtained_collisions.end()) {
        mismatch_pairs.emplace_back(std::make_pair(eA, eB));
      }
    }
  }

  // Tree stats

  // auto host_indices_all = SyclProximityEngineAttorney::GetHostIndicesAll(
  //     bvh_data, host_mesh_data.total_elements, mem_mgr, q_device);
  // for (uint32_t i = 0; i < SyclProximityEngineAttorney::get_num_meshes(impl);
  //      ++i) {
  //   fmt::print("Mesh {}\n", i);
  //   const auto host_bvh = SyclProximityEngineAttorney::get_host_bvh(impl, i);
  //   std::string filepath = fmt::format("histogram_{}.json", i);
  //   const auto [height, num_leaves, balance_factor, average_depth,
  //               bounds_valid] = CheckSyclBvhProperties(host_bvh, filepath);
  //   fmt::print(
  //       "Mesh {}: height: {}, num_leaves: {}, balance_factor: {}, "
  //       "average_depth: {}, bounds_valid: {}\n",
  //       i, height, num_leaves, balance_factor, average_depth, bounds_valid);
  // }

  // fmt::print("Tree AABBs of all internal nodes\n");

  // // Tree AABBs of all internal nodes
  // for (uint32_t i = 0; i < SyclProximityEngineAttorney::get_num_meshes(impl);
  //      ++i) {
  //   fmt::print("Mesh {}\n", i);
  //   const auto host_bvh = SyclProximityEngineAttorney::get_host_bvh(impl, i);
  //   SyclProximityEngineAttorney::PrintNodeBoundingBoxes(host_bvh,
  //                                                       host_indices_all, i);
  // }

  // Verify that each of the mismatches is TRULY a false positive from the CPU
  // broadphase. To do this, we compute the Aabb's of the element pairs in the
  // world frame, and then compute the intersection of those Aabb's and verify
  // that each intersection is empty.
  const auto CalcAabb = [](const Vector3d& a, const Vector3d& b,
                           const Vector3d& c, const Vector3d& d) {
    Vector3d min = a;
    Vector3d max = a;
    min = min.cwiseMin(b);
    max = max.cwiseMax(b);
    min = min.cwiseMin(c);
    max = max.cwiseMax(c);
    min = min.cwiseMin(d);
    max = max.cwiseMax(d);
    return std::make_pair(min, max);
  };
  // Store the element counts in vector in order to identify which geometry the
  // element is from
  auto find_geometry_index = [](const int global_index,
                                const std::vector<int>& scan) {
    auto it = std::upper_bound(scan.begin(), scan.end(), global_index);
    return static_cast<int>(std::distance(scan.begin(), it)) - 1;
  };
  std::unordered_map<uint32_t, hydroelastic::SoftGeometry>
      index_to_soft_geometry{
          {0, soft_geometryA}, {1, soft_geometryB}, {2, soft_geometryC}};
  std::unordered_map<uint32_t, RigidTransformd> Transforms{
      {0, X_WA}, {1, X_WB}, {2, X_WC}};
  std::vector<int> element_counts_scan = {0, num_A, num_A + num_B};
  // Check mismatch pairs for false positives
  for (auto [eA, eB] : mismatch_pairs) {
    int geom_eA = find_geometry_index(eA, element_counts_scan);
    int geom_eB = find_geometry_index(eB, element_counts_scan);
    int local_eA = eA - element_counts_scan[geom_eA];
    int local_eB = eB - element_counts_scan[geom_eB];
    auto [minA, maxA] =
        CalcAabb(Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(0)),
                 Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(1)),
                 Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(2)),
                 Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(3)));

    auto [minB, maxB] =
        CalcAabb(Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(0)),
                 Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(1)),
                 Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(2)),
                 Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(3)));

    const Vector3d intersection_min = minA.cwiseMax(minB);
    const Vector3d intersection_max = maxA.cwiseMin(maxB);
    const Vector3d intersection_widths = intersection_max - intersection_min;
    EXPECT_LE(intersection_widths.minCoeff(), 0);
  }

  // Get polygon areas and centroids
  const std::vector<double> polygon_areas =
      SyclProximityEngineAttorney::get_polygon_areas(impl);
  const std::vector<Vector3d> polygon_centroids =
      SyclProximityEngineAttorney::get_polygon_centroids(impl);
}

GTEST_TEST(SPETest, FourSpheresColliding) {
  constexpr double radius = 0.5;
  constexpr double resolution_hint_A = 0.5 * radius;
  constexpr double resolution_hint_B = 0.5 * radius;
  constexpr double resolution_hint_C = 0.5 * radius;
  constexpr double resolution_hint_D = 0.5 * radius;
  constexpr double hydroelastic_modulus = 1e+7;

  // Sphere A
  const Sphere sphereA(radius);
  auto meshA =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereA, resolution_hint_A,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureA = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereA, meshA.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereA(*meshA);

  hydroelastic::SoftGeometry soft_geometryA(
      hydroelastic::SoftMesh(std::move(meshA), std::move(pressureA)));
  const GeometryId sphereA_id = GeometryId::get_new_id();

  // Sphere B
  const Sphere sphereB(radius);
  auto meshB =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereB, resolution_hint_B,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  // Sphere C
  const Sphere sphereC(radius);
  auto meshC =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereC, resolution_hint_C,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureC = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereC, meshC.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereC(*meshC);
  hydroelastic::SoftGeometry soft_geometryC(
      hydroelastic::SoftMesh(std::move(meshC), std::move(pressureC)));
  const GeometryId sphereC_id = GeometryId::get_new_id();

  // Sphere D
  const Sphere sphereD(radius);
  auto meshD =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereD, resolution_hint_D,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureD = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereD, meshD.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereD(*meshD);
  hydroelastic::SoftGeometry soft_geometryD(
      hydroelastic::SoftMesh(std::move(meshD), std::move(pressureD)));
  const GeometryId sphereD_id = GeometryId::get_new_id();

  // Arbitrarily pose the spheres into a colliding configuration.
  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.2 * radius, 0.1 * radius, 0.3 * radius});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{0.1 * radius, 0.2 * radius, 0.3 * radius});
  const RigidTransformd X_WC =
      RigidTransformd(Vector3d{0.2 * radius, 0.2 * radius, 0.3 * radius});
  const RigidTransformd X_WD =
      RigidTransformd(Vector3d{0.15 * radius, 0.15 * radius, 0.3 * radius});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const RigidTransformd X_AC = X_WA.InvertAndCompose(X_WC);
  const RigidTransformd X_AD = X_WA.InvertAndCompose(X_WD);
  const RigidTransformd X_BC = X_WB.InvertAndCompose(X_WC);
  const RigidTransformd X_BD = X_WB.InvertAndCompose(X_WD);
  const RigidTransformd X_CD = X_WC.InvertAndCompose(X_WD);

  // Compute the candidate tets for all pairs.
  std::vector<std::pair<int, int>> candidate_tetrahedra_AB;
  const auto callback_AB = [&candidate_tetrahedra_AB, &soft_geometryA,
                            &soft_geometryB](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet1);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_B || max_B < min_A))
      candidate_tetrahedra_AB.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereA.Collide(bvhSphereB, X_AB, callback_AB);

  std::vector<std::pair<int, int>> candidate_tetrahedra_AC;
  const auto callback_AC = [&candidate_tetrahedra_AC, &soft_geometryA,
                            &soft_geometryC](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_C = soft_geometryC.pressure_field().EvaluateMin(tet1);
    const double max_C = soft_geometryC.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_C || max_C < min_A))
      candidate_tetrahedra_AC.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereA.Collide(bvhSphereC, X_AC, callback_AC);

  std::vector<std::pair<int, int>> candidate_tetrahedra_AD;
  const auto callback_AD = [&candidate_tetrahedra_AD, &soft_geometryA,
                            &soft_geometryD](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_D = soft_geometryD.pressure_field().EvaluateMin(tet1);
    const double max_D = soft_geometryD.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_D || max_D < min_A))
      candidate_tetrahedra_AD.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereA.Collide(bvhSphereD, X_AD, callback_AD);

  std::vector<std::pair<int, int>> candidate_tetrahedra_BC;
  const auto callback_BC = [&candidate_tetrahedra_BC, &soft_geometryB,
                            &soft_geometryC](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet0);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet0);
    const double min_C = soft_geometryC.pressure_field().EvaluateMin(tet1);
    const double max_C = soft_geometryC.pressure_field().EvaluateMax(tet1);
    if (!(max_B < min_C || max_C < min_B))
      candidate_tetrahedra_BC.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereB.Collide(bvhSphereC, X_BC, callback_BC);

  std::vector<std::pair<int, int>> candidate_tetrahedra_BD;
  const auto callback_BD = [&candidate_tetrahedra_BD, &soft_geometryB,
                            &soft_geometryD](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet0);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet0);
    const double min_D = soft_geometryD.pressure_field().EvaluateMin(tet1);
    const double max_D = soft_geometryD.pressure_field().EvaluateMax(tet1);
    if (!(max_B < min_D || max_D < min_B))
      candidate_tetrahedra_BD.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereB.Collide(bvhSphereD, X_BD, callback_BD);

  std::vector<std::pair<int, int>> candidate_tetrahedra_CD;
  const auto callback_CD = [&candidate_tetrahedra_CD, &soft_geometryC,
                            &soft_geometryD](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_C = soft_geometryC.pressure_field().EvaluateMin(tet0);
    const double max_C = soft_geometryC.pressure_field().EvaluateMax(tet0);
    const double min_D = soft_geometryD.pressure_field().EvaluateMin(tet1);
    const double max_D = soft_geometryD.pressure_field().EvaluateMax(tet1);
    if (!(max_C < min_D || max_D < min_C))
      candidate_tetrahedra_CD.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereC.Collide(bvhSphereD, X_CD, callback_CD);

  // Convert candidate tets to collision_filter_ that can be compared to one
  // from sycl_proximity_engine
  const int num_A = soft_geometryA.mesh().num_elements();
  const int num_B = soft_geometryB.mesh().num_elements();
  const int num_C = soft_geometryC.mesh().num_elements();
  const int num_D = soft_geometryD.mesh().num_elements();

  // Expected candidate tetrahedra
  std::unordered_map<SortedPair<GeometryId>, std::vector<std::pair<int, int>>>
      expected_candidate_tetrahedra;
  for (auto [eA, eB] : candidate_tetrahedra_AB) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereA_id,
                                                         sphereB_id)]
        .emplace_back(eA, eB + num_A);
  }
  for (auto [eA, eC] : candidate_tetrahedra_AC) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereA_id,
                                                         sphereC_id)]
        .emplace_back(eA, eC + num_A + num_B);
  }
  for (auto [eA, eD] : candidate_tetrahedra_AD) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereA_id,
                                                         sphereD_id)]
        .emplace_back(eA, eD + num_A + num_B + num_C);
  }
  for (auto [eB, eC] : candidate_tetrahedra_BC) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereB_id,
                                                         sphereC_id)]
        .emplace_back(eB + num_A, eC + num_A + num_B);
  }

  for (auto [eB, eD] : candidate_tetrahedra_BD) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereB_id,
                                                         sphereD_id)]
        .emplace_back(eB + num_A, eD + num_A + num_B + num_C);
  }
  for (auto [eC, eD] : candidate_tetrahedra_CD) {
    expected_candidate_tetrahedra[SortedPair<GeometryId>(sphereC_id,
                                                         sphereD_id)]
        .emplace_back(eC + num_A + num_B, eD + num_A + num_B + num_C);
  }

  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(soft_geometryA, X_WA));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(soft_geometryB, X_WB));
  auto [minC, maxC] = ComputeTotalBounds(TransformMesh(soft_geometryC, X_WC));
  auto [minD, maxD] = ComputeTotalBounds(TransformMesh(soft_geometryD, X_WD));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {sphereA_id, soft_geometryA},
      {sphereB_id, soft_geometryB},
      {sphereC_id, soft_geometryC},
      {sphereD_id, soft_geometryD}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{
      {sphereA_id, minA},
      {sphereB_id, minB},
      {sphereC_id, minC},
      {sphereD_id, minD}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{
      {sphereA_id, maxA},
      {sphereB_id, maxB},
      {sphereC_id, maxC},
      {sphereD_id, maxD}};

  // Instantiate SyclProximityEngine to obtain collision filter
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);

  // Move spheres closer so that they collide
  const std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {sphereA_id, X_WA},
      {sphereB_id, X_WB},
      {sphereC_id, X_WC},
      {sphereD_id, X_WD}};
  const std::vector<SortedPair<GeometryId>> collision_candidates{
      SortedPair<GeometryId>(sphereA_id, sphereB_id),
      SortedPair<GeometryId>(sphereA_id, sphereC_id),
      SortedPair<GeometryId>(sphereA_id, sphereD_id),
      SortedPair<GeometryId>(sphereB_id, sphereC_id),
      SortedPair<GeometryId>(sphereB_id, sphereD_id),
      SortedPair<GeometryId>(sphereC_id, sphereD_id)};
  engine.UpdateCollisionCandidates(collision_candidates);
  const auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks
  const auto impl = SyclProximityEngineAttorney::get_impl(engine);

  auto collision_candidates_to_data =
      SyclProximityEngineAttorney::get_collision_candidates_to_data(impl);
  // Package obtained collisions as pair vector
  std::vector<std::pair<int, int>> obtained_collisions;
  std::vector<std::pair<int, int>> mismatch_pairs;
  // Compare obtained collision pairs with expected from Drake
  for (auto candidate : collision_candidates) {
    auto [cc, ci] = collision_candidates_to_data[candidate];
    auto ExpMeshA_indices = expected_candidate_tetrahedra[candidate];
    fmt::print("Obtained collisions {} for geometry {} and {}\n",
               cc.total_collisions, candidate.first(), candidate.second());
    fmt::print("Expected collisions {} for geometry {} and {}\n",
               ExpMeshA_indices.size(), candidate.first(), candidate.second());
    // For each candidate on the GPU, we look for the corresponding
    // candidate on the CPU. We should expect to find all.
    for (uint32_t i = 0; i < cc.total_collisions; ++i) {
      obtained_collisions.emplace_back(
          std::make_pair(static_cast<int>(ci.collision_indices_A[i]),
                         static_cast<int>(ci.collision_indices_B[i])));
      auto it = std::find(
          ExpMeshA_indices.begin(), ExpMeshA_indices.end(),
          std::make_pair(static_cast<int>(ci.collision_indices_A[i]),
                         static_cast<int>(ci.collision_indices_B[i])));
      EXPECT_NE(it, ExpMeshA_indices.end());
    }
    // Inverse check: Find all the candidates found on the CPU but not on the
    // GPU
    for (auto [eA, eB] : ExpMeshA_indices) {
      auto it = std::find(obtained_collisions.begin(),
                          obtained_collisions.end(), std::make_pair(eA, eB));
      if (it == obtained_collisions.end()) {
        mismatch_pairs.emplace_back(std::make_pair(eA, eB));
      }
    }
  }

  const auto CalcAabb = [](const Vector3d& a, const Vector3d& b,
                           const Vector3d& c, const Vector3d& d) {
    Vector3d min = a;
    Vector3d max = a;
    min = min.cwiseMin(b);
    max = max.cwiseMax(b);
    min = min.cwiseMin(c);
    max = max.cwiseMax(c);
    min = min.cwiseMin(d);
    max = max.cwiseMax(d);
    return std::make_pair(min, max);
  };

  // Store the element counts in vector in order to identify which geometry the
  // element is from
  auto find_geometry_index = [](const int global_index,
                                const std::vector<int>& scan) {
    auto it = std::upper_bound(scan.begin(), scan.end(), global_index);
    return static_cast<int>(std::distance(scan.begin(), it)) - 1;
  };
  std::unordered_map<uint32_t, hydroelastic::SoftGeometry>
      index_to_soft_geometry{{0, soft_geometryA},
                             {1, soft_geometryB},
                             {2, soft_geometryC},
                             {3, soft_geometryD}};
  std::unordered_map<uint32_t, RigidTransformd> Transforms{
      {0, X_WA}, {1, X_WB}, {2, X_WC}, {3, X_WD}};
  std::vector<int> element_counts_scan = {0, num_A, num_A + num_B,
                                          num_A + num_B + num_C};
  // Check mismatch pairs for false positives
  for (auto [eA, eB] : mismatch_pairs) {
    int geom_eA = find_geometry_index(eA, element_counts_scan);
    int geom_eB = find_geometry_index(eB, element_counts_scan);
    int local_eA = eA - element_counts_scan[geom_eA];
    int local_eB = eB - element_counts_scan[geom_eB];
    auto [minA, maxA] =
        CalcAabb(Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(0)),
                 Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(1)),
                 Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(2)),
                 Transforms.at(geom_eA) *
                     index_to_soft_geometry.at(geom_eA).mesh().vertex(
                         index_to_soft_geometry.at(geom_eA)
                             .mesh()
                             .element(local_eA)
                             .vertex(3)));

    auto [minB, maxB] =
        CalcAabb(Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(0)),
                 Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(1)),
                 Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(2)),
                 Transforms.at(geom_eB) *
                     index_to_soft_geometry.at(geom_eB).mesh().vertex(
                         index_to_soft_geometry.at(geom_eB)
                             .mesh()
                             .element(local_eB)
                             .vertex(3)));

    const Vector3d intersection_min = minA.cwiseMax(minB);
    const Vector3d intersection_max = maxA.cwiseMin(maxB);
    const Vector3d intersection_widths = intersection_max - intersection_min;
    EXPECT_LE(intersection_widths.minCoeff(), 0);
  }
}

// Tests the efficiency and correctness of the BVH trees constructed by the
// SYCL proximity engine
GTEST_TEST(SPEBvhTest, TwoSpheresTreeStats) {
  constexpr double radiusA = 1.0;
  constexpr double resolution_hintA = 0.2 * radiusA;
  constexpr double radiusB = 0.5;
  constexpr double resolution_hintB = 0.5 * radiusB;
  constexpr double hydroelastic_modulus = 1e+7;

  // Sphere A
  const Sphere sphereA(radiusA);
  auto meshA =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereA, resolution_hintA,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureA = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereA, meshA.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereA(*meshA);

  const hydroelastic::SoftGeometry soft_geometryA(
      hydroelastic::SoftMesh(std::move(meshA), std::move(pressureA)));
  const GeometryId sphereA_id = GeometryId::get_new_id();

  // Sphere B
  const Sphere sphereB(radiusB);
  auto meshB =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereB, resolution_hintB,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  const hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  const int num_A = soft_geometryA.mesh().num_elements();
  const int num_B = soft_geometryB.mesh().num_elements();
  fmt::print("Number of elements in sphere A: {}, sphere B: {}\n", num_A,
             num_B);

  // Arbitrarily pose the spheres into a colliding configuration.
  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.0 * radiusA, 0.0 * radiusA, 0.3 * radiusA});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{1.0 * radiusB, 0.0 * radiusB, 0.3 * radiusB});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);

  // Create inputs to SyclProximityEngine
  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(soft_geometryA, X_WA));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(soft_geometryB, X_WB));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {sphereA_id, soft_geometryA}, {sphereB_id, soft_geometryB}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{
      {sphereA_id, minA}, {sphereB_id, minB}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{
      {sphereA_id, maxA}, {sphereB_id, maxB}};

  // Instantiate SyclProximityEngine to initialize memory for the GPU
  // datastructures
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);

  // Move spheres closer so they collide
  const std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {sphereA_id, X_WA}, {sphereB_id, X_WB}};
  engine.UpdateCollisionCandidates(
      {SortedPair<GeometryId>(sphereA_id, sphereB_id)});
  const auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Extract Impl from the engine in order to get access to the private data
  // members
  const auto impl = SyclProximityEngineAttorney::get_impl(engine);
  // Check BVH properties for both spheres using the helper
  for (uint32_t i = 0; i < SyclProximityEngineAttorney::get_num_meshes(impl);
       ++i) {
    const auto host_bvh = SyclProximityEngineAttorney::get_host_bvh(impl, i);
    std::string filepath = fmt::format("histogram_{}.json", i);
    const auto [height, num_leaves, balance_factor, average_depth,
                bounds_valid] = CheckSyclBvhProperties(host_bvh, filepath);
    fmt::print(
        "Mesh {}: height: {}, num_leaves: {}, balance_factor: {}, "
        "average_depth: {}, bounds_valid: {}\n",
        i, height, num_leaves, balance_factor, average_depth, bounds_valid);
  }
}

GTEST_TEST(SPEBvhTest, ThreeSpheresTreeStats) {
  constexpr double radiusA = 1.0;
  constexpr double resolution_hintA = 0.2 * radiusA;
  constexpr double radiusB = 0.5;
  constexpr double resolution_hintB = 0.5 * radiusB;
  constexpr double radiusC = 1.5;
  constexpr double resolution_hintC = 0.1 * radiusC;
  constexpr double hydroelastic_modulus = 1e+7;

  // Sphere A
  const Sphere sphereA(radiusA);
  auto meshA =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereA, resolution_hintA,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureA = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereA, meshA.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereA(*meshA);

  const hydroelastic::SoftGeometry soft_geometryA(
      hydroelastic::SoftMesh(std::move(meshA), std::move(pressureA)));
  const GeometryId sphereA_id = GeometryId::get_new_id();

  // Sphere B
  const Sphere sphereB(radiusB);
  auto meshB =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereB, resolution_hintB,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  const hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  // Sphere C
  const Sphere sphereC(radiusC);
  auto meshC =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereC, resolution_hintC,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureC = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereC, meshC.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereC(*meshC);
  const hydroelastic::SoftGeometry soft_geometryC(
      hydroelastic::SoftMesh(std::move(meshC), std::move(pressureC)));
  const GeometryId sphereC_id = GeometryId::get_new_id();

  const int num_A = soft_geometryA.mesh().num_elements();
  const int num_B = soft_geometryB.mesh().num_elements();
  const int num_C = soft_geometryC.mesh().num_elements();
  fmt::print("Number of elements in sphere A: {}, sphere B: {}, sphere C: {}\n",
             num_A, num_B, num_C);

  // Arbitrarily pose the spheres into a colliding configuration.
  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.0 * radiusA, 0.0 * radiusA, 0.3 * radiusA});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{1.0 * radiusB, 0.0 * radiusB, 0.3 * radiusB});
  const RigidTransformd X_WC =
      RigidTransformd(Vector3d{0.0 * radiusC, 1.0 * radiusC, 0.3 * radiusC});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const RigidTransformd X_AC = X_WA.InvertAndCompose(X_WC);
  const RigidTransformd X_BC = X_WB.InvertAndCompose(X_WC);

  // Create inputs to SyclProximityEngine
  auto [minA, maxA] = ComputeTotalBounds(TransformMesh(soft_geometryA, X_WA));
  auto [minB, maxB] = ComputeTotalBounds(TransformMesh(soft_geometryB, X_WB));
  auto [minC, maxC] = ComputeTotalBounds(TransformMesh(soft_geometryC, X_WC));
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {sphereA_id, soft_geometryA},
      {sphereB_id, soft_geometryB},
      {sphereC_id, soft_geometryC}};
  std::unordered_map<GeometryId, Vector3<double>> total_lower{
      {sphereA_id, minA}, {sphereB_id, minB}, {sphereC_id, minC}};
  std::unordered_map<GeometryId, Vector3<double>> total_upper{
      {sphereA_id, maxA}, {sphereB_id, maxB}, {sphereC_id, maxC}};

  // Instantiate SyclProximityEngine to initialize memory for the GPU
  // datastructures
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries, total_lower, total_upper);

  // Move spheres closer so they collide
  const std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {sphereA_id, X_WA}, {sphereB_id, X_WB}, {sphereC_id, X_WC}};
  engine.UpdateCollisionCandidates(
      {SortedPair<GeometryId>(sphereA_id, sphereB_id),
       SortedPair<GeometryId>(sphereA_id, sphereC_id),
       SortedPair<GeometryId>(sphereB_id, sphereC_id)});
  const auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Extract Impl from the engine in order to get access to the private data
  // members
  const auto impl = SyclProximityEngineAttorney::get_impl(engine);
  // Check BVH properties for all three spheres using the helper
  for (uint32_t i = 0; i < SyclProximityEngineAttorney::get_num_meshes(impl);
       ++i) {
    const auto host_bvh = SyclProximityEngineAttorney::get_host_bvh(impl, i);
    std::string filepath = fmt::format("histogram_{}.json", i);
    const auto [height, num_leaves, balance_factor, average_depth,
                bounds_valid] = CheckSyclBvhProperties(host_bvh, filepath);
    fmt::print(
        "Mesh {}: height: {}, num_leaves: {}, balance_factor: {}, "
        "average_depth: {}, bounds_valid: {}\n",
        i, height, num_leaves, balance_factor, average_depth, bounds_valid);
  }
}

}  // namespace
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
