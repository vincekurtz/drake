#pragma once

#include "drake/common/eigen_types.h"
#include "drake/geometry/query_results/signed_distance_pair.h"

namespace drake {
namespace traj_opt {

/**
 * A simple struct to hold contact force data, including contributions in the
 * normal and tangential directions, and a SignedDistancePair containing
 * information about which bodies are in contact.
 */
template <typename T>
struct ContactForceData {
  // Contact forces in the normal direction
  T fn = 0;

  // Tangential component of contact forces
  Vector2<T> ft = Vector2<T>::Zero();

  // Signed distance pair for the bodies that are in contact
  geometry::SignedDistancePair<T> sdp;
};

}  // namespace traj_opt
}  // namespace drake
