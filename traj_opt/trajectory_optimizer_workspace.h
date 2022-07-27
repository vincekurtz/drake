#pragma once

#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace traj_opt {

using Eigen::VectorXd;
using multibody::MultibodyForces;
using multibody::MultibodyPlant;

/**
 * A container for scratch variables that we use in various intermediate
 * multibody dynamics computations. Allows us to avoid extra allocations when
 * speed is important.
 */
struct TrajectoryOptimizerWorkspace {
  // Construct a workspace who's size matches the given plant.
  explicit TrajectoryOptimizerWorkspace(const MultibodyPlant<double>& plant)
      : f_ext(plant) {
    a.setZero(plant.num_velocities());
  }

  // Generalized accelerations
  VectorXd a;

  // Storage for external forces, including gravity
  MultibodyForces<double> f_ext;
};

}  // namespace traj_opt
}  // namespace drake
