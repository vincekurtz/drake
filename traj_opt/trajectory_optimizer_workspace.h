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
 * computations. Allows us to avoid extra allocations when speed is important.
 */
struct TrajectoryOptimizerWorkspace {
  // Construct a workspace with size matching the given plant.
  explicit TrajectoryOptimizerWorkspace(const MultibodyPlant<double>& plant)
      : f_ext(plant) {
    const int nq = plant.num_positions();
    const int nv = plant.num_velocities();

    // Set vector sizes
    a.resize(nv);
    q_eps_t.resize(nq);
    v_eps_t.resize(nv);
    v_eps_tp.resize(nv);
    tau_eps_tm.resize(nv);
    tau_eps_t.resize(nv);
    tau_eps_tp.resize(nv);
  }

  // Generalized accelerations
  VectorXd a;

  // External forces, such as gravity
  MultibodyForces<double> f_ext;

  // Perturbed copies of q_t, v_t, v_{t+1}, tau_{t-1}, tau_t, and tau_{t+1}.
  // These are all of the quantities that change when we perturb q_t, and are
  // used to compute inverse dynamics partials with finite differences.
  VectorXd q_eps_t;
  VectorXd v_eps_t;
  VectorXd v_eps_tp;
  VectorXd tau_eps_tm;
  VectorXd tau_eps_t;
  VectorXd tau_eps_tp;
};

}  // namespace traj_opt
}  // namespace drake
