#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace traj_opt {

using multibody::MultibodyForces;
using multibody::MultibodyPlant;
using multibody::SpatialForce;
using multibody::SpatialVelocity;

/**
 * A simple workspace class that allows us to pre-allocate variables of type T.
 * Keeps track of which variables are in use and which are not.
 */
template <typename T>
class SimpleWorkspace {
 public:
  /**
   * Initialize a simple workspace storing variables of type T
   *
   * @param size number of variables to store
   * @param default_value default value to initialize the variables
   */
  SimpleWorkspace(const int size, const T default_value)
      : size_(size), vars_(size, default_value), in_use_(size, false) {}

  /**
   * Get a mutable reference to a variable from the workspace that is not
   * currently in use.
   *
   * @return T& the pre-allocated variable
   */
  T& get() {
    for (int i = 0; i < size_; ++i) {
      if (!in_use_[i]) {
        in_use_[i] = true;
        return vars_[i];
      }
    }
    throw std::runtime_error(
        "Out of workspace memory! Make sure you are calling release on "
        "workspace elements once you're finished with them. Otherwise, try "
        "initializing the workspace with a larger size.");
  }

  void release(const T& var) {
    for (int i = 0; i < size_; ++i) {
      if (&var == &vars_[i]) {
        in_use_[i] = false;
        return;
      }
    }
    throw std::runtime_error(
        "The given variable is not stored in this workspace.");
  }

 private:
  const int size_;            // number of variables we're storing
  std::vector<T> vars_;       // the stored variables
  std::vector<bool> in_use_;  // flag for which variables are available
};

template <typename T>
class TrajectoryOptimizerWorkspace {
 public:
  TrajectoryOptimizerWorkspace(const int num_steps,
                               const int num_equality_constraints,
                               const MultibodyPlant<T>& plant)
      : nq_(plant.num_positions()),
        nv_(plant.num_velocities()),
        num_vars_(nq_ * (num_steps + 1)),
        q_size_workspace_(4, VectorX<T>(nq_)),
        v_size_workspace_(32, VectorX<T>(nv_)),
        num_vars_size_workspace_(2, VectorX<T>(num_vars_)),
        num_vars_times_num_eq_size_workspace_(
            1, MatrixX<T>(num_vars_, num_equality_constraints)),
        q_sequence_workspace_(
            2, std::vector<VectorX<T>>(num_steps, VectorX<T>(nq_))),
        multibody_forces_workspace_(1, MultibodyForces<T>(plant)),
        vector3_workspace_(10, Vector3<T>()),
        spatial_velocity_workspace_(2, SpatialVelocity<T>()),
        spatial_force_workspace_(2, SpatialForce<T>()) {}

  // Get/release a reference to an Eigen vector of size nq
  VectorX<T>& get_q_size_tmp() { return q_size_workspace_.get(); }
  void release_q_size_tmp(const VectorX<T>& var) {
    q_size_workspace_.release(var);
  }

  // Get/release a reference to an Eigen vector of size nv
  VectorX<T>& get_v_size_tmp() { return v_size_workspace_.get(); }
  void release_v_size_tmp(const VectorX<T>& var) {
    v_size_workspace_.release(var);
  }

  // Get/release a reference to an Eigen vector the same size as all the
  // variables
  VectorX<T>& get_num_vars_size_tmp() { return num_vars_size_workspace_.get(); }
  void release_num_vars_size_tmp(const VectorX<T>& var) {
    num_vars_size_workspace_.release(var);
  }

  // Get/release a reference to an Eigen Matrix the size of the equality
  // constraint Jacobian
  MatrixX<T>& get_num_vars_times_num_eq_size_tmp() {
    return num_vars_times_num_eq_size_workspace_.get();
  }
  void release_num_vars_times_num_eq_size_tmp(const MatrixX<T>& var) {
    num_vars_times_num_eq_size_workspace_.release(var);
  }

  // Get/release a reference to a std::vector<Vector> the same size as the
  // sequence of generalized positions
  std::vector<VectorX<T>>& get_q_sequence_tmp() {
    return q_sequence_workspace_.get();
  }
  void release_q_sequence_tmp(const std::vector<VectorX<T>>& var) {
    q_sequence_workspace_.release(var);
  }

  // Get/release a reference to a MultibodyForces object
  MultibodyForces<T>& get_multibody_forces_tmp() {
    return multibody_forces_workspace_.get();
  }
  void release_multibody_forces_tmp(const MultibodyForces<T>& var) {
    multibody_forces_workspace_.release(var);
  }

  // Get/release a reference to a Vector3<T>
  Vector3<T>& get_vector3_tmp() {
    return vector3_workspace_.get();
  }
  void release_vector3_tmp(const Vector3<T>& var) {
    vector3_workspace_.release(var);
  }
  
  // Get/release a reference to a spatial velocity
  SpatialVelocity<T>& get_spatial_velocity_tmp() {
    return spatial_velocity_workspace_.get();
  }
  void release_spatial_velocity_tmp(const SpatialVelocity<T>& var) {
    spatial_velocity_workspace_.release(var);
  }
  
  // Get/release a reference to a spatial force
  SpatialForce<T>& get_spatial_force_tmp() {
    return spatial_force_workspace_.get();
  }
  void release_spatial_force_tmp(const SpatialForce<T>& var) {
    spatial_force_workspace_.release(var);
  }

 private:
  const int nq_;
  const int nv_;
  const int num_vars_;
  SimpleWorkspace<VectorX<T>> q_size_workspace_;
  SimpleWorkspace<VectorX<T>> v_size_workspace_;
  SimpleWorkspace<VectorX<T>> num_vars_size_workspace_;
  SimpleWorkspace<MatrixX<T>> num_vars_times_num_eq_size_workspace_;
  SimpleWorkspace<std::vector<VectorX<T>>> q_sequence_workspace_;
  SimpleWorkspace<MultibodyForces<T>> multibody_forces_workspace_;
  SimpleWorkspace<Vector3<T>> vector3_workspace_;
  SimpleWorkspace<SpatialVelocity<T>> spatial_velocity_workspace_;
  SimpleWorkspace<SpatialForce<T>> spatial_force_workspace_;
};

}  // namespace traj_opt
}  // namespace drake
