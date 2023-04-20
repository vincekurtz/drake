#pragma once

#include <vector>
#include <iostream> //DEBUG TODO

#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace traj_opt {

using multibody::MultibodyForces;
using multibody::MultibodyPlant;

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
    std::cout << "size: " << size_ << std::endl;
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
            1, std::vector<VectorX<T>>(num_steps, VectorX<T>(nq_))),
        multibody_forces_workspace_(1, MultibodyForces<T>(plant)) {}

  // Get/release a reference to an Eigen vector of size nq
  VectorX<T>& get_q_size_tmp() {
    return q_size_workspace_.get();
  }
  void release_q_size_tmp(const VectorX<T>& var) {
    q_size_workspace_.release(var);
  }

  // Get/release a reference to an Eigen vector of size nv
  VectorX<T>& get_v_size_tmp() {
    return v_size_workspace_.get();
  }
  void release_v_size_tmp(const VectorX<T>& var) {
    v_size_workspace_.release(var);
  }

  // Get/release a reference to an Eigen vector the same size as all the variables
  VectorX<T>& get_num_vars_size_tmp() {
    return num_vars_size_workspace_.get();
  }
  void release_num_vars_size_tmp(const VectorX<T>& var) {
    num_vars_size_workspace_.release(var);
  }

  // Get/release a reference to an Eigen Matrix the size of the equality constraint Jacobian
  MatrixX<T>& get_num_vars_times_num_eq_size_tmp() {
    return num_vars_times_num_eq_size_workspace_.get();
  }
  void release_num_vars_times_num_eq_size_tmp(const MatrixX<T>& var) {
    num_vars_times_num_eq_size_workspace_.release(var);
  }

  // Get/release a reference to a std::vector<Vector> the same size as the sequence of generalized positions
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
};

///**
// * A container for scratch variables that we use in various intermediate
// * computations. Allows us to avoid extra allocations when speed is important.
// */
//template <typename T>
//struct TrajectoryOptimizerWorkspace {
//  // Construct a workspace with size matching the given plant.
//  TrajectoryOptimizerWorkspace(const int num_steps,
//                               const MultibodyPlant<T>& plant)
//      : f_ext(plant) {
//    const int nq = plant.num_positions();
//    const int nv = plant.num_velocities();
//    const int num_vars = nq * (num_steps + 1);
//
//    // Get number of unactuated DoFs
//    int num_unactuated = 0;
//    const MatrixX<T> B = plant.MakeActuationMatrix();
//    if (B.size() > 0) {
//      // if B is of size zero, assume the system is fully actuated
//      for (int i = 0; i < nv; ++i) {
//        if (B.row(i).sum() == 0) {
//          ++num_unactuated;
//        }
//      }
//    }
//    const int num_eq_cons = num_unactuated * num_steps;
//
//    // Set vector sizes
//    q_size_tmp1.resize(nq);
//    q_size_tmp2.resize(nq);
//    q_size_tmp3.resize(nq);
//    q_size_tmp4.resize(nq);
//
//    v_size_tmp1.resize(nv);
//    v_size_tmp2.resize(nv);
//    v_size_tmp3.resize(nv);
//    v_size_tmp4.resize(nv);
//    v_size_tmp5.resize(nv);
//    v_size_tmp6.resize(nv);
//    v_size_tmp7.resize(nv);
//    v_size_tmp8.resize(nv);
//
//    tau_size_tmp1.resize(nv);
//    tau_size_tmp2.resize(nv);
//    tau_size_tmp3.resize(nv);
//    tau_size_tmp4.resize(nv);
//    tau_size_tmp5.resize(nv);
//    tau_size_tmp6.resize(nv);
//    tau_size_tmp7.resize(nv);
//    tau_size_tmp8.resize(nv);
//    tau_size_tmp9.resize(nv);
//    tau_size_tmp10.resize(nv);
//    tau_size_tmp11.resize(nv);
//    tau_size_tmp12.resize(nv);
//
//    a_size_tmp1.resize(nv);
//    a_size_tmp2.resize(nv);
//    a_size_tmp3.resize(nv);
//    a_size_tmp4.resize(nv);
//    a_size_tmp5.resize(nv);
//    a_size_tmp6.resize(nv);
//    a_size_tmp7.resize(nv);
//    a_size_tmp8.resize(nv);
//    a_size_tmp9.resize(nv);
//    a_size_tmp10.resize(nv);
//    a_size_tmp11.resize(nv);
//    a_size_tmp12.resize(nv);
//
//    num_vars_size_tmp1.resize(num_vars);
//    num_vars_size_tmp2.resize(num_vars);
//
//    num_vars_by_num_eq_cons_tmp.resize(num_vars, num_eq_cons);
//
//    // Allocate sequences
//    q_sequence_tmp1.assign(num_steps, VectorX<T>(nq));
//    q_sequence_tmp2.assign(num_steps, VectorX<T>(nq));
//  }
//
//  // Storage for multibody forces
//  MultibodyForces<T> f_ext;
//
//  // Storage of size nq
//  VectorX<T> q_size_tmp1;
//  VectorX<T> q_size_tmp2;
//  VectorX<T> q_size_tmp3;
//  VectorX<T> q_size_tmp4;
//
//  // Storage of size nv
//  // These are named v, tau, and a, but this distinction is just for
//  // convienience.
//  VectorX<T> v_size_tmp1;
//  VectorX<T> v_size_tmp2;
//  VectorX<T> v_size_tmp3;
//  VectorX<T> v_size_tmp4;
//  VectorX<T> v_size_tmp5;
//  VectorX<T> v_size_tmp6;
//  VectorX<T> v_size_tmp7;
//  VectorX<T> v_size_tmp8;
//
//  VectorX<T> tau_size_tmp1;
//  VectorX<T> tau_size_tmp2;
//  VectorX<T> tau_size_tmp3;
//  VectorX<T> tau_size_tmp4;
//  VectorX<T> tau_size_tmp5;
//  VectorX<T> tau_size_tmp6;
//  VectorX<T> tau_size_tmp7;
//  VectorX<T> tau_size_tmp8;
//  VectorX<T> tau_size_tmp9;
//  VectorX<T> tau_size_tmp10;
//  VectorX<T> tau_size_tmp11;
//  VectorX<T> tau_size_tmp12;
//
//  VectorX<T> a_size_tmp1;
//  VectorX<T> a_size_tmp2;
//  VectorX<T> a_size_tmp3;
//  VectorX<T> a_size_tmp4;
//  VectorX<T> a_size_tmp5;
//  VectorX<T> a_size_tmp6;
//  VectorX<T> a_size_tmp7;
//  VectorX<T> a_size_tmp8;
//  VectorX<T> a_size_tmp9;
//  VectorX<T> a_size_tmp10;
//  VectorX<T> a_size_tmp11;
//  VectorX<T> a_size_tmp12;
//
//  // Storage of sequence of q
//  std::vector<VectorX<T>> q_sequence_tmp1;
//  std::vector<VectorX<T>> q_sequence_tmp2;
//
//  // Vector of all decision variables
//  VectorX<T> num_vars_size_tmp1;
//  VectorX<T> num_vars_size_tmp2;
//
//  // Matrix of size (number of variables) * (number of equality constraints)
//  MatrixX<T> num_vars_by_num_eq_cons_tmp;
//};

}  // namespace traj_opt
}  // namespace drake
