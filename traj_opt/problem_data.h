#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Struct containing everything that specifies a system trajectory, including
 * state (q, v) and generalized forces (tau) for each timestep.
 */
struct TrajectoryData {
  // Generalized positions at each timestep.
  std::vector<VectorXd> q;

  // Generalized velocities at each timestep.
  std::vector<VectorXd> v;

  // Generalized forces at each timestep.
  // Note that these are different from control inputs u, since
  // tau may include nonzero (ficticious) forces on unactuated DoFs.
  // (At optimality we hope to constrain tau = B*u.)
  std::vector<VectorXd> tau;
};

/**
 * Struct containing dynamics terms, like the mass matrix and Coriolis terms.
 */
struct DynamicsData {
  // The mass matrix M(q_t) for each timestep.
  std::vector<MatrixXd> M;

  // The damping matrix D. This is constant over time.
  MatrixXd D;

  // The collection of nonlinear Coriolis and gravitational terms k(q_t,v_t) for
  // each timestep.
  std::vector<MatrixXd> k;

  // The contact/constraint Jacobian J(q_t) for each timestep.
  // Its size will change over time and as the trajectory changes.
  std::vector<MatrixXd> J;

  // The contact impulses gamma(v_{t+1}, q_t) for each timestep.
  // Its size will change over time and as the trajectory changes.
  std::vector<VectorXd> gamma;

  // The matrix mapping v to qdot for each timestep:
  //   qdot_t = N(q_t) * v_t
  std::vector<MatrixXd> N;

  // The matrix mapping qdot to v for each timestep:
  //   v_t = N_plus(q_t) * qdot_t
  std::vector<MatrixXd> N_plus;
};

/**
 * Struct containing the gradients of key dynamics terms that are used to
 * compute the (approximate) gradient and Hessian for our Gauss-Newton steps.
 */
struct GradientData {
  // Partial of v_t with respect to q_t,
  //    d(v_t)/d(q_t),
  // for each timestep.
  //
  // TODO(vincekurtz): these will be the identity matrix most of the time,
  // leading to lots of unnecessary multiplications with dense matrices. We
  // should replace these matrices with some operators like MultiplyByDvtDqt and
  // PreMultiplyByDvtDqt to avoid such extra computations.
  std::vector<MatrixXd> dvt_dqt;

  // Partial of v_{t+1} ("v_t plus") with respect to q_t,
  //    d(v_{t+1})/d(q_t),
  // for each timestep.
  //
  // TODO(vincekurtz): these will be the identity matrix most of the time,
  // leading to lots of unnecessary multiplications. We should replace these
  // matrices with some operators like MultiplyByDvtpDqt and
  // PreMultiplyByDvtpDqt to avoid such extra computations.
  std::vector<MatrixXd> dvtp_dqt;

  // Partial of tau_{t-1} ("tau_t minus") with respect to q_t,
  //    d(tau_{t-1})/d(q_t)
  // for each timestep.
  std::vector<MatrixXd> dtautm_dqt;

  // Partial of tau_t with respect to q_t,
  //    d(tau_t)/d(q_t)
  // for each timestep.
  std::vector<MatrixXd> dtaut_dqt;

  // Partial of tau_{t+1} ("tau_t plus") with respect to q_t,
  //    d(tau_{t+1})/d(q_t)
  // for each timestep.
  std::vector<MatrixXd> dtautp_dqt;
};

/**
 * Struct containing everything we need to compute the gradient and Hessian for our optimization problem. 
 */
struct ProblemData {
  // Container for the trajectory (q, v, tau)
  TrajectoryData traj_data;

  // Container for dynamics terms (M, D, k, J, gamma, N, N+)
  DynamicsData dyn_data;

  // Container for gradient terms (dv/dq, dtau/dq)
  GradientData grad_data;
};

}  // namespace traj_opt
}  // namespace drake
