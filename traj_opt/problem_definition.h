#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * A struct for specifying the optimization problem
 *
 *    min x_err(T)'*Qf*x_err(T) + sum{ x_err(t)'*Q*x_err(t) + u(t)'*R*u(t) }
 *    s.t. x(0) = x0
 *         multibody dynamics with contact
 *
 *  where x(t) = [q(t); v(t)] and
 *  Q = [ Qq  0  ]
 *      [ 0   Qv ].
 */
struct ProblemDefinition {
  // Time horizon (number of steps) for the optimization problem
  int T;

  // Running cost coefficients for generalized positions
  MatrixXd Qq;

  // Running cost coefficients for generalized velocities
  MatrixXd Qv;

  // Terminal cost coefficients for generalized positions
  MatrixXd Qf_q;

  // Terminal cost coefficients for generalized velocities
  MatrixXd Qf_v;

  // Control cost coefficients
  MatrixXd R;

  // Target generalized positions
  VectorXd q_nom;

  // Target generalized velocities
  VectorXd v_nom;

  // Initial generalized positions
  VectorXd q0;

  // Initial generalized velocities
  VectorXd v0;
};

}  // namespace traj_opt
}  // namespace drake
