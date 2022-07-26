#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Struct containing the gradients of key dynamics terms that are used to
 * compute the (approximate) gradient and Hessian for our Gauss-Newton steps.
 */
struct GradientData {
  // Partials of tau_{t-1} ("tau_t minus") with respect to q_t,
  //
  //    d(tau_{t-1})/d(q_t)
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps.
  // For values where tau_{t-1} is undefined, we simply store zeros, i.e.,
  //
  //    [ 0, d(tau_0)/d(q_1), d(tau_1)/d(q_2), ... ,
  //                                     d(tau_{num_steps-1})/d(q_{num_steps})]
  std::vector<MatrixXd> dtaum_dq;

  // Partials of tau_t with respect to q_t,
  //
  //    d(tau_t)/d(q_t)
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps.
  // For values where tau_t is undefined, we simply store zeros, i.e.,
  //
  //    [ d(tau_0)/d(q_0), d(tau_1)/d(q_1), ... ,
  //                                d(tau_{num_steps-1})/d(q_{num_steps-1}), 0]
  std::vector<MatrixXd> dtau_dq;

  // Partial of tau_{t+1} ("tau_t plus") with respect to q_t,
  //
  //    d(tau_{t+1})/d(q_t)
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps.
  // For values where tau_{t+1} is undefined, we simply store zeros, i.e.,
  //
  //    [ d(tau_1)/d(q_0), d(tau_2)/d(q_1), ... ,
  //                              d(tau_{num_steps-1})/d(q_{num_steps-2}), 0, 0]
  std::vector<MatrixXd> dtaup_dq;
};

}  // namespace traj_opt
}  // namespace drake