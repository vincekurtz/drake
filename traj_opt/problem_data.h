#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Struct storing gradients of generalized velocities (v) with respect to
 * generalized positions (q).
 *
 * TODO(vincekurtz): extend to quaternion DoFs, where these quantities are
 * different for each timestep, and include a factor of N+(q).
 */
struct VelocityPartials {
  double dvt_dqt;
  double dvt_dqm;
};

/**
 * Struct containing the gradients of generalized forces (tau) with respect to
 * generalized positions (q).
 *
 * This is essentially a tri-diagonal matrix, since
 * tau_t is a function of q at times t-1, t, and t+1.
 */
template <typename T>
struct InverseDynamicsPartials {
  /**
   * Constructor which allocates variables of the proper sizes.
   *
   * @param num_steps number of time steps in the optimization problem
   * @param nv number of generalized velocities (size of tau and v)
   * @param nq number of generalized positions (size of q)
   */
  InverseDynamicsPartials(const int num_steps, const int nv, const int nq) {
    dtau_dqm.assign(num_steps, MatrixX<T>(nv, nq));
    dtau_dqt.assign(num_steps, MatrixX<T>(nv, nq));
    dtau_dqp.assign(num_steps, MatrixX<T>(nv, nq));

    // Set all derivatives w.r.t q(-1) to NaN
    dtau_dqm[0].setConstant(nv, nq, NAN);
  }

  // Return the number of steps allocated in this object.
  int size() const { return dtau_dqt.size(); }

  // Partials of tau_t with respect to q_{t-1} ("q_t minus"),
  //
  //    d(tau_t)/d(q_{t-1})
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps-1.
  // For t=0 we store NaN, since q_{t-1} is undefined, i.e.,
  //
  //    [ NaN, d(tau_1)/d(q_0) , d(tau_2)/d(q_1), ... ,
  //                                  d(tau_{num_steps-1})/d(q_{num_steps-2})]
  //
  std::vector<MatrixX<T>> dtau_dqm;

  // Partials of tau_t with respect to q_t,
  //
  //    d(tau_t)/d(q_t)
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps-1.
  // For values where tau_t or q_t is undefined or constant, we simply store
  // zeros, i.e.,
  //
  //    [ d(tau_0)/d(q_0), d(tau_1)/d(q_1), ... ,
  //                                    d(tau_{num_steps-1})/d(q_{num_steps-1})]
  //
  std::vector<MatrixX<T>> dtau_dqt;

  // Partial of tau_t with respect to q_{t+1} ("q_t plus"),
  //
  //    d(tau_t)/d(q_{t+1})
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps-1.
  // For values where tau_t or q_{t+1} is undefined or constant, we simply store
  // zeros, i.e.,
  //
  //    [ d(tau_0)/d(q_1), d(tau_1)/d(q_2), ... ,
  //                                     d(tau_{num_steps-1})/d(q_{num_steps})]
  //
  std::vector<MatrixX<T>> dtau_dqp;
};

struct TrajectoryOptimizerCache {
  TrajectoryOptimizerCache(const int num_steps, const int nv, const int nq)
      : id_partials(num_steps, nv, nq) {
    v.assign(num_steps + 1, VectorXd(nv));
    a.assign(num_steps, VectorXd(nv));
    tau.assign(num_steps, VectorXd(nv));
  }
  // Generalized velocities at each timestep
  // [v(0), v(1), ..., v(num_steps)]
  std::vector<VectorXd> v;

  // Generalized accelerations at each timestep
  // [a(0), a(1), ..., a(num_steps-1)]
  std::vector<VectorXd> a;

  // Generalized forces at each timestep
  // [tau(0), tau(1), ..., tau(num_steps-1)]
  std::vector<VectorXd> tau;

  // Storage for dv(t)/dq(t) and dv(t)/dq(t-1)
  VelocityPartials v_partials;

  // Storage for dtau(t)/dq(t-1), dtau(t)/dq(t), and dtau(t)/dq(t+1)
  InverseDynamicsPartials<double> id_partials;
};

/**
 * Struct for storing the "state" of the trajectory optimizer.
 *
 * The only actual state is the sequence of generalized positions q at each
 * timestep. This class stores that directly, but also a "cache" of other values
 * computed from q, such as generalized velocities and forces at each timesteps,
 * relevant dynamics partials, etc.
 */
struct TrajectoryOptimizerState {
  /**
   * Constructor which allocates things of the proper sizes.
   *
   * @param num_steps number of timesteps in the optimization problem
   * @param nv number of multibody velocities
   * @param nq number of multipody positions
   */
  TrajectoryOptimizerState(const int num_steps, const int nv, const int nq)
      : cache(num_steps, nv, nq) {
    q.assign(num_steps, VectorXd(nq));
  }

  // Sequence of generalized velocities at each timestep,
  // [q(0), q(1), ..., q(num_steps)]
  // TODO(vincekurtz): consider storing as a single VectorXd for better memory
  // layout.
  std::vector<VectorXd> q;

  // Storage for all other quantities that are computed from q, and are useful
  // for our calculations
  TrajectoryOptimizerCache cache;
};

}  // namespace traj_opt
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
  class ::drake::traj_opt::InverseDynamicsPartials
)
