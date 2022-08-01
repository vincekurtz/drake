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
  VelocityPartials(const int num_steps, const int nv, const int nq) {
    dvt_dqt.assign(num_steps + 1, MatrixXd(nv, nq));
    dvt_dqm.assign(num_steps + 1, MatrixXd(nv, nq));

    // Derivatives w.r.t. q(-1) are undefined
    dvt_dqm[0].setConstant(nv, nq, NAN);
  }
  // Partials of v_t w.r.t. q_t at each time step:
  //
  //    [d(v_0)/d(q_0), d(v_1)/d(q_1), ... , d(v_{num_steps})/d(q_{num_steps}) ]
  //
  std::vector<MatrixXd> dvt_dqt;

  // Partials of v_t w.r.t. q_{t-1} at each time step:
  //
  //    [NaN, d(v_1)/d(q_0), ... , d(v_{num_steps})/d(q_{num_steps-1}) ]
  //
  std::vector<MatrixXd> dvt_dqm;
};

/**
 * Struct containing the gradients of generalized forces (tau) with respect to
 * generalized positions (q).
 *
 * This is essentially a tri-diagonal matrix, since
 * tau_t is a function of q at times t-1, t, and t+1.
 */
struct InverseDynamicsPartials {
  /**
   * Constructor which allocates variables of the proper sizes.
   *
   * @param num_steps number of time steps in the optimization problem
   * @param nv number of generalized velocities (size of tau and v)
   * @param nq number of generalized positions (size of q)
   */
  InverseDynamicsPartials(const int num_steps, const int nv, const int nq) {
    dtau_dqm.assign(num_steps, MatrixXd(nv, nq));
    dtau_dqt.assign(num_steps, MatrixXd(nv, nq));
    dtau_dqp.assign(num_steps, MatrixXd(nv, nq));

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
  std::vector<MatrixXd> dtau_dqm;

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
  std::vector<MatrixXd> dtau_dqt;

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
  std::vector<MatrixXd> dtau_dqp;
};

/**
 * Struct for holding quantities that are computed from the optimizer state,
 */
struct TrajectoryOptimizerCache {
  TrajectoryOptimizerCache(const int num_steps, const int nv, const int nq)
      : v_partials(num_steps, nv, nq), id_partials(num_steps, nv, nq) {
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
  InverseDynamicsPartials id_partials;

  // Flag for cache invalidation
  bool up_to_date{false};
};

/**
 * Struct for storing the "state" of the trajectory optimizer.
 *
 * The only actual state is the sequence of generalized positions q at each
 * timestep. This class stores that directly, but also a "cache" of other values
 * computed from q, such as generalized velocities and forces at each timesteps,
 * relevant dynamics partials, etc.
 */
class TrajectoryOptimizerState {
 public:
  /**
   * Constructor which allocates things of the proper sizes.
   *
   * @param num_steps number of timesteps in the optimization problem
   * @param nv number of multibody velocities
   * @param nq number of multipody positions
   */
  TrajectoryOptimizerState(const int num_steps, const int nv, const int nq)
      : cache_(num_steps, nv, nq) {
    q_.assign(num_steps + 1, VectorXd(nq));
  }

  /**
   * Getter for the sequence of generalized velocities.
   *
   * @return const std::vector<VectorXd>& q
   */
  const std::vector<VectorXd>& q() const { return q_; }

  /**
   * Setter for the sequence of generalized velocities. Invalidates the cache.
   *
   * @param q
   */
  void set_q(const std::vector<VectorXd>& q) {
    q_ = q;
    cache_.up_to_date = false;
  }

  /**
   * Getter for the cache, containing other values computed from q, such as
   * generalized velocities, forces, and various dynamics derivatives.
   *
   * @return const TrajectoryOptimizerCache& cache
   */
  const TrajectoryOptimizerCache& cache() const { return cache_; }

  /**
   *
   * Get a mutable copy of the cache, containing other values computed from q,
   * such as generalized velocities, forces, and various dynamics derivatives.
   *
   * @return TrajectoryOptimizerCache&
   */
  TrajectoryOptimizerCache& mutable_cache() const { return cache_; }

 private:
  // Sequence of generalized velocities at each timestep,
  // [q(0), q(1), ..., q(num_steps)]
  // TODO(vincekurtz): consider storing as a single VectorXd for better memory
  // layout.
  std::vector<VectorXd> q_;

  // Storage for all other quantities that are computed from q, and are useful
  // for our calculations
  mutable TrajectoryOptimizerCache cache_;
};

}  // namespace traj_opt
}  // namespace drake
