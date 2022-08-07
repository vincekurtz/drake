#include "drake/traj_opt/trajectory_optimizer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

#include "drake/traj_opt/penta_diagonal_solver.h"

namespace drake {
namespace traj_opt {

using internal::PentaDiagonalFactorization;
using internal::PentaDiagonalFactorizationStatus;
using multibody::Joint;
using multibody::JointIndex;
using multibody::MultibodyPlant;
using systems::System;

template <typename T>
TrajectoryOptimizer<T>::TrajectoryOptimizer(
    const MultibodyPlant<T>* plant, const ProblemDefinition& prob,
    const std::optional<SolverParameters>& params)
    : plant_(plant), prob_(prob) {
  // Set solver parameters if they're supplied. Otherwise we keep the defaults
  // from solver_parameters.h
  if (params != std::nullopt) {
    params_ = *params;
  }

  // Create a context for dynamics computations
  context_ = plant_->CreateDefaultContext();

  // Define joint damping coefficients.
  joint_damping_ = VectorX<T>::Zero(plant_->num_velocities());

  for (JointIndex j(0); j < plant_->num_joints(); ++j) {
    const Joint<T>& joint = plant_->get_joint(j);
    const int velocity_start = joint.velocity_start();
    const int nv = joint.num_velocities();
    joint_damping_.segment(velocity_start, nv) = joint.damping_vector();
  }
}

template <typename T>
T TrajectoryOptimizer<T>::CalcCost(
    const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().up_to_date) {
    // Only update the parts of the cache that we will actually use.
    // Don't update any of the partial derivatives, since those
    // are expensive to compute.
    TrajectoryOptimizerCache<T>& cache = state.mutable_cache();
    TrajectoryOptimizerWorkspace<T>& workspace = state.workspace;
    std::vector<VectorX<T>>& v = cache.v;
    std::vector<VectorX<T>>& a = cache.a;
    std::vector<VectorX<T>>& tau = cache.tau;

    const std::vector<VectorX<T>>& q = state.q();
    CalcVelocities(q, &v);
    CalcAccelerations(v, &a);
    CalcInverseDynamics(q, v, a, &workspace, &tau);

    // TODO(vincekurtz): consider adding an additional flag to state for whether
    // v, a, tau are updated, and whether v_partials and id_partials are
    // updated.
  }
  return CalcCost(state.q(), state.cache().v, state.cache().tau,
                  &state.workspace);
}

template <typename T>
T TrajectoryOptimizer<T>::CalcCost(
    const std::vector<VectorX<T>>& q, const std::vector<VectorX<T>>& v,
    const std::vector<VectorX<T>>& tau,
    TrajectoryOptimizerWorkspace<T>* workspace) const {
  VectorX<T> cost = VectorX<T>::Zero(1);
  VectorX<T>& q_err = workspace->q_size_tmp;
  VectorX<T>& v_err = workspace->v_size_tmp1;

  // Running cost
  for (int t = 0; t < num_steps(); ++t) {
    q_err = q[t] - prob_.q_nom;
    v_err = v[t] - prob_.v_nom;
    cost += q_err.transpose() * prob_.Qq * q_err;
    cost += v_err.transpose() * prob_.Qv * v_err;
    cost += tau[t].transpose() * prob_.R * tau[t];
  }

  // Scale running cost by dt (so the optimization problem we're solving doesn't
  // change so dramatically when we change the time step).
  cost *= time_step();

  // Terminal cost
  q_err = q[num_steps()] - prob_.q_nom;
  v_err = v[num_steps()] - prob_.v_nom;
  cost += q_err.transpose() * prob_.Qf_q * q_err;
  cost += v_err.transpose() * prob_.Qf_v * v_err;

  return cost[0];
}

template <typename T>
void TrajectoryOptimizer<T>::CalcVelocities(const std::vector<VectorX<T>>& q,
                                            std::vector<VectorX<T>>* v) const {
  // x = [x0, x1, ..., xT]
  DRAKE_DEMAND(static_cast<int>(q.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(v->size()) == num_steps() + 1);

  v->at(0) = prob_.v_init;
  for (int t = 1; t <= num_steps(); ++t) {
    v->at(t) = (q[t] - q[t - 1]) / time_step();
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcAccelerations(
    const std::vector<VectorX<T>>& v, std::vector<VectorX<T>>* a) const {
  DRAKE_DEMAND(static_cast<int>(v.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(a->size()) == num_steps());

  for (int t = 0; t < num_steps(); ++t) {
    a->at(t) = (v[t + 1] - v[t]) / time_step();
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamics(
    const std::vector<VectorX<T>>& q, const std::vector<VectorX<T>>& v,
    const std::vector<VectorX<T>>& a,
    TrajectoryOptimizerWorkspace<T>* workspace,
    std::vector<VectorX<T>>* tau) const {
  // Generalized forces aren't defined for the last timestep
  // TODO(vincekurtz): additional checks that q_t, v_t, tau_t are the right size
  // for the plant?
  DRAKE_DEMAND(static_cast<int>(q.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(v.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(a.size()) == num_steps());
  DRAKE_DEMAND(static_cast<int>(tau->size()) == num_steps());

  for (int t = 0; t < num_steps(); ++t) {
    // All dynamics terms are treated implicitly, i.e.,
    // tau[t] = M(q[t+1]) * a[t] - k(q[t+1],v[t+1]) - f_ext[t+1]
    CalcInverseDynamicsSingleTimeStep(q[t + 1], v[t + 1], a[t], workspace,
                                      &tau->at(t));
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamicsSingleTimeStep(
    const VectorX<T>& q, const VectorX<T>& v, const VectorX<T>& a,
    TrajectoryOptimizerWorkspace<T>* workspace, VectorX<T>* tau) const {
  plant().SetPositions(context_.get(), q);
  plant().SetVelocities(context_.get(), v);
  plant().CalcForceElementsContribution(*context_, &workspace->f_ext);

  // Inverse dynamics computes tau = M*a - k(q,v) - f_ext
  *tau = plant().CalcInverseDynamics(*context_, a, workspace->f_ext);

  // TODO(vincekurtz) add in contact/constriant contribution
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamicsPartials(
    const std::vector<VectorX<T>>& q, const std::vector<VectorX<T>>& v,
    const std::vector<VectorX<T>>& a, const std::vector<VectorX<T>>& tau,
    TrajectoryOptimizerWorkspace<T>* workspace,
    InverseDynamicsPartials<T>* id_partials) const {
  // TODO(vincekurtz): use a solver flag to choose between finite differences
  // and an analytical approximation
  CalcInverseDynamicsPartialsFiniteDiff(q, v, a, tau, workspace, id_partials);
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamicsPartialsFiniteDiff(
    const std::vector<VectorX<T>>& q, const std::vector<VectorX<T>>& v,
    const std::vector<VectorX<T>>& a, const std::vector<VectorX<T>>& tau,
    TrajectoryOptimizerWorkspace<T>* workspace,
    InverseDynamicsPartials<T>* id_partials) const {
  using std::abs;
  using std::max;
  // Check that id_partials has been allocated correctly.
  DRAKE_DEMAND(id_partials->size() == num_steps());

  // Get references to the partials that we'll be setting
  std::vector<MatrixX<T>>& dtau_dqm = id_partials->dtau_dqm;
  std::vector<MatrixX<T>>& dtau_dqt = id_partials->dtau_dqt;
  std::vector<MatrixX<T>>& dtau_dqp = id_partials->dtau_dqp;

  // Get references to perturbed versions of q, v, tau, and a, at (t-1, t, t).
  // These are all of the quantities that change when we perturb q_t.
  VectorX<T>& q_eps_t = workspace->q_size_tmp;
  VectorX<T>& v_eps_t = workspace->v_size_tmp1;
  VectorX<T>& v_eps_tp = workspace->v_size_tmp2;
  VectorX<T>& a_eps_tm = workspace->a_size_tmp1;
  VectorX<T>& a_eps_t = workspace->a_size_tmp2;
  VectorX<T>& a_eps_tp = workspace->a_size_tmp3;
  VectorX<T>& tau_eps_tm = workspace->tau_size_tmp1;
  VectorX<T>& tau_eps_t = workspace->tau_size_tmp2;
  VectorX<T>& tau_eps_tp = workspace->tau_size_tmp3;

  // Store small perturbations
  const double eps = sqrt(std::numeric_limits<double>::epsilon());
  T dq_i;
  T dv_i;
  T da_i;
  for (int t = 0; t <= num_steps(); ++t) {
    // N.B. A perturbation of qt propagates to tau[t-1], tau[t] and tau[t+1].
    // Therefore we compute one column of grad_tau at a time. That is, once the
    // loop on position indices i is over, we effectively computed the t-th
    // column of grad_tau.

    // Set perturbed versions of variables
    q_eps_t = q[t];
    v_eps_t = v[t];
    if (t < num_steps()) {
      // v[num_steps + 1] is not defined
      v_eps_tp = v[t + 1];
      // a[num_steps] is not defined
      a_eps_t = a[t];
    }
    if (t < num_steps() - 1) {
      // a[num_steps + 1] is not defined
      a_eps_tp = a[t + 1];
    }
    if (t > 0) {
      // a[-1] is undefined
      a_eps_tm = a[t - 1];
    }

    for (int i = 0; i < plant().num_positions(); ++i) {
      // Determine perturbation sizes to avoid losing precision to floating
      // point error
      dq_i = eps * max(1.0, abs(q_eps_t(i)));
      dv_i = dq_i / time_step();
      da_i = dv_i / time_step();

      // Perturb q_t[i], v_t[i], and a_t[i]
      // TODO(vincekurtz): add N(q)+ factor to consider quaternion DoFs.
      q_eps_t(i) += dq_i;

      if (t == 0) {
        // v[0] is constant
        a_eps_t(i) -= 1.0 * da_i;
      } else {
        v_eps_t(i) += dv_i;
        a_eps_tm(i) += da_i;
        a_eps_t(i) -= 2.0 * da_i;
      }
      v_eps_tp(i) -= dv_i;
      a_eps_tp(i) += da_i;

      // Compute perturbed tau(q) and calculate the nonzero entries of dtau/dq
      // via finite differencing
      if (t > 0) {
        // tau[t-1] = ID(q[t], v[t], a[t-1])
        CalcInverseDynamicsSingleTimeStep(q_eps_t, v_eps_t, a_eps_tm, workspace,
                                          &tau_eps_tm);
        dtau_dqp[t - 1].col(i) = (tau_eps_tm - tau[t - 1]) / dq_i;
      }
      if (t < num_steps()) {
        // tau[t] = ID(q[t+1], v[t+1], a[t])
        CalcInverseDynamicsSingleTimeStep(q[t + 1], v_eps_tp, a_eps_t,
                                          workspace, &tau_eps_t);
        dtau_dqt[t].col(i) = (tau_eps_t - tau[t]) / dq_i;
      }
      if (t < num_steps() - 1) {
        // tau[t+1] = ID(q[t+2], v[t+2], a[t+1])
        CalcInverseDynamicsSingleTimeStep(q[t + 2], v[t + 2], a_eps_tp,
                                          workspace, &tau_eps_tp);
        dtau_dqm[t + 1].col(i) = (tau_eps_tp - tau[t + 1]) / dq_i;
      }

      // Unperturb q_t[i], v_t[i], and a_t[i]
      q_eps_t(i) = q[t](i);
      v_eps_t(i) = v[t](i);
      if (t < num_steps()) {
        v_eps_tp(i) = v[t + 1](i);
        a_eps_t(i) = a[t](i);
      }
      if (t < num_steps() - 1) {
        a_eps_tp(i) = a[t + 1](i);
      }
      if (t > 0) {
        a_eps_tm(i) = a[t - 1](i);
      }
    }
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcVelocityPartials(
    const std::vector<VectorX<T>>&, VelocityPartials<T>* v_partials) const {
  if (plant().num_velocities() != plant().num_positions()) {
    throw std::runtime_error("Quaternion DoFs not yet supported");
  } else {
    const int nq = plant().num_positions();
    for (int t = 0; t <= num_steps(); ++t) {
      v_partials->dvt_dqt[t] = 1 / time_step() * MatrixXd::Identity(nq, nq);
      if (t > 0) {
        v_partials->dvt_dqm[t] = -1 / time_step() * MatrixXd::Identity(nq, nq);
      }
    }
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcGradientFiniteDiff(
    const TrajectoryOptimizerState<T>& state, EigenPtr<VectorX<T>> g) const {
  using std::abs;
  using std::max;

  // Perturbed versions of q
  std::vector<VectorX<T>>& q_plus = state.workspace.q_sequence_tmp1;
  std::vector<VectorX<T>>& q_minus = state.workspace.q_sequence_tmp2;
  q_plus = state.q();
  q_minus = state.q();

  // non-constant copy of state that we can perturb
  TrajectoryOptimizerState<T> state_eps(state);

  // Set first block of g (derivatives w.r.t. q_0) to zero, since q0 = q_init
  // are constant.
  g->topRows(plant().num_positions()).setZero();

  // Iterate through rows of g using finite differences
  const double eps = cbrt(std::numeric_limits<double>::epsilon());
  T dqt_i;
  int j = plant().num_positions();
  for (int t = 1; t <= num_steps(); ++t) {
    for (int i = 0; i < plant().num_positions(); ++i) {
      // Set finite difference step size
      dqt_i = eps * max(1.0, abs(state.q()[t](i)));
      q_plus[t](i) += dqt_i;
      q_minus[t](i) -= dqt_i;

      // Set g_j = using central differences
      state_eps.set_q(q_plus);
      T L_plus = CalcCost(state_eps);
      state_eps.set_q(q_minus);
      T L_minus = CalcCost(state_eps);
      (*g)(j) = (L_plus - L_minus) / (2 * dqt_i);

      // reset our perturbed Q and move to the next row of g.
      q_plus[t](i) = state.q()[t](i);
      q_minus[t](i) = state.q()[t](i);
      ++j;
    }
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcGradient(
    const TrajectoryOptimizerState<T>& state, EigenPtr<VectorX<T>> g) const {
  if (!state.cache().up_to_date) UpdateCache(state);
  const TrajectoryOptimizerCache<T>& cache = state.cache();

  // Set some aliases
  const double dt = time_step();
  const int nq = plant().num_positions();
  const std::vector<VectorX<T>>& q = state.q();
  const std::vector<VectorX<T>>& v = cache.v;
  const std::vector<VectorX<T>>& tau = cache.tau;
  const std::vector<MatrixX<T>>& dvt_dqt = cache.v_partials.dvt_dqt;
  const std::vector<MatrixX<T>>& dvt_dqm = cache.v_partials.dvt_dqm;
  const std::vector<MatrixX<T>>& dtau_dqp = cache.id_partials.dtau_dqp;
  const std::vector<MatrixX<T>>& dtau_dqt = cache.id_partials.dtau_dqt;
  const std::vector<MatrixX<T>>& dtau_dqm = cache.id_partials.dtau_dqm;
  TrajectoryOptimizerWorkspace<T>* workspace = &state.workspace;

  // Set first block of g (derivatives w.r.t. q_0) to zero, since q0 = q_init
  // are constant.
  g->topRows(plant().num_positions()).setZero();

  // Scratch variables for storing intermediate cost terms
  VectorX<T>& qt_term = workspace->q_size_tmp;
  VectorX<T>& vt_term = workspace->v_size_tmp1;
  VectorX<T>& vp_term = workspace->v_size_tmp2;
  VectorX<T>& taum_term = workspace->tau_size_tmp1;
  VectorX<T>& taut_term = workspace->tau_size_tmp2;
  VectorX<T>& taup_term = workspace->tau_size_tmp3;

  for (int t = 1; t < num_steps(); ++t) {
    // Contribution from position cost
    qt_term = (q[t] - prob_.q_nom).transpose() * 2 * prob_.Qq * dt;

    // Contribution from velocity cost
    vt_term = (v[t] - prob_.v_nom).transpose() * 2 * prob_.Qv * dt * dvt_dqt[t];
    if (t == num_steps() - 1) {
      // The terminal cost needs to be handled differently
      vp_term = (v[t + 1] - prob_.v_nom).transpose() * 2 * prob_.Qf_v *
                dvt_dqm[t + 1];
    } else {
      vp_term = (v[t + 1] - prob_.v_nom).transpose() * 2 * prob_.Qv * dt *
                dvt_dqm[t + 1];
    }

    // Contribution from control cost
    taum_term = tau[t - 1].transpose() * 2 * prob_.R * dt * dtau_dqp[t - 1];
    taut_term = tau[t].transpose() * 2 * prob_.R * dt * dtau_dqt[t];
    if (t == num_steps() - 1) {
      // There is no constrol input at the final timestep
      taup_term.setZero(nq);
    } else {
      taup_term = tau[t + 1].transpose() * 2 * prob_.R * dt * dtau_dqm[t + 1];
    }

    // Put it all together to get the gradient w.r.t q[t]
    g->segment(t * nq, nq) =
        qt_term + vt_term + vp_term + taum_term + taut_term + taup_term;
  }

  // Last step is different, because there is terminal cost and v[t+1] doesn't
  // exist
  taum_term = tau[num_steps() - 1].transpose() * 2 * prob_.R * dt *
              dtau_dqp[num_steps() - 1];
  qt_term = (q[num_steps()] - prob_.q_nom).transpose() * 2 * prob_.Qf_q;
  vt_term = (v[num_steps()] - prob_.v_nom).transpose() * 2 * prob_.Qf_v *
            dvt_dqt[num_steps()];
  g->tail(nq) = qt_term + vt_term + taum_term;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcHessian(
    const TrajectoryOptimizerState<T>& state, PentaDiagonalMatrix<T>* H) const {
  DRAKE_DEMAND(H->is_symmetric());
  DRAKE_DEMAND(H->block_rows() == num_steps() + 1);
  DRAKE_DEMAND(H->block_size() == plant().num_positions());

  // Make sure the cache is up to date
  if (!state.cache().up_to_date) UpdateCache(state);
  const TrajectoryOptimizerCache<T>& cache = state.cache();

  // Some convienient aliases
  const double dt = time_step();
  const MatrixX<T> Qq = 2 * prob_.Qq * dt;
  const MatrixX<T> Qv = 2 * prob_.Qv * dt;
  const MatrixX<T> R = 2 * prob_.R * dt;
  const MatrixX<T> Qf_q = 2 * prob_.Qf_q;
  const MatrixX<T> Qf_v = 2 * prob_.Qf_v;
  const std::vector<MatrixX<T>>& dvt_dqt = cache.v_partials.dvt_dqt;
  const std::vector<MatrixX<T>>& dvt_dqm = cache.v_partials.dvt_dqm;
  const std::vector<MatrixX<T>>& dtau_dqp = cache.id_partials.dtau_dqp;
  const std::vector<MatrixX<T>>& dtau_dqt = cache.id_partials.dtau_dqt;
  const std::vector<MatrixX<T>>& dtau_dqm = cache.id_partials.dtau_dqm;

  // Get mutable references to the non-zero bands of the Hessian
  std::vector<MatrixX<T>>& A = H->mutable_A();  // 2 rows below diagonal
  std::vector<MatrixX<T>>& B = H->mutable_B();  // 1 row below diagonal
  std::vector<MatrixX<T>>& C = H->mutable_C();  // diagonal

  // Fill in the non-zero blocks
  C[0].setIdentity();  // Initial condition q0 fixed at t=0
  for (int t = 1; t < num_steps(); ++t) {
    // dg_t/dq_t
    MatrixX<T>& dgt_dqt = C[t];
    dgt_dqt = Qq;
    dgt_dqt += dvt_dqt[t].transpose() * Qv * dvt_dqt[t];
    dgt_dqt += dtau_dqp[t - 1].transpose() * R * dtau_dqp[t - 1];
    dgt_dqt += dtau_dqt[t].transpose() * R * dtau_dqt[t];
    if (t < num_steps() - 1) {
      dgt_dqt += dtau_dqm[t + 1].transpose() * R * dtau_dqm[t + 1];
      dgt_dqt += dvt_dqm[t + 1].transpose() * Qv * dvt_dqm[t + 1];
    } else {
      dgt_dqt += dvt_dqm[t + 1].transpose() * Qf_v * dvt_dqm[t + 1];
    }

    // dg_t/dq_{t+1}
    MatrixX<T>& dgt_dqp = B[t + 1];
    dgt_dqp = dtau_dqp[t].transpose() * R * dtau_dqt[t];
    if (t < num_steps() - 1) {
      dgt_dqp += dtau_dqt[t + 1].transpose() * R * dtau_dqm[t + 1];
      dgt_dqp += dvt_dqt[t + 1].transpose() * Qv * dvt_dqm[t + 1];
    } else {
      dgt_dqp += dvt_dqt[t + 1].transpose() * Qf_v * dvt_dqm[t + 1];
    }

    // dg_t/dq_{t+2}
    if (t < num_steps() - 1) {
      MatrixX<T>& dgt_dqpp = A[t + 2];
      dgt_dqpp = dtau_dqp[t + 1].transpose() * R * dtau_dqm[t + 1];
    }
  }

  // dg_t/dq_t for the final timestep
  MatrixX<T>& dgT_dqT = C[num_steps()];
  dgT_dqT = Qf_q;
  dgT_dqT += dvt_dqt[num_steps()].transpose() * Qf_v * dvt_dqt[num_steps()];
  dgT_dqT +=
      dtau_dqp[num_steps() - 1].transpose() * R * dtau_dqp[num_steps() - 1];

  // Copy lower triangular part to upper triangular part
  H->MakeSymmetric();
}

template <typename T>
void TrajectoryOptimizer<T>::UpdateCache(
    const TrajectoryOptimizerState<T>& state) const {
  TrajectoryOptimizerCache<T>& cache = state.mutable_cache();
  TrajectoryOptimizerWorkspace<T>& workspace = state.workspace;

  // Some aliases for things that we'll set
  std::vector<VectorX<T>>& v = cache.v;
  std::vector<VectorX<T>>& a = cache.a;
  std::vector<VectorX<T>>& tau = cache.tau;
  InverseDynamicsPartials<T>& id_partials = cache.id_partials;
  VelocityPartials<T>& v_partials = cache.v_partials;

  // The generalized positions that everything is computed from
  const std::vector<VectorX<T>>& q = state.q();

  // Compute corresponding generalized velocities
  // TODO(vincekurtz) consider making this & similar functions private
  CalcVelocities(q, &v);

  // Compute corresponding generalized accelerations
  CalcAccelerations(v, &a);

  // Compute corresponding generalized torques
  CalcInverseDynamics(q, v, a, &workspace, &tau);

  // Compute partial derivatives of inverse dynamics d(tau)/d(q)
  CalcInverseDynamicsPartials(q, v, a, tau, &workspace, &id_partials);

  // Compute partial derivatives of velocities d(v)/d(q)
  CalcVelocityPartials(q, &v_partials);

  // Set cache invalidation flag
  cache.up_to_date = true;
}

template <typename T>
void TrajectoryOptimizer<T>::PrintLinesearchResidual(
    const T L, const std::vector<VectorX<T>>& q, const VectorX<T>& dq,
    TrajectoryOptimizerState<T>* state) const {
  // TODO(vincekurtz): save to a CSV file instead, including sampled alpha 
  // and sampled dq values
  const int nq = plant().num_positions();
  std::cout << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << "Linsearch Residual: " << std::endl;
  std::cout << "dq: " << dq << std::endl << std::endl;
  double d_alpha = 0.01;
  double alpha = -1.0;

  while (alpha <= 1.0) {
    // Compute phi(alpha) = L - L(q + alpha * dq)
    for (int t = 1; t <= num_steps(); ++t) {
      state->set_qt(q[t] + alpha * dq.segment(t * nq, nq), t);
    }
    std::cout << CalcCost(*state) - L << ", ";
    alpha += d_alpha;
  }
  std::cout << std::endl;

  std::cout << "==================================" << std::endl;
  std::cout << std::endl;
}

template <typename T>
std::tuple<double, int> TrajectoryOptimizer<T>::Linesearch(
    const T L, const std::vector<VectorX<T>>& q, const VectorX<T>& dq,
    const VectorX<T>& g, TrajectoryOptimizerState<T>* state) const {
  // TODO(vincekurtz): add other linesearch strategies
  if (params_.linesearch_method == LinesearchMethod::kBacktrackingArmijo) {
    return BacktrackingArmijoLinesearch(L, q, dq, g, state);
  } else {
    throw std::runtime_error("Unknown linesearch method");
  }
}

template <typename T>
std::tuple<double, int> TrajectoryOptimizer<T>::BacktrackingArmijoLinesearch(
    const T L, const std::vector<VectorX<T>>& q, const VectorX<T>& dq,
    const VectorX<T>& g, TrajectoryOptimizerState<T>* state) const {
  // For now we'll just do a simple backtracking linesearch
  // TODO(vincekurtz): give solver options for different linesearch methods
  const int nq = plant().num_positions();

  // TODO(vincekurtz): set these in SolverOptions
  const double c = 1e-1;
  const double rho = 0.9;

  double alpha = 1.0 / rho;        // get alpha = 1 on first iteration
  T L_prime = g.transpose() * dq;  // gradient of L w.r.t. alpha
  T L_new;                         // L(q + alpha * dq)

  // Minimum acceptable cost reduction.
  // Minimum cost reduction comes from Armijo's criterion, plus a
  // fudge factor to deal with the fact that our gradient and Hessian
  // are noisy, having come from finite differences.
  // TODO(vincekurtz): does this "fudge factor" idea make sense?
  T L_min;
  double fudge_factor = 0; //sqrt(std::numeric_limits<double>::epsilon());

  int i = 0;  // Iteration counter
  do {
    // Reduce alpha
    // N.B. we start with alpha = 1/rho, so we get alpha = 1 on the first
    // iteration.
    alpha *= rho;

    // Compute L_ls = L(q + alpha * dq)
    for (int t = 0; t <= num_steps(); ++t) {
      state->set_qt(q[t] + alpha * dq.segment(t * nq, nq), t);
    }
    L_new = CalcCost(*state);  // N.B. this is also computing extra
                               // gradient data that we don't really need.

    L_min = L + c * alpha * L_prime + fudge_factor;

    ++i;
  } while ((L_new > L_min) && (i < params_.max_linesearch_iterations));

  return {alpha, i};
}

template <typename T>
SolverFlag TrajectoryOptimizer<T>::Solve(const std::vector<VectorX<T>>&,
                                         TrajectoryOptimizerSolution<T>*,
                                         SolutionData<T>*) const {
  throw std::runtime_error(
      "TrajectoryOptimizer::Solve only supports T=double.");
}

template <>
SolverFlag TrajectoryOptimizer<double>::Solve(
    const std::vector<VectorXd>& q_guess,
    TrajectoryOptimizerSolution<double>* solution,
    SolutionData<double>* solution_data) const {
  // The guess must be consistent with the initial condition
  DRAKE_DEMAND(q_guess[0] == prob_.q_init);

  // solution_data must be empty
  DRAKE_DEMAND(solution_data->iteration_times.size() == 0);
  DRAKE_DEMAND(solution_data->iteration_costs.size() == 0);
  DRAKE_DEMAND(solution_data->linesearch_iterations.size() == 0);
  DRAKE_DEMAND(solution_data->linesearch_alphas.size() == 0);
  DRAKE_DEMAND(solution_data->gradient_norm.size() == 0);

  // Detailed solution data
  std::vector<double>& iteration_times = solution_data->iteration_times;
  std::vector<double>& iteration_costs = solution_data->iteration_costs;
  std::vector<int>& linesearch_iterations =
      solution_data->linesearch_iterations;
  std::vector<double>& linesearch_alphas = solution_data->linesearch_alphas;
  std::vector<double>& gradient_norm = solution_data->gradient_norm;

  // Allocate a state variable
  TrajectoryOptimizerState<double> state = CreateState();
  state.set_q(q_guess);

  // Allocate a separate state variable for linesearch
  TrajectoryOptimizerState<double> ls_state(state);

  // Allocate gradient, Hessian, and search direction
  const int nq = plant().num_positions();
  const int num_vars = (num_steps() + 1) * nq;
  VectorXd g(num_vars);
  PentaDiagonalMatrix<double> H(num_steps() + 1, nq);
  VectorXd dq(num_vars);

  // Define printout data
  std::cout << "---------------------------------------------------------"
            << std::endl;
  std::cout << "|  iter  |   cost   |  alpha  |  LS_iters  |  time (s)  |   || g ||   |"
            << std::endl;
  std::cout << "---------------------------------------------------------"
            << std::endl;

  // Allocate timing variables
  auto start_time = std::chrono::high_resolution_clock::now();
  auto iter_start_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> iter_time;
  std::chrono::duration<double> solve_time;

  // Gauss-Newton iterations
  int k = 0;  // iteration counter
  do {
    iter_start_time = std::chrono::high_resolution_clock::now();

    // Update the cache
    UpdateCache(state);

    // Compute the total cost
    iteration_costs.push_back(CalcCost(state));

    // Compute gradient and Hessian
    CalcGradient(state, &g);
    CalcHessian(state, &H);

    // Solve for search direction H*dq = -g
    dq = -g;
    PentaDiagonalFactorization Hchol(H);
    DRAKE_DEMAND(Hchol.status() == PentaDiagonalFactorizationStatus::kSuccess);
    Hchol.SolveInPlace(&dq);

    //// DEBUG: use dense Hessian
    //MatrixXd H_dense = H.MakeDense();
    //dq = H_dense.llt().solve(-g);

    //// DEBUG: ignore linesearch
    //double alpha = 1.0;
    //int ls_iters = 0;

    // Solve the linsearch
    // N.B. we use a separate state variable since we will need to compute
    // L(q+alpha*dq) (at the very least), and we don't want to change state.q
    auto [alpha, ls_iters] =
        Linesearch(iteration_costs[k], state.q(), dq, g, &ls_state);

    if (ls_iters >= params_.max_linesearch_iterations) {
      // Early termination if linesearch is taking too long
      std::cout << "LINESEARCH FAILED" << std::endl;
      std::cout << "Reached maximum linesearch iterations (" << ls_iters << ")."
                << std::endl;

      // DEBUG: print linesearch residual so we can plot in python
      PrintLinesearchResidual(iteration_costs[k], state.q(), dq, &ls_state);

      // We'll still record iteration data for playback later
      solution_data->solve_time = NAN;
      linesearch_iterations.push_back(ls_iters);
      linesearch_alphas.push_back(alpha);
      iter_time = std::chrono::high_resolution_clock::now() - iter_start_time;
      iteration_times.push_back(iter_time.count());
      gradient_norm.push_back(g.norm());

      return SolverFlag::kLinesearchMaxIters;
    }
    
    // Update the decision variables
    for (int t = 1; t <= num_steps(); ++t) {
      // q[t] = q[t] + alpha * dq[t]
      state.set_qt(state.q()[t] + alpha * dq.segment(t * nq, nq), t);
    }

    ++k;
    iter_time = std::chrono::high_resolution_clock::now() - iter_start_time;

    // Nice little printout of our problem data
    printf("| %6d ", k);
    printf("| %8.3f ", iteration_costs[k - 1]);
    printf("| %7.4f ", alpha);
    printf("| %6d     ", ls_iters);
    printf("| %8.8f ", iter_time.count());
    printf("| %4.3e |\n", g.norm());

    //// DEBUG: print condition number
    //double condition_number = H_dense.norm() * H_dense.inverse().norm();
    //printf("cond. num.: %4.3e\n", condition_number);

    // Record iteration data
    linesearch_iterations.push_back(ls_iters);
    linesearch_alphas.push_back(alpha);
    iteration_times.push_back(iter_time.count());
    gradient_norm.push_back(g.norm());
  } while (k < params_.max_iterations);

  std::cout << "---------------------------------------------------------"
            << std::endl;

  // Record the total solve time
  solve_time = std::chrono::high_resolution_clock::now() - start_time;
  solution_data->solve_time = solve_time.count();

  solution->q = state.q();
  return SolverFlag::kSuccess;
}

}  // namespace traj_opt
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::traj_opt::TrajectoryOptimizer)
