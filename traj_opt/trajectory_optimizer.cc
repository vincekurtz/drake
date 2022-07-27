#include "drake/traj_opt/trajectory_optimizer.h"

#include <iostream>
#include <limits>

namespace drake {
namespace traj_opt {

using multibody::Joint;
using multibody::JointIndex;
using multibody::MultibodyPlant;
using systems::System;

TrajectoryOptimizer::TrajectoryOptimizer(const MultibodyPlant<double>* plant,
                                         const ProblemDefinition& prob)
    : plant_(plant), prob_(prob) {
  context_ = plant_->CreateDefaultContext();

  // Define joint damping coefficients.
  joint_damping_ = VectorXd::Zero(plant_->num_velocities());

  for (JointIndex j(0); j < plant_->num_joints(); ++j) {
    const Joint<double>& joint = plant_->get_joint(j);
    const int velocity_start = joint.velocity_start();
    const int nv = joint.num_velocities();
    joint_damping_.segment(velocity_start, nv) = joint.damping_vector();
  }
}

void TrajectoryOptimizer::CalcV(const std::vector<VectorXd>& q,
                                std::vector<VectorXd>* v) const {
  // x = [x0, x1, ..., xT]
  DRAKE_DEMAND(static_cast<int>(q.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(v->size()) == num_steps() + 1);

  v->at(0) = prob_.v_init;
  for (int i = 1; i <= num_steps(); ++i) {
    v->at(i) = (q[i] - q[i - 1]) / time_step();
  }
}

void TrajectoryOptimizer::CalcTau(const std::vector<VectorXd>& q,
                                  const std::vector<VectorXd>& v,
                                  TrajectoryOptimizerWorkspace* workspace,
                                  std::vector<VectorXd>* tau) const {
  // Generalized forces aren't defined for the last timestep
  // TODO(vincekurtz): additional checks that q_t, v_t, tau_t are the right size
  // for the plant?
  DRAKE_DEMAND(static_cast<int>(q.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(v.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(tau->size()) == num_steps());

  for (int t = 0; t < num_steps(); ++t) {
    InverseDynamicsHelper(q[t], v[t + 1], v[t], workspace, &tau->at(t));
  }
}

void TrajectoryOptimizer::InverseDynamicsHelper(
    const VectorXd& q, const VectorXd& v_next, const VectorXd& v,
    TrajectoryOptimizerWorkspace* workspace, VectorXd* tau) const {
  plant().SetPositions(context_.get(), q);
  plant().SetVelocities(context_.get(), v);
  plant().CalcForceElementsContribution(*context_, &workspace->f_ext);

  // Inverse dynamics computes M*a + D*v - k(q,v)
  workspace->a = (v_next - v) / time_step();
  *tau = plant().CalcInverseDynamics(*context_, workspace->a, workspace->f_ext);

  // CalcInverseDynamics considers damping from v_t (D*v_t), but we want to
  // consider damping from v_{t+1} (D*v_{t+1}).
  tau->array() += joint_damping_.array() * (v_next.array() - v.array());

  // TODO(vincekurtz) add in contact/constriant contribution
}

void TrajectoryOptimizer::CalcInverseDynamicsPartials(
    const std::vector<VectorXd>& q, const std::vector<VectorXd>& v,
    TrajectoryOptimizerWorkspace* workspace, GradientData* grad_data) const {
  // TODO(vincekurtz): use a solver flag to choose between finite differences
  // and an analytical approximation
  CalcInverseDynamicsPartialsFiniteDiff(q, v, workspace, grad_data);
}

void TrajectoryOptimizer::CalcInverseDynamicsPartialsFiniteDiff(
    const std::vector<VectorXd>& q, const std::vector<VectorXd>& v,
    TrajectoryOptimizerWorkspace* workspace, GradientData* grad_data) const {
  // Allocate storage space
  // TODO(vincekurtz): allocate a GradientData struct of the proper size earlier
  // in the process, and check that it has the correct size here.
  const int nv = plant().num_velocities();
  const int nq = plant().num_positions();
  std::vector<MatrixXd> dtau_dqm(num_steps(), MatrixXd(nv, nq));
  std::vector<MatrixXd> dtau_dq(num_steps(), MatrixXd(nv, nq));
  std::vector<MatrixXd> dtau_dqp(num_steps(), MatrixXd(nv, nq));

  // All derivatives w.r.t. q0 are zero, since q0 = q_init is fixed. We only
  // include them in GradientData so we can index by t.
  dtau_dqm[0].setZero();
  dtau_dqm[1].setZero();
  dtau_dq[0].setZero();

  // Compute tau(q) [all timesteps] using the orignal value of q
  // TODO(vincekurtz): consider passing this as an argument along with q and v,
  // perhaps combined into a TrajectoryData struct
  std::vector<VectorXd> tau(num_steps());
  CalcTau(q, v, workspace, &tau);

  // Perturbed versions of q_t, v_t, v_{t+1}, tau_{t-1}, tau_t, and tau_{t - 1}.
  // These are all of the quantities that change when we perturb q_t.
  VectorXd q_eps_t(nq);
  VectorXd v_eps_t(nv);
  VectorXd v_eps_tp(nv);
  VectorXd tau_eps_tm(nv);
  VectorXd tau_eps_t(nv);
  VectorXd tau_eps_tp(nv);

  const double eps = sqrt(std::numeric_limits<double>::epsilon());
  for (int t = 1; t <= num_steps(); ++t) {
    for (int i = 0; i < nq; ++i) {
      // Perturb q_t by epsilon
      q_eps_t = q[t];
      q_eps_t(i) += eps;

      // Compute perturbed v(q_t) and tau(q_t) accordingly
      // TODO(vincekurtz): add N(q)+ factor to consider quaternion DoFs.
      v_eps_t = (q_eps_t - q[t - 1]) / time_step();
      InverseDynamicsHelper(q[t - 1], v_eps_t, v[t - 1], workspace,
                            &tau_eps_tm);

      if (t < num_steps()) {
        v_eps_tp = (q[t + 1] - q_eps_t) / time_step();
        InverseDynamicsHelper(q_eps_t, v_eps_tp, v_eps_t, workspace,
                              &tau_eps_t);
      }

      if (t < num_steps() - 1) {
        InverseDynamicsHelper(q[t + 1], v[t + 2], v_eps_tp, workspace,
                              &tau_eps_tp);
      }

      // Compute the nonzero entries of dtau/dq via finite differencing
      dtau_dqp[t - 1].col(i) = (tau_eps_tm - tau[t - 1]) / eps;
      if (t < num_steps()) {
        dtau_dq[t].col(i) = (tau_eps_t - tau[t]) / eps;
      }
      if (t < num_steps() - 1) {
        dtau_dqm[t + 1].col(i) = (tau_eps_tp - tau[t + 1]) / eps;
      }
    }
  }

  // Put the results into the GradientData struct
  grad_data->dtau_dqm = dtau_dqm;
  grad_data->dtau_dq = dtau_dq;
  grad_data->dtau_dqp = dtau_dqp;
}

}  // namespace traj_opt
}  // namespace drake
