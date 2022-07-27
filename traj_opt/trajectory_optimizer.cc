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

    // Acceleration at time t
    VectorXd& a = workspace->a_size_tmp1;
    a = (v[t+1] - v[t])/time_step();
    
    // All dynamics terms are treated implicitly, i.e., 
    // tau[t] = M(q[t+1]) * a[t] - k(q[t+1],v[t+1]) - f_ext[t+1]
    InverseDynamicsHelper(q[t+1], v[t+1], a, workspace, &tau->at(t));
  }
}

void TrajectoryOptimizer::InverseDynamicsHelper(
    const VectorXd& q, const VectorXd& v, const VectorXd& a,
    TrajectoryOptimizerWorkspace* workspace, VectorXd* tau) const {
  plant().SetPositions(context_.get(), q);
  plant().SetVelocities(context_.get(), v);
  plant().CalcForceElementsContribution(*context_, &workspace->f_ext);

  // Inverse dynamics computes tau = M*a - k(q,v) - f_ext
  *tau = plant().CalcInverseDynamics(*context_, a, workspace->f_ext);

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
  // Check that grad_data has been allocated correctly.
  DRAKE_DEMAND(grad_data->size() == num_steps());

  // Get references to the partials that we'll be setting
  std::vector<MatrixXd>& dtau_dqm = grad_data->dtau_dqm;
  std::vector<MatrixXd>& dtau_dq = grad_data->dtau_dq;
  std::vector<MatrixXd>& dtau_dqp = grad_data->dtau_dqp;

  // Compute tau(q) [all timesteps] using the orignal value of q
  // TODO(vincekurtz): consider passing this as an argument along with q and v,
  // perhaps combined into a TrajectoryData struct
  std::vector<VectorXd> tau(num_steps());
  CalcTau(q, v, workspace, &tau);

  // Get references to perturbed versions of q, v, tau, and a, at (t-1, t, t).
  // These are all of the quantities that change when we perturb q_t.
  VectorXd& q_eps_t = workspace->q_size_tmp;
  VectorXd& v_eps_t = workspace->v_size_tmp1;
  VectorXd& v_eps_tp = workspace->v_size_tmp2;
  VectorXd& a_eps_tm = workspace->a_size_tmp1;
  VectorXd& a_eps_t = workspace->a_size_tmp2;
  VectorXd& a_eps_tp = workspace->a_size_tmp3;
  VectorXd& tau_eps_tm = workspace->tau_size_tmp1;
  VectorXd& tau_eps_t = workspace->tau_size_tmp2;
  VectorXd& tau_eps_tp = workspace->tau_size_tmp3;

  const double eps = sqrt(std::numeric_limits<double>::epsilon());
  double dqt_i;
  for (int t = 1; t <= num_steps(); ++t) {
    for (int i = 0; i < plant().num_positions(); ++i) {
      // Perturb q_t by epsilon
      q_eps_t = q[t];
      dqt_i = eps * std::max(1.0, std::abs(q_eps_t(i)));  // avoid losing precision
      q_eps_t(i) += dqt_i;

      // Compute perturbed v(q_t)
      // TODO(vincekurtz): add N(q)+ factor to consider quaternion DoFs.
      v_eps_t = (q_eps_t - q[t - 1]) / time_step();
      if (t < num_steps()) {
        v_eps_tp = (q[t + 1] - q_eps_t) / time_step();
      }

      // tau[t-1] = ID(q[t], v[t], a[t-1])
      a_eps_tm = (v_eps_t - v[t-1]) / time_step();
      InverseDynamicsHelper(q_eps_t, v_eps_t, a_eps_tm, workspace,
                            &tau_eps_tm);

      // tau[t] = ID(q[t+1], v[t+1], a[t])
      if (t < num_steps()) {
        a_eps_t = (v_eps_tp - v_eps_t) / time_step();
        InverseDynamicsHelper(q[t+1], v_eps_tp, a_eps_t, workspace,
                              &tau_eps_t);
      }

      // tau[t+1] = ID(q[t+2], v[t+2], a[t+1])
      if (t < num_steps() - 1) {
        a_eps_tp = (v[t+2] - v_eps_tp) / time_step();
        InverseDynamicsHelper(q[t + 2], v[t + 2], a_eps_tp, workspace,
                              &tau_eps_tp);
      }

      // Compute the nonzero entries of dtau/dq via finite differencing
      dtau_dqp[t - 1].col(i) = (tau_eps_tm - tau[t - 1]) / dqt_i;
      if (t < num_steps()) {
        dtau_dq[t].col(i) = (tau_eps_t - tau[t]) / dqt_i;
      }
      if (t < num_steps() - 1) {
        dtau_dqm[t + 1].col(i) = (tau_eps_tp - tau[t + 1]) / dqt_i;
      }
    }
  }
}

}  // namespace traj_opt
}  // namespace drake
