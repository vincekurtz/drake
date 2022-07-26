#include "drake/traj_opt/trajectory_optimizer.h"

#include <iostream>
#include <limits>

namespace drake {
namespace traj_opt {

using multibody::Joint;
using multibody::JointIndex;
using multibody::MultibodyForces;
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
                                  const std::vector<VectorXd>& v, VectorXd* a,
                                  MultibodyForces<double>* f_ext,
                                  std::vector<VectorXd>* tau) const {
  // Generalized forces aren't defined for the last timestep
  // TODO(vincekurtz): additional checks that q_t, v_t, tau_t are the right size
  // for the plant?
  DRAKE_DEMAND(static_cast<int>(q.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(v.size()) == num_steps() + 1);
  DRAKE_DEMAND(static_cast<int>(tau->size()) == num_steps());

  for (int t = 0; t < num_steps(); ++t) {
    plant().SetPositions(context_.get(), q[t]);
    plant().SetVelocities(context_.get(), v[t]);
    plant().CalcForceElementsContribution(*context_, f_ext);

    // Inverse dynamics computes M*a + D*v - k(q,v)
    *a = (v[t + 1] - v[t]) / time_step();
    tau->at(t) = plant().CalcInverseDynamics(*context_, *a, *f_ext);

    // CalcInverseDynamics considers damping from v_t (D*v_t), but we want to
    // consider damping from v_{t+1} (D*v_{t+1}).
    tau->at(t).array() +=
        joint_damping_.array() * (v[t + 1].array() - v[t].array());

    // TODO(vincekurtz) add in contact/constriant contribution
  }
}

void TrajectoryOptimizer::CalcInverseDynamicsPartials(
    const std::vector<VectorXd>& q, const std::vector<VectorXd>& v,
    GradientData* grad_data) const {
  // TODO(vincekurtz): use a solver flag to choose between finite differences
  // and an analytical approximation
  CalcInverseDynamicsPartialsFiniteDiff(q, v, grad_data);
}

void TrajectoryOptimizer::CalcInverseDynamicsPartialsFiniteDiff(
    const std::vector<VectorXd>& q, const std::vector<VectorXd>& v,
    GradientData* grad_data) const {
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
  VectorXd a(plant().num_velocities());    // scratch space
  MultibodyForces<double> f_ext(plant());  // scratch space
  std::vector<VectorXd> tau(num_steps());
  CalcTau(q, v, &a, &f_ext, &tau);

  // Purturbed versions of q, v, and tau
  std::vector<VectorXd> q_eps(q);
  std::vector<VectorXd> v_eps(v);
  std::vector<VectorXd> tau_eps(tau);

  const double eps = sqrt(std::numeric_limits<double>::epsilon());
  for (int t = 1; t <= num_steps(); ++t) {
    for (int i = 0; i < nq; ++i) {
      // Purturb q_t[i] by epsilon
      q_eps = q;
      q_eps[t](i) += eps;

      // Compute tau(q + S*epsilon), where S is a selection matrix for the i^th
      // element
      CalcV(q_eps, &v_eps);
      CalcTau(q_eps, v_eps, &a, &f_ext, &tau_eps);

      // Update the non-zero entries of dtau/dq
      dtau_dq[t].col(i) = (tau_eps[t] - tau[t]) / eps;
      dtau_dqp[t - 1].col(i) = (tau_eps[t - 1] - tau[t - 1]) / eps;
      if (t < num_steps()) {
        dtau_dqm[t + 1].col(i) = (tau_eps[t + 1] - tau[t + 1]) / eps;
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
