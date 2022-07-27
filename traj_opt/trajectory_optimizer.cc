#include "drake/traj_opt/trajectory_optimizer.h"

#include <iostream>

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

double TrajectoryOptimizer::CalcCost(const std::vector<VectorXd>& q,
                                     const std::vector<VectorXd>& v,
                                     const std::vector<VectorXd>& tau) const {
  double cost = 0;
  // TODO(vincekurtz): add q_err and v_err into the
  // TrajectoryOptimizerWorkspace, once #17 lands
  VectorXd q_err;
  VectorXd v_err;

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

  return cost;
}

double TrajectoryOptimizer::CalcCost(const std::vector<VectorXd>& q) const {
  std::vector<VectorXd> v(num_steps() + 1);
  std::vector<VectorXd> tau(num_steps());

  // TODO: replace with workspace
  VectorXd a;
  MultibodyForces<double> f_ext(plant());

  CalcV(q, &v);
  CalcTau(q, v, &a, &f_ext, &tau);

  return CalcCost(q, v, tau);
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

void TrajectoryOptimizer::CalcGradientFiniteDiff(const std::vector<VectorXd>& q, EigenPtr<VectorXd> g) const {
  // Compute the baseline cost
  double L = CalcCost(q);
  std::cout << L << std::endl;

  // Perturbed versions of q and L
  // TODO(vincekurtz): store in the workspace
  std::vector<VectorXd> q_eps(q);

  // Set first block of g (derivatives w.r.t. q_0) to zero, since q0 = q_init are constant.
  g->topRows(plant().num_positions()).setZero();

  // Iterate through rows of g using finite differences
  double eps = sqrt(std::numeric_limits<double>::epsilon());
  int j = plant().num_positions();
  for (int t=1; t<=num_steps(); ++t ) {
    for (int i=0; i<plant().num_positions(); ++i) {
      // Perturb Q_j
      q_eps[t](i) += eps;

      // Set g_j = ( L(Q) + L(Q+epsilon) ) / epsilon
      double L_eps = CalcCost(q_eps);
      (*g)(j) = (L + L_eps) / eps;

      // reset our perturbed Q and move to the next row of g.
      q_eps[t](i) = q[t](i);
      ++j;
    }
  }

  std::cout << *g << std::endl;

}

}  // namespace traj_opt
}  // namespace drake
