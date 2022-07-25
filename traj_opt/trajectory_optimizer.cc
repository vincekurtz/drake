#include "drake/traj_opt/trajectory_optimizer.h"

#include <iostream>

namespace drake {
namespace traj_opt {

using multibody::MultibodyPlant;
using multibody::MultibodyForces;
using multibody::JointIndex;
using multibody::Joint;
using systems::System;

TrajectoryOptimizer::TrajectoryOptimizer(
    std::unique_ptr<const MultibodyPlant<double>> plant,
    const ProblemDefinition& prob)
    : prob_(prob) {
  plant_ = std::move(plant);
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
  DRAKE_DEMAND(static_cast<int>(q.size()) == T() + 1);
  DRAKE_DEMAND(static_cast<int>(v->size()) == T() + 1);

  v->at(0) = prob_.v_init;
  for (int i = 1; i <= T(); ++i) {
    v->at(i) = (q[i] - q[i - 1]) / time_step();
  }
}

void TrajectoryOptimizer::CalcTau(const std::vector<VectorXd>& q,
                                  const std::vector<VectorXd>& v,
                                  std::vector<VectorXd>* tau) const {
  // Generalized forces aren't defined for the last timestep
  // TODO(vincekurtz): additional checks that q_t, v_t, tau_t are the right size
  // for the plant?
  DRAKE_DEMAND(static_cast<int>(q.size()) == T() + 1);
  DRAKE_DEMAND(static_cast<int>(v.size()) == T() + 1);
  DRAKE_DEMAND(static_cast<int>(tau->size()) == T());

  const int nv = plant().num_velocities();
  VectorXd a(nv);                          // acceleration
  MultibodyForces<double> f_ext(plant());  // external forces

  for (int t = 0; t < T(); ++t) {
    a = (v[t+1] - v[t])/time_step();
    plant().SetPositions(context_.get(), q[t]);
    plant().SetVelocities(context_.get(), v[t]);
    plant().CalcForceElementsContribution(*context_, &f_ext);

    // Inverse dynamics computes M*a + D*v - k(q,v)
    tau->at(t) = plant().CalcInverseDynamics(*context_, a, f_ext);

    // CalcInverseDynamics considers damping from v_t (D*v_t), but we want to
    // consider damping from v_{t+1} (D*v_{t+1}).
    tau->at(t).array() += joint_damping_.array() * (v[t+1].array() - v[t].array());

    // TODO(vincekurtz) add in contact/constriant contribution
  }

}

}  // namespace traj_opt
}  // namespace drake
