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
  (void)q;
  (void)v;
  (void)grad_data;
  std::cout << "hello world" << std::endl;
}

void TrajectoryOptimizer::CalcDtaumDq(const std::vector<VectorXd>& q,
                                      const int t,
                                      Eigen::Ref<MatrixXd> dtaum_dq) const {
  // TODO(vincekurtz): use a more efficient approximation
  CalcDtausDqtFiniteDiff(q, t-1, t, dtaum_dq);
}

void TrajectoryOptimizer::CalcDtauDq(const std::vector<VectorXd>& q,
                                      const int t,
                                      Eigen::Ref<MatrixXd> dtaum_dq) const {
  // TODO(vincekurtz): use a more efficient approximation
  CalcDtausDqtFiniteDiff(q, t, t, dtaum_dq);
}

void TrajectoryOptimizer::CalcDtaupDq(const std::vector<VectorXd>& q,
                                      const int t,
                                      Eigen::Ref<MatrixXd> dtaum_dq) const {
  // TODO(vincekurtz): use a more efficient approximation
  CalcDtausDqtFiniteDiff(q, t+1, t, dtaum_dq);
}

void TrajectoryOptimizer::CalcDtausDqtFiniteDiff(
    const std::vector<VectorXd>& q, const int s, const int t,
    Eigen::Ref<MatrixXd> dtaus_dqt) const {
  DRAKE_DEMAND(dtaus_dqt.rows() ==          // nv and not nu, since tau is
               plant().num_velocities());  // generalized forces, not control
  DRAKE_DEMAND(dtaus_dqt.cols() == plant().num_positions());

  // Compute generalized forces from q. This is very gross and brute-force,
  // since we compute everything for every timestep.
  std::vector<VectorXd> v(num_steps() + 1);
  VectorXd a(plant().num_velocities());
  MultibodyForces<double> f_ext(plant());
  std::vector<VectorXd> tau(num_steps());
  CalcV(q, &v);
  CalcTau(q, v, &a, &f_ext, &tau);

  // Modulate qt[i] to define each row of dtaum_dq
  std::vector<VectorXd> q_eps = q;
  std::vector<VectorXd> tau_eps(num_steps());

  const double eps = 100*sqrt(std::numeric_limits<double>::epsilon());
  for (int i = 0; i < plant().num_positions(); ++i) {
    q_eps[t](i) += eps;
    CalcV(q_eps, &v);
    CalcTau(q_eps, v, &a, &f_ext, &tau_eps);
    dtaus_dqt.row(i) = (tau_eps[s] - tau[s]) / eps;
  }
}

}  // namespace traj_opt
}  // namespace drake
