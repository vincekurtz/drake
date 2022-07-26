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
  // TODO(vincekurtz): use an analytical approximation rather than finite differences
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
  std::vector<MatrixXd> dtaum_dq(num_steps()+1, MatrixXd(nv, nq));
  std::vector<MatrixXd> dtau_dq(num_steps()+1, MatrixXd(nv, nq));
  std::vector<MatrixXd> dtaup_dq(num_steps()+1, MatrixXd(nv, nq));

  // All derivatives w.r.t. q0 are zero, since q0 = q_init is fixed. We only
  // include them in GradientData so we can index by t. 
  dtaum_dq[0].setZero();
  dtau_dq[0].setZero();
  dtaup_dq[0].setZero();

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

      // Update the nozero entries of dtau_t/dq_t
      dtaum_dq[t].col(i) = (tau_eps[t-1] - tau[t-1]) / eps;

      if ( t == num_steps() ) {
        dtau_dq[t].setZero();  // tau[num_steps] is undefined
      } else {
        dtau_dq[t].col(i) = (tau_eps[t] - tau[t]) / eps;
      }

      if ( (t == num_steps()) || (t == (num_steps()-1) )) {
        dtaup_dq[t].setZero();  // tau[num_steps (+ 1)] is undefined
      } else {
        dtaup_dq[t].col(i) = (tau_eps[t+1] - tau[t+1]) / eps;
      }

    }
  }

  // Put the results into the GradientData struct
  grad_data->dtaum_dq = dtaum_dq;
  grad_data->dtau_dq = dtau_dq;
  grad_data->dtaup_dq = dtaup_dq;
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
