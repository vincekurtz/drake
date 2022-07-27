#include "drake/traj_opt/trajectory_optimizer.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/problem_data.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer_workspace.h"

namespace drake {
namespace traj_opt {
namespace internal {

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;
using multibody::DiscreteContactSolver;
using multibody::MultibodyPlant;
using multibody::Parser;
using test::LimitMalloc;

GTEST_TEST(TrajectoryOptimizerTest, PendulumDtauDq) {
  const int num_steps = 5;
  const double dt = 1e-2;

  // Set up a system model
  MultibodyPlant<double> plant(dt);
  const std::string urdf_file =
      FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.set_discrete_contact_solver(DiscreteContactSolver::kSap);
  plant.Finalize();

  // Create a trajectory optimizer
  ProblemDefinition opt_prob;
  opt_prob.q_init = Vector1d(0.0);
  opt_prob.v_init = Vector1d(0.1);
  opt_prob.num_steps = num_steps;
  TrajectoryOptimizerWorkspace workspace(plant);
  TrajectoryOptimizer optimizer(&plant, opt_prob);

  // Create some fake data
  std::vector<VectorXd> q;
  q.push_back(opt_prob.q_init);
  for (int t = 1; t <= num_steps; ++t) {
    q.push_back(Vector1d(0.0 + 0.6 * t));
  }

  // Compute inverse dynamics partials
  GradientData grad_data;
  std::vector<VectorXd> v(num_steps + 1);
  optimizer.CalcV(q, &v);
  optimizer.CalcInverseDynamicsPartials(q, v, &workspace, &grad_data);

  // Compute ground truth partials from the pendulum model
  //
  //     m*l^2*a + m*g*l*sin(q) + b*v = tau
  //
  // where q is the joint angle and a = dv/dt, v = dq/dt.
  const double m = 1.0;
  const double l = 0.5;
  const double b = 0.1;
  const double g = 9.81;

  GradientData grad_data_gt;
  MatrixXd dtaut_dqtm(1, 1);
  MatrixXd dtaut_dqt(1, 1);
  MatrixXd dtaut_dqtp(1, 1);
  for (int t = 0; t < num_steps; ++t) {
    dtaut_dqtp(0, 0) = 1 / dt / dt * m * l * l + 1 / dt * b;
    dtaut_dqt(0, 0) =
        -2 / dt / dt * m * l * l - 1 / dt * b + m * g * l * cos(q[t](0));
    dtaut_dqtm(0, 0) = 1 / dt / dt * m * l * l;

    // q0 = q_init is fixed, so all the derivatives w.r.t. q0 are zero
    if (t == 0) {
      dtaut_dqtm(0, 0) = 0;
      dtaut_dqt(0, 0) = 0;
    } else if (t == 1) {
      dtaut_dqtm(0, 0) = 0;
    }

    grad_data_gt.dtau_dqm.push_back(dtaut_dqtm);
    grad_data_gt.dtau_dq.push_back(dtaut_dqt);
    grad_data_gt.dtau_dqp.push_back(dtaut_dqtp);
  }

  // Compare the computed values and the analytical ground truth
  const double kTolerance = sqrt(std::numeric_limits<double>::epsilon());
  for (int t = 0; t < num_steps; ++t) {
    EXPECT_TRUE(CompareMatrices(grad_data.dtau_dqm[t], grad_data_gt.dtau_dqm[t],
                                kTolerance, MatrixCompareType::relative));
    EXPECT_TRUE(CompareMatrices(grad_data.dtau_dq[t], grad_data_gt.dtau_dq[t],
                                kTolerance, MatrixCompareType::relative));
    EXPECT_TRUE(CompareMatrices(grad_data.dtau_dqp[t], grad_data_gt.dtau_dqp[t],
                                kTolerance, MatrixCompareType::relative));
  }
}

/**
 * Test our computation of generalized velocities
 *
 *   v_t = (q_t - q_{t-1})/dt
 *
 * and generalized forces
 *
 *   tau_t = InverseDynamics(a_t, v_t, q_t)
 *
 * where a_t = (v_{t+1}-v_t)/dt.
 *
 */
GTEST_TEST(TrajectoryOptimizerTest, PendulumCalcVAndTau) {
  const int num_steps = 5;
  const double dt = 1e-2;

  // Set up the system model
  MultibodyPlant<double> plant(dt);
  const std::string urdf_file =
      FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.set_discrete_contact_solver(DiscreteContactSolver::kSap);
  plant.Finalize();
  auto plant_context = plant.CreateDefaultContext();

  // Define a sequence of ground-truth generalized forces
  std::vector<VectorXd> tau_gt;
  for (int t = 0; t < num_steps; ++t) {
    tau_gt.push_back(Vector1d(sin(t)));
  }

  // Simulate forward, recording the state
  MatrixXd x(plant.num_multibody_states(), num_steps + 1);
  auto x0 = x.leftCols<1>();
  x0 << 1.3, 0.4;
  plant.SetPositionsAndVelocities(plant_context.get(), x.leftCols<1>());

  auto state = plant.AllocateDiscreteVariables();
  for (int t = 0; t < num_steps; ++t) {
    plant.get_actuation_input_port().FixValue(plant_context.get(), tau_gt[t]);
    plant.CalcDiscreteVariableUpdates(*plant_context, state.get());
    plant_context->SetDiscreteState(state->get_value());
    x.col(t + 1) = state->get_value();
  }

  // Construct a std::vector of generalized positions (q)
  std::vector<VectorXd> q;
  for (int t = 0; t <= num_steps; ++t) {
    q.emplace_back(x.block<1, 1>(0, t));
  }

  // Create a trajectory optimizer object
  ProblemDefinition opt_prob;
  opt_prob.q_init = x0.topRows<1>();
  opt_prob.v_init = x0.bottomRows<1>();
  opt_prob.num_steps = num_steps;
  TrajectoryOptimizerWorkspace workspace(plant);
  TrajectoryOptimizer optimizer(&plant, opt_prob);

  // Compute v and tau from q
  std::vector<VectorXd> v(num_steps + 1, VectorXd(1));
  std::vector<VectorXd> tau(num_steps, VectorXd(1));
  {
    // It appears, via trial and error, that CalcInverseDynamics makes exactly
    // 15 allocations for this example.
    LimitMalloc guard({.max_num_allocations = 15});
    optimizer.CalcV(q, &v);
    optimizer.CalcTau(q, v, &workspace, &tau);
  }

  // Check that our computed values match the true (recorded) ones
  const double kToleranceV = std::numeric_limits<double>::epsilon() / dt;
  for (int t = 0; t <= num_steps; ++t) {
    EXPECT_TRUE(CompareMatrices(v[t], x.block<1, 1>(1, t), kToleranceV,
                                MatrixCompareType::relative));
  }

  const double kToleranceTau = std::numeric_limits<double>::epsilon() / dt / dt;
  for (int t = 0; t < num_steps; ++t) {
    EXPECT_TRUE(CompareMatrices(tau[t], tau_gt[t], kToleranceTau,
                                MatrixCompareType::relative));
  }
}

/**
 * Test our computation of generalized velocities
 *
 *   v_t = (q_t - q_{t-1})/dt
 *
 */
GTEST_TEST(TrajectoryOptimizerTest, CalcV) {
  const int num_steps = 5;
  const double dt = 1e-2;

  // Create a TrajectoryOptimizer object
  MultibodyPlant<double> plant(dt);
  plant.Finalize();
  ProblemDefinition opt_prob;
  opt_prob.q_init = Vector2d(0.1, 0.2);
  opt_prob.v_init = Vector2d(0.5 / dt, 1.5 / dt);
  opt_prob.num_steps = num_steps;
  TrajectoryOptimizer optimizer(&plant, opt_prob);

  // Construct a std::vector of generalized positions (q)
  // where q(t) = [0.1 + 0.5*t]
  //              [0.2 + 1.5*t]
  std::vector<VectorXd> q;
  for (int t = 0; t <= num_steps; ++t) {
    q.push_back(Vector2d(0.1 + 0.5 * t, 0.2 + 1.5 * t));
  }

  // Compute v from q
  std::vector<VectorXd> v(num_steps + 1);
  optimizer.CalcV(q, &v);

  // Check that our computed v is correct
  const double kTolerance = std::numeric_limits<double>::epsilon() / dt;
  for (int t = 0; t <= num_steps; ++t) {
    EXPECT_TRUE(CompareMatrices(v[t], opt_prob.v_init, kTolerance,
                                MatrixCompareType::relative));
  }
}

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
