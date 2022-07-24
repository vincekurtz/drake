#include "drake/traj_opt/trajectory_optimizer.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/solution_data.h"

namespace drake {
namespace traj_opt {
namespace internal {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using multibody::MultibodyPlant;
using multibody::Parser;


/**
 * Test our computation of generalized forces
 *
 *   tau_t = InverseDynamics(q_{t-1}, q_t, q_{t+1})
 *
 */
GTEST_TEST(TrajectoryOptimizerTest, PendulumCalcTau) {
  const int num_steps = 5;

  // Set up the system model
  // TODO(vincekurtz): set SAP as discrete contact solver
  auto plant = std::make_unique<MultibodyPlant<double>>(1e-2);
  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/pendulum.urdf");
  Parser(plant.get()).AddAllModelsFromFile(urdf_file);
  plant->Finalize();
  auto plant_context = plant->CreateDefaultContext();

  // Define sequence of control torques 
  std::vector<Vector1d> tau;
  for (int t=0; t<num_steps; ++t) {
    tau.emplace_back(Vector1d(sin(t)));
  }

  // Simulate forward, recording the state
  MatrixXd x(plant->num_multibody_states(), num_steps + 1);

  auto x0 = x.leftCols<1>();  // set initial state
  x0 << 1.3, 0.4;
  plant->SetPositionsAndVelocities(plant_context.get(), x.leftCols<1>());

  auto state = plant->AllocateDiscreteVariables();
  for (int i = 0; i < num_steps; ++i) {
    plant->get_actuation_input_port().FixValue(plant_context.get(), tau[i]);
    plant->CalcDiscreteVariableUpdates(*plant_context, state.get());
    plant_context->SetDiscreteState(state->get_value());
    x.col(i + 1) = state->get_value();
  }

  // Write generalized position trajectory (q) as a std::vector 
  std::vector<VectorXd> q;
  for (int t = 0; t <= num_steps; ++t) {
    q.emplace_back(x.block<1,1>(0,t));
  }

  // Create a trajectory optimizer
  ProblemDefinition opt_prob;
  opt_prob.q0 = x0.topRows<1>();
  opt_prob.v0 = x.bottomRows<1>();
  opt_prob.T = num_steps;
  TrajectoryOptimizer optimizer(std::move(plant), opt_prob);

  // Compute tau from q using the trajectory optimizer

  EXPECT_TRUE(1 == 0);
}

/**
 * Test our computation of generalized velocities
 *
 *   v_t = (q_t - q_{t-1})/dt
 *
 */
GTEST_TEST(TrajectoryOptimizerTest, PendulumCalcV) {
  const int num_steps = 5;

  // Set up the system model
  // TODO(vincekurtz): set SAP as discrete contact solver
  auto plant = std::make_unique<MultibodyPlant<double>>(1e-2);
  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/pendulum.urdf");
  Parser(plant.get()).AddAllModelsFromFile(urdf_file);
  plant->Finalize();
  auto plant_context = plant->CreateDefaultContext();

  // Simulate forward, recording the state
  MatrixXd x(plant->num_multibody_states(), num_steps + 1);
  auto q = x.topRows<1>();
  auto v = x.bottomRows<1>();

  auto x0 = x.leftCols<1>();  // set initial state
  x0 << 1.3, 0.4;
  plant->SetPositionsAndVelocities(plant_context.get(), x.leftCols<1>());

  auto state = plant->AllocateDiscreteVariables();
  for (int i = 0; i < num_steps; ++i) {
    // Use some non-zero actuations to make things more interesting
    plant->get_actuation_input_port().FixValue(plant_context.get(),
                                               sin(plant->time_step() * i));

    plant->CalcDiscreteVariableUpdates(*plant_context, state.get());
    plant_context->SetDiscreteState(state->get_value());
    x.col(i + 1) = state->get_value();
  }

  // Create a trajectory optimizer
  ProblemDefinition opt_prob;
  opt_prob.q0 = q.col(0);
  opt_prob.v0 = v.col(0);
  opt_prob.T = num_steps;
  TrajectoryOptimizer optimizer(std::move(plant), opt_prob);

  // Trajectory optimizer computes v from q
  std::vector<VectorXd> q_vector;
  for (int i = 0; i <= num_steps; ++i) {
    q_vector.emplace_back(q.col(i));
  }

  std::vector<VectorXd> v_vector(num_steps + 1);
  optimizer.CalcV(q_vector, &v_vector);

  // Check that our computed v matches the recorded v
  MatrixXd v_computed(v.rows(), v.cols());
  for (int i = 0; i <= num_steps; ++i) {
    v_computed.col(i) = v_vector[i];
  }

  const double kTolerance = std::numeric_limits<double>::epsilon() * 100;
  EXPECT_TRUE(
      CompareMatrices(v, v_computed, kTolerance, MatrixCompareType::relative));
}

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
