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
using multibody::DiscreteContactSolver;
using multibody::MultibodyPlant;
using multibody::Parser;

/**
 * Test our computation of generalized velocities
 *
 *   v_t = (q_t - q_{t-1})/dt
 *
 */
GTEST_TEST(TrajectoryOptimizerTest, PendulumCalcV) {
  const int num_steps = 5;

  // Set up the system model
  auto plant = std::make_unique<MultibodyPlant<double>>(1e-2);
  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/pendulum.urdf");
  Parser(plant.get()).AddAllModelsFromFile(urdf_file);
  plant->set_discrete_contact_solver(DiscreteContactSolver::kSap);
  plant->Finalize();
  auto plant_context = plant->CreateDefaultContext();

  // Simulate forward, recording the state
  MatrixXd x(plant->num_multibody_states(), num_steps + 1);
  auto x0 = x.leftCols<1>();
  x0 << 1.3, 0.4;
  plant->SetPositionsAndVelocities(plant_context.get(), x.leftCols<1>());

  auto state = plant->AllocateDiscreteVariables();
  for (int t = 0; t < num_steps; ++t) {
    plant->get_actuation_input_port().FixValue(plant_context.get(), sin(t));
    plant->CalcDiscreteVariableUpdates(*plant_context, state.get());
    plant_context->SetDiscreteState(state->get_value());
    x.col(t + 1) = state->get_value();
  }

  // Formate a std::vector of generalized positions (q)
  std::vector<VectorXd> q;
  for (int t = 0; t <= num_steps; ++t) {
    q.emplace_back(x.block<1, 1>(0, t));
  }

  // Create a trajectory optimizer
  ProblemDefinition opt_prob;
  opt_prob.q0 = x.block<1, 1>(0, 0);
  opt_prob.v0 = x.block<1, 1>(1, 0);
  opt_prob.T = num_steps;
  TrajectoryOptimizer optimizer(std::move(plant), opt_prob);

  // Compute v as from q using the optimizer
  std::vector<VectorXd> v(num_steps + 1);
  optimizer.CalcV(q, &v);

  // Check that our computed v matches the recorded v
  const double kTolerance = std::numeric_limits<double>::epsilon() * 100;
  for (int t = 0; t <= num_steps; ++t) {
    EXPECT_TRUE(CompareMatrices(v[t], x.block<1, 1>(1, t), kTolerance,
                                MatrixCompareType::relative));
  }
}

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
