#include "drake/traj_opt/trajectory_optimizer.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/problem_definition.h"

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
  auto plant = std::make_unique<MultibodyPlant<double>>(dt);
  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/pendulum.urdf");
  Parser(plant.get()).AddAllModelsFromFile(urdf_file);
  plant->set_discrete_contact_solver(DiscreteContactSolver::kSap);
  plant->Finalize();
  auto plant_context = plant->CreateDefaultContext();

  // Define a sequence of ground-truth generalized forces
  std::vector<VectorXd> tau_gt;
  for (int t = 0; t < num_steps; ++t) {
    tau_gt.push_back(Vector1d(sin(t)));
  }

  // Simulate forward, recording the state
  MatrixXd x(plant->num_multibody_states(), num_steps + 1);
  auto x0 = x.leftCols<1>();
  x0 << 1.3, 0.4;
  plant->SetPositionsAndVelocities(plant_context.get(), x.leftCols<1>());

  auto state = plant->AllocateDiscreteVariables();
  for (int t = 0; t < num_steps; ++t) {
    plant->get_actuation_input_port().FixValue(plant_context.get(), tau_gt[t]);
    plant->CalcDiscreteVariableUpdates(*plant_context, state.get());
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
  opt_prob.T = num_steps;
  TrajectoryOptimizer optimizer(std::move(plant), opt_prob);

  // Compute v and tau from q
  std::vector<VectorXd> v(num_steps + 1);
  std::vector<VectorXd> tau(num_steps);
  optimizer.CalcV(q, &v);
  optimizer.CalcTau(q, v, &tau);

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

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
