#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/traj_opt/trajectory_optimizer.h"

#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace drake {
namespace traj_opt {
namespace internal {

using Eigen::Vector3d;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;

/**
 * Try to approximate derivatives through contact by freezing the contact
 * Jacobian.
 */
GTEST_TEST(FreezeContactJacobianTest, Spinner) {
  // Set up a simple system with contact
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = 1.0;
  auto [plant, scene_graph] = multibody::AddMultibodyPlant(config, &builder);
  Parser(&plant).AddAllModelsFromFile(FindResourceOrThrow(
      "drake/traj_opt/examples/models/spinner_sphere.urdf"));
  plant.Finalize();
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  Context<double>* plant_context =
      &diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set up an optimization problem
  const int num_steps = 1;
  ProblemDefinition opt_prob;
  opt_prob.num_steps = num_steps;
  opt_prob.q_init = Vector3d(0.2, 1.5, 0.0);
  opt_prob.v_init = Vector3d(0.0, 0.0, 0.0);
  for (int t = 0; t <= num_steps; ++t) {
    opt_prob.q_nom.push_back(Vector3d(0.4, 1.5, 0.0));
    opt_prob.v_nom.push_back(Vector3d(0.0, 0.0, 0.0));
  }
  SolverParameters solver_params;
  solver_params.F = 1.0;
  solver_params.delta = 0.01;
  solver_params.dissipation_velocity = 0.1;
  solver_params.force_at_a_distance = true;
  solver_params.smoothing_factor = 1.0;
  solver_params.friction_coefficient = 0.0;  // no friction
  solver_params.stiction_velocity = 0.1;

  // Create a double-type optimizer
  TrajectoryOptimizer<double> optimizer(diagram.get(), &plant, opt_prob,
                                        solver_params);
  TrajectoryOptimizerState<double> state = optimizer.CreateState();

  // Create an autodiff optimizer
  auto diagram_ad = systems::System<double>::ToAutoDiffXd(*diagram);
  const auto& plant_ad = dynamic_cast<const MultibodyPlant<AutoDiffXd>&>(
      diagram_ad->GetSubsystemByName(plant.get_name()));
  auto diagram_context_ad = diagram_ad->CreateDefaultContext();
  Context<AutoDiffXd>* plant_context_ad =
      &diagram_ad->GetMutableSubsystemContext(plant_ad,
                                              diagram_context_ad.get());
  TrajectoryOptimizer<AutoDiffXd> optimizer_ad(diagram_ad.get(), &plant_ad,
                                               opt_prob, solver_params);
  TrajectoryOptimizerState<AutoDiffXd> state_ad = optimizer_ad.CreateState();

  // Create some fake data
  std::vector<VectorXd> q;
  q.push_back(opt_prob.q_init);
  q.push_back(Vector3d(0.4, 1.5, 0.0));
  state.set_q(q);

  std::vector<VectorX<AutoDiffXd>> q_ad;
  q_ad.push_back(q[0]);
  q_ad.push_back(math::InitializeAutoDiff(q[1]));
  state_ad.set_q(q_ad);

  // Compute inverse dynamics as τ = ID(q, v, a, γ)
  const std::vector<VectorXd>& tau = optimizer.EvalTau(state);
  const VectorXd tau_gt = tau[0];

  // Compute inverse dynamics as τ = ID(q, v, a) - J'γ
  const std::vector<VectorXd>& v = optimizer.EvalV(state);
  const std::vector<VectorXd>& a = optimizer.EvalA(state);
  MultibodyForces<double> f_ext(plant);
  plant.SetPositions(plant_context, q[1]);
  plant.SetVelocities(plant_context, v[1]);
  plant.CalcForceElementsContribution(*plant_context, &f_ext);
  const VectorXd tau_id =
      plant.CalcInverseDynamics(*plant_context, a[0], f_ext);

  const TrajectoryOptimizerCache<double>::ContactJacobianData jacobian_data =
      optimizer.EvalContactJacobianData(state);
  const MatrixXd& J = jacobian_data.J[1];

  std::vector<VectorXd> gamma(num_steps + 1);
  optimizer.CalcContactImpulses(state, &gamma);
  const VectorXd tau_c = J.transpose() * gamma[1];
  const VectorXd tau_separate = tau_id - tau_c;

  // Verify that ID(q, v, a, γ) = ID(q, v, a) - J'γ
  PRINT_VAR(tau_gt.transpose());
  PRINT_VAR(tau_separate.transpose());
  const double kTolerance = 100 * std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(CompareMatrices(tau_gt, tau_separate, kTolerance,
                              MatrixCompareType::relative));

  // Compute inverse dynamics both ways with autodiff
  const std::vector<VectorX<AutoDiffXd>>& tau_ad =
      optimizer_ad.EvalTau(state_ad);
  const VectorX<AutoDiffXd> tau_gt_ad = tau_ad[0];

  const std::vector<VectorX<AutoDiffXd>>& v_ad = optimizer_ad.EvalV(state_ad);
  const std::vector<VectorX<AutoDiffXd>>& a_ad = optimizer_ad.EvalA(state_ad);
  MultibodyForces<AutoDiffXd> f_ext_ad(plant_ad);
  plant_ad.SetPositions(plant_context_ad, q_ad[1]);
  plant_ad.SetVelocities(plant_context_ad, v_ad[1]);
  plant_ad.CalcForceElementsContribution(*plant_context_ad, &f_ext_ad);
  const VectorX<AutoDiffXd> tau_id_ad =
      plant_ad.CalcInverseDynamics(*plant_context_ad, a_ad[0], f_ext_ad);
  const TrajectoryOptimizerCache<AutoDiffXd>::ContactJacobianData
      jacobian_data_ad = optimizer_ad.EvalContactJacobianData(state_ad);
  const MatrixX<AutoDiffXd>& J_ad = jacobian_data_ad.J[1];
  std::vector<VectorX<AutoDiffXd>> gamma_ad(num_steps + 1);
  optimizer_ad.CalcContactImpulses(state_ad, &gamma_ad);
  const VectorX<AutoDiffXd> tau_c_ad = J_ad.transpose() * gamma_ad[1];
  const VectorX<AutoDiffXd> tau_separate_ad = tau_id_ad - tau_c_ad;

  // Verify inverse dynamics with autodiff match what we get with double
  EXPECT_TRUE(CompareMatrices(tau_gt, math::ExtractValue(tau_gt_ad), kTolerance,
                              MatrixCompareType::relative));
  EXPECT_TRUE(CompareMatrices(tau_separate, math::ExtractValue(tau_separate_ad),
                              kTolerance, MatrixCompareType::relative));
  EXPECT_TRUE(CompareMatrices(math::ExtractValue(tau_gt_ad),
                              math::ExtractValue(tau_separate_ad), kTolerance,
                              MatrixCompareType::relative));

  // Verify that we get the same derivatives when we autodiff thru everything,
  // i.e., ∂/∂q[ID(q, v, a, γ)] = ∂/∂q[ID(q, v, a) - J'γ]
  const MatrixXd dtau_dq_gt = math::ExtractGradient(tau_gt_ad);
  const MatrixXd dtau_dq_separate = math::ExtractGradient(tau_separate_ad);

  EXPECT_TRUE(CompareMatrices(dtau_dq_gt, dtau_dq_separate, kTolerance,
                              MatrixCompareType::relative));

  // Try to approximate the derivatives with a frozen Jacobian, i.e.
  // ∂/∂q[ID(q, v, a)] - J'∂/∂q[γ]
  const MatrixXd dtau_id_dq = math::ExtractGradient(tau_id_ad);
  const MatrixXd dgamma_dq = math::ExtractGradient(gamma_ad[1]);
  const MatrixXd dtau_dq_approx = dtau_id_dq - J.transpose() * dgamma_dq;

  PRINT_VARn(dtau_dq_gt);
  PRINT_VARn(dtau_dq_approx);

  // Verify that the approximation error is due to the missing Jacobian terms
  const MatrixXd missing_terms =
      math::ExtractGradient(-J_ad.transpose() * gamma[1]);
  EXPECT_TRUE(CompareMatrices(dtau_dq_gt, dtau_dq_approx + missing_terms,
                              kTolerance, MatrixCompareType::relative));
}

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
