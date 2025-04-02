#include "drake/systems/analysis/convex_integrator.h"

#include <iostream>

#include <gtest/gtest.h>

#include "drake/common/drake_assert.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/controllers/pid_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/visualization/visualization_config_functions.h"

namespace drake {
namespace systems {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::JointActuator;
using multibody::JointActuatorIndex;
using multibody::MultibodyPlant;
using multibody::Parser;
using systems::ConstantVectorSource;
using systems::FirstOrderTaylorApproximation;
using systems::controllers::PidController;
using visualization::AddDefaultVisualization;

class ConvexIntegratorTester {
 public:
  ConvexIntegratorTester() = delete;

  static void LinearizeExternalSystem(ConvexIntegrator<double>* integrator,
                                      const double h, VectorXd* K,
                                      VectorXd* u0) {
    integrator->LinearizeExternalSystem(h, K, u0);
  }

  static SapContactProblem<double> MakeSapContactProblem(
      ConvexIntegrator<double>* integrator, const Context<double>& context,
      const double h) {
    return integrator->MakeSapContactProblem(context, h);
  }
};

// MJCF model of a simple double pendulum
const char double_pendulum_xml[] = R"""(
<?xml version="1.0"?>
<mujoco model="double_pendulum">
<worldbody>
  <body>
  <joint type="hinge" axis="0 1 0" pos="0 0 0.1" damping="1e-3"/>
  <geom type="capsule" size="0.01 0.1"/>
  <body>
    <joint type="hinge" axis="0 1 0" pos="0 0 -0.1" damping="1e-3"/>
    <geom type="capsule" size="0.01 0.1" pos="0 0 -0.2"/>
  </body>
  </body>
</worldbody>
</mujoco> 
)""";

// MJCF model of a cylinder falling on a table
const char cylinder_xml[] = R"""(
<?xml version="1.0"?>
<mujoco model="robot">
<worldbody>
  <geom name="table_top" type="box" pos="0.0 0.0 0.0" size="0.55 1.1 0.05" rgba="0.9 0.8 0.7 1"/>
  <body>
    <joint type="free"/>
    <geom name="object" type="cylinder" pos="0.0 0.0 0.5" euler="80 0 0" size="0.1 0.1" rgba="1.0 1.0 1.0 1.0"/>
  </body>
</worldbody>
</mujoco>
)""";

// MJCF model of an actuated double pendulum
const char actuated_pendulum_xml[] = R"""(
<?xml version="1.0"?>
<mujoco model="robot">
  <worldbody>
    <body>
      <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.1"/>
      <geom type="capsule" size="0.01 0.1"/>
      <body>
        <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 -0.1"/>
        <geom type="capsule" size="0.01 0.1" pos="0 0 -0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1" ctrlrange="-2 2"/>
    <motor joint="joint2" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
)""";

// Simulate a simple double pendulum system with the convex integrator. Play
// the sim back over meshcat so we can see what's going on.
GTEST_TEST(ConvexIntegratorTest, DoublePendulumSim) {
  // Start meshcat
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();

  // Set up the system diagram
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);
  Parser(&plant, &scene_graph).AddModelsFromString(double_pendulum_xml, "xml");
  plant.Finalize();
  AddDefaultVisualization(&builder, meshcat);
  auto diagram = builder.Build();

  // Create context and set the initial state
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  Eigen::Vector2d q0(3.0, 0.1);
  plant.SetPositions(&plant_context, q0);

  // Set up the simulator
  Simulator<double> simulator(*diagram, std::move(diagram_context));
  SimulatorConfig config;
  config.target_realtime_rate = 1.0;
  config.publish_every_time_step = true;
  ApplySimulatorConfig(config, &simulator);

  // Set the integrator
  ConvexIntegrator<double>& integrator =
      simulator.reset_integrator<ConvexIntegrator<double>>();
  integrator.set_maximum_step_size(0.1);
  simulator.Initialize();

  // Simulate for a few seconds
  const int fps = 64;
  meshcat->StartRecording(fps);
  simulator.AdvanceTo(0.1);
  meshcat->StopRecording();
  meshcat->PublishRecording();

  std::cout << std::endl;
  PrintSimulatorStatistics(simulator);
  std::cout << std::endl;
}

// Run a short simulation with the convex integrator. Most useful as a quick
// sanity check.
GTEST_TEST(ConvexIntegratorTest, ShortSim) {
  // Time step
  const double h = 0.01;

  // Create a continuous-time system
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);
  Parser(&plant, &scene_graph).AddModelsFromString(cylinder_xml, "xml");
  plant.Finalize();
  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  ConvexIntegrator<double>& integrator =
      simulator.reset_integrator<ConvexIntegrator<double>>();
  integrator.set_maximum_step_size(h);
  simulator.Initialize();

  simulator.AdvanceTo(1.0);
}

// Simulate a cylinder falling on a table with the convex integrator. Play
// the sim back over meshcat so we can see what's going on.
GTEST_TEST(ConvexIntegratorTest, CylinderSim) {
  // Start meshcat
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();

  // Set up the system diagram
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);
  Parser(&plant, &scene_graph).AddModelsFromString(cylinder_xml, "xml");
  plant.Finalize();
  AddDefaultVisualization(&builder, meshcat);

  // Set up hydroelastic contact
  geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.compliance_type = "compliant";
  scene_graph.set_config(scene_graph_config);
  auto diagram = builder.Build();

  // Set up the simulator
  Simulator<double> simulator(*diagram);
  SimulatorConfig config;
  config.target_realtime_rate = 1.0;
  config.publish_every_time_step = true;
  ApplySimulatorConfig(config, &simulator);

  // Set the integrator
  ConvexIntegrator<double>& integrator =
      simulator.reset_integrator<ConvexIntegrator<double>>();
  integrator.set_maximum_step_size(0.1);
  simulator.Initialize();

  // Simulate for a few seconds
  const int fps = 32;
  meshcat->StartRecording(fps);
  simulator.AdvanceTo(10.0);
  meshcat->StopRecording();
  meshcat->PublishRecording();

  std::cout << std::endl;
  PrintSimulatorStatistics(simulator);
  std::cout << std::endl;
}

// Run tests with an actuated pendulum and an external (PID) controller
// GTEST_TEST(ConvexIntegratorTest, ActuatedPendulum) {
//   // Some options
//   const double h = 0.01;

//   VectorXd Kp(2), Kd(2), Ki(2);
//   Kp << 0.24, 0.19;
//   Kd << 0.35, 0.3;
//   Ki << 0.0, 0.0;

//   // Start meshcat
//   auto meshcat = std::make_shared<drake::geometry::Meshcat>();

//   // Set up the system diagram
//   DiagramBuilder<double> builder;
//   auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);
//   Parser(&plant, &scene_graph)
//       .AddModelsFromString(actuated_pendulum_xml, "xml");
//   plant.Finalize();

//   AddDefaultVisualization(&builder, meshcat);

//   VectorXd x_nom(4);
//   x_nom << M_PI_2, M_PI_2, 0.0, 0.0;
//   auto target_state = builder.AddSystem<ConstantVectorSource<double>>(x_nom);

//   // PD controller is an external system
//   auto ctrl = builder.AddSystem<PidController>(Kp, Ki, Kd);

//   builder.Connect(target_state->get_output_port(),
//                   ctrl->get_input_port_desired_state());
//   builder.Connect(plant.get_state_output_port(),
//                   ctrl->get_input_port_estimated_state());
//   builder.Connect(ctrl->get_output_port(), plant.get_actuation_input_port());

//   auto diagram = builder.Build();

//   // Set up the simulator
//   Simulator<double> simulator(*diagram);
//   SimulatorConfig config;
//   config.target_realtime_rate = 1.0;
//   config.publish_every_time_step = true;
//   ApplySimulatorConfig(config, &simulator);

//   ConvexIntegrator<double>& integrator =
//       simulator.reset_integrator<ConvexIntegrator<double>>();
//   integrator.set_requested_minimum_step_size(1e-5);
//   integrator.set_fixed_step_mode(true);
//   integrator.set_maximum_step_size(h);

//   // Set an interesting initial state
//   VectorXd q0(2), v0(2);
//   q0 << 0.1, 0.2;
//   v0 << 0.3, 0.4;
//   Context<double>& plant_context =
//       plant.GetMyMutableContextFromRoot(&simulator.get_mutable_context());
//   plant.SetPositions(&plant_context, q0);
//   plant.SetVelocities(&plant_context, v0);
//   simulator.Initialize();

//   // Linearize the non-plant system dynamics around the current state
//   const int nv = plant.num_velocities();
//   VectorXd A(nv, nv);
//   VectorXd tau(nv);
//   ConvexIntegratorTester::LinearizeExternalSystem(&integrator, h, &A, &tau);

//   // Reference linearization via autodiff
//   const Context<double>& ctrl_context =
//       ctrl->GetMyContextFromRoot(simulator.get_context());
//   auto true_linearization = FirstOrderTaylorApproximation(
//       *ctrl, ctrl_context, ctrl->get_input_port_estimated_state().get_index(),
//       ctrl->get_output_port().get_index());

//   const MatrixXd B = plant.MakeActuationMatrix();
//   const MatrixXd& D = true_linearization->D();
//   const MatrixXd K = D.rightCols(2) + h * D.leftCols(2);  // N(q) = I
//   const VectorXd u0 = plant.get_actuation_input_port().Eval(plant_context) -
//                       D.rightCols(2) * plant.GetVelocities(plant_context);

//   const MatrixXd A_ref = -B * K;
//   const VectorXd tau_ref = B * u0;

//   // Confirm that our finite difference linearization is close to the reference
//   const double kTolerance = std::sqrt(std::numeric_limits<double>::epsilon());

//   EXPECT_TRUE(
//       CompareMatrices(A, A_ref, kTolerance, MatrixCompareType::relative));
//   EXPECT_TRUE(
//       CompareMatrices(tau, tau_ref, kTolerance, MatrixCompareType::relative));

//   // Compute the gradient of the cost, and check that this matches the momentum
//   // balance conditions, M(v − v*) + h A v − h τ₀ = 0.
//   const VectorXd v = v0;
//   MatrixXd M(nv, nv);
//   plant.CalcMassMatrix(plant_context, &M);
//   MultibodyForces<double> f_ext(plant);
//   plant.CalcForceElementsContribution(plant_context, &f_ext);
//   const VectorXd k =
//       plant.CalcInverseDynamics(plant_context, VectorXd::Zero(2), f_ext);
//   const VectorXd v_star = v0 - h * M.ldlt().solve(k);
//   const VectorXd dl_ref = M * (v - v_star) + h * A * v - h * tau;

//   SapContactProblem<double> problem =
//       ConvexIntegratorTester::MakeSapContactProblem(&integrator, plant_context,
//                                                     h);
//   SapModel<double> model(&problem);
//   auto model_context = model.MakeContext();
//   Eigen::VectorBlock<VectorXd> v_model =
//       model.GetMutableVelocities(model_context.get());
//   model.velocities_permutation().Apply(v, &v_model);
//   const VectorXd dl = model.EvalCostGradient(*model_context);

//   EXPECT_TRUE(
//       CompareMatrices(dl, dl_ref, kTolerance, MatrixCompareType::relative));

//   // Simulate for a few seconds
//   const int fps = 32;
//   meshcat->StartRecording(fps);
//   simulator.AdvanceTo(10.0);
//   meshcat->StopRecording();
//   meshcat->PublishRecording();

//   std::cout << std::endl;
//   PrintSimulatorStatistics(simulator);
//   std::cout << std::endl;
// }

// Test implicit joint effort limits
GTEST_TEST(ConvexIntegratorTest, EffortLimits) {
  // Set up the a system model with effort limits
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);
  Parser(&plant, &scene_graph)
      .AddModelsFromString(actuated_pendulum_xml, "xml");
  plant.Finalize();

  // Connect a high-gain PD controller
  VectorXd Kp(2), Kd(2), Ki(2);
  Kp << 1e3, 1e3;
  Kd << 0.1, 0.1;
  Ki << 0.0, 0.0;
  auto ctrl = builder.AddSystem<PidController>(Kp, Ki, Kd);

  VectorXd x_nom(4);
  x_nom << M_PI_2, M_PI_2, 0.0, 0.0;
  auto target_state = builder.AddSystem<ConstantVectorSource<double>>(x_nom);

  builder.Connect(target_state->get_output_port(),
                  ctrl->get_input_port_desired_state());
  builder.Connect(plant.get_state_output_port(),
                  ctrl->get_input_port_estimated_state());
  builder.Connect(ctrl->get_output_port(), plant.get_actuation_input_port());

  // Compile the system diagram
  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Get the actuator effort limits
  VectorXd effort_limits(2);
  for (JointActuatorIndex a : plant.GetJointActuatorIndices()) {
    const JointActuator<double>& actuator = plant.get_joint_actuator(a);
    const int i = actuator.input_start();
    const int n = actuator.num_inputs();
    effort_limits.segment(i, n) =
        VectorXd::Constant(n, actuator.effort_limit());
  }

  // Set up the integrator
  ConvexIntegrator<double> integrator(*diagram, diagram_context.get());
  integrator.set_maximum_step_size(0.1);
  integrator.set_fixed_step_mode(true);
  integrator.Initialize();

  const VectorXd q0 = plant.GetPositions(plant_context);
  const VectorXd v0 = plant.GetVelocities(plant_context);

  // Compute some dynamics terms
  MatrixXd M(2, 2);
  VectorXd k(2);
  MultibodyForces<double> f_ext(plant);
  plant.CalcMassMatrix(plant_context, &M);
  plant.CalcForceElementsContribution(plant_context, &f_ext);
  k = plant.CalcInverseDynamics(plant_context, VectorXd::Zero(2), f_ext);

  // Simulate for a step
  const double h = 0.01;
  EXPECT_TRUE(integrator.IntegrateWithSingleFixedStepToTime(h));

  // Compare requested and applied actuator forces
  const VectorXd u_req = plant.get_actuation_input_port().Eval(plant_context);
  fmt::print("u_req = {}\n", fmt_eigen(u_req.transpose()));

  // TODO(vincekurtz): report this in plant.get_net_actuation_output_port()
  const VectorXd v = plant.GetVelocities(plant_context);
  const VectorXd u_app = M * (v - v0) / h + k;
  fmt::print("u_app = {}\n", fmt_eigen(u_app.transpose()));
}

}  // namespace systems
}  // namespace drake
