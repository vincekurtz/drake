#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/multibody/tree/prismatic_joint.h"

/**
 * A super simple example system which contains one degree of freedom and one
 * constraint.
 *
 * TODO(vincekurtz): write a unified example class to combine this and the two
 * acrobots one box example. And make it easy to add others.
 */

namespace drake {

using multibody::AddMultibodyPlant;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::PrismaticJoint;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;
using multibody::contact_solvers::internal::SapSolverParameters;
using multibody::internal::CompliantContactManager;
using systems::Context;
using systems::DiagramBuilder;
using systems::DiscreteValues;
using systems::System;

namespace examples {
namespace multibody {
namespace sap_autodiff {
namespace {

DEFINE_bool(test_autodiff, true,
            "Whether to run some autodiff tests. If false, runs a quick "
            "simulation of the scenario instead.");
DEFINE_string(algebra, "both",
              "Type of algebra to use for testing autodiff. Options are: "
              "'sparse', 'dense', or 'both'.");
DEFINE_int32(num_steps, 1,
             "Number of timesteps to simulate for testing autodiff.");
DEFINE_bool(constrained, true,
            "Whether the initial state is such that sphere is in its lowest "
            "possible position (constraint active) or not.");

void CreateDoublePlant(MultibodyPlant<double>* plant, const bool dense_algebra) {
  // Some parameters
  const double radius = 0.1;
  const double mass = 0.8;
  const double lower_limit = 0.1;

  // Create the plant
  UnitInertia<double> G_Bo = UnitInertia<double>::SolidCube(2*radius);
  SpatialInertia<double> M_Bo(mass, Vector3<double>::Zero(), G_Bo);
  const RigidBody<double>& body = plant->AddRigidBody("body", M_Bo);
  const math::RigidTransform<double> X;
  plant->RegisterVisualGeometry(body, X, geometry::Box(2*radius, 2*radius, 2*radius), "body");
  plant->RegisterVisualGeometry(plant->world_body(), X, geometry::Cylinder(0.01,10), "vertical_rod");
  plant->AddJoint<PrismaticJoint>(
      "joint", plant->world_body(), std::nullopt, body, std::nullopt,
      Vector3<double>::UnitZ(), lower_limit);
  plant->Finalize();

  // Specify the SAP solver and parameters
  auto manager = std::make_unique<CompliantContactManager<double>>();
  SapSolverParameters sap_params;
  sap_params.use_dense_algebra = dense_algebra;
  sap_params.line_search_type = SapSolverParameters::LineSearchType::kBackTracking;
  manager->set_sap_solver_parameters(sap_params);
  plant->SetDiscreteUpdateManager(std::move(manager));
}

void SimulateWithVisualizer(const VectorX<double>& x0) {
  // Set up a system diagram, complete with plant and scene graph
  systems::DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = 1e-2;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  // Create the plant model
  CreateDoublePlant(&plant, false);

  // Connect to Drake visualizer and compile the diagram
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, context.get());

  // Set initial conditions
  plant.SetPositionsAndVelocities(&plant_context, x0);

  // Set up and run the simulator
  systems::Simulator<double> simulator(*diagram, std::move(context));
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(1.0);
}

/**
 * Simulate several steps and use autodiff to compute gradients with respect to
 * the initial state. Return the final state and gradient matrix along with the
 * computation time.
 *
 * @param x0            Initial state of the system
 * @param num_steps     Number of timesteps to simulate
 * @param dense_algebra Whether to use dense algebra for sap
 * @return std::tuple<double, VectorX<double>, MatrixX<double>> tuple of runtime
 * in seconds, x, dx_dx0.
 */
std::tuple<double, VectorX<double>, MatrixX<double>> TakeAutodiffSteps(
    const VectorX<double>& x0, const int num_steps, const bool dense_algebra) {
  // Create a double plant and scene graph
  MultibodyPlantConfig config;
  config.time_step = 1e-2;
  config.contact_model = "hydroelastic";
  DiagramBuilder<double> builder;
  auto [plant_double, scene_graph_double] = AddMultibodyPlant(config, &builder);
  CreateDoublePlant(&plant_double, dense_algebra);
  auto diagram_double = builder.Build();

  // Convert to autodiff
  auto diagram = systems::System<double>::ToAutoDiffXd(*diagram_double);
  auto diagram_context = diagram->CreateDefaultContext();
  const MultibodyPlant<AutoDiffXd>* plant =
      dynamic_cast<const MultibodyPlant<AutoDiffXd>*>(
          &diagram->GetSubsystemByName("plant"));
  systems::Context<AutoDiffXd>& plant_context =
      plant->GetMyMutableContextFromRoot(diagram_context.get());

  // Set initial conditions
  const VectorX<AutoDiffXd> x0_ad = math::InitializeAutoDiff(x0);
  plant->SetPositionsAndVelocities(&plant_context, x0_ad);

  // Step forward in time
  std::unique_ptr<DiscreteValues<AutoDiffXd>> state =
      plant->AllocateDiscreteVariables();

  std::chrono::duration<double> elapsed;
  auto st = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_steps; ++i) {
    plant->CalcDiscreteVariableUpdates(plant_context, state.get());
    plant_context.SetDiscreteState(state->value());
  }

  elapsed = std::chrono::high_resolution_clock::now() - st;
  VectorX<AutoDiffXd> x = state->value();

  return {elapsed.count(), math::ExtractValue(x), math::ExtractGradient(x)};
}

int do_main() {
  // Define the initial state
  VectorX<double> x0(2);
  x0 << 0.05, 0;

  if (!FLAGS_constrained) {
    // Move the sphere up away from the joint limits
    x0(0) += 2;
  }

  if (FLAGS_test_autodiff) {
    if (FLAGS_algebra == "sparse" || FLAGS_algebra == "dense") {
      // Simulate several steps, then print the final state and gradients
      auto [runtime, x, dx] =
          TakeAutodiffSteps(x0, FLAGS_num_steps, (FLAGS_algebra == "dense"));

      std::cout << "runtime: " << runtime << std::endl;
      std::cout << "x: \n" << x << std::endl;
      std::cout << "dx/dx0: \n" << dx << std::endl;

    } else if (FLAGS_algebra == "both") {
      // Simulate several steps with both sparse (fancy) and dense (baseline)
      // methods and compare the results

      auto [st_dense, x_dense, dx_dense] =
          TakeAutodiffSteps(x0, FLAGS_num_steps, true);
      auto [st_sparse, x_sparse, dx_sparse] =
          TakeAutodiffSteps(x0, FLAGS_num_steps, false);

      const VectorX<double> val_diff = x_dense - x_sparse;
      const MatrixX<double> grad_diff = dx_dense - dx_sparse;

      std::cout << "Baseline (dense algebra) time: " << st_dense << std::endl;
      std::cout << "SAP (sparse algebra) time: " << st_sparse << std::endl;
      std::cout << "Value error: " << val_diff.norm() << std::endl;
      std::cout << "Gradient error: " << grad_diff.norm() << std::endl;
    }
  } else {
    // Run a full simulation
    SimulateWithVisualizer(x0);
  }

  return 0;
}

}  // namespace
}  // namespace sap_autodiff
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::sap_autodiff::do_main();
}
