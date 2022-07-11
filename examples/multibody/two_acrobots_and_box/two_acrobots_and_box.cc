#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {

using multibody::AddMultibodyPlant;
using multibody::MultibodyPlant;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using multibody::contact_solvers::internal::SapSolverParameters;
using multibody::internal::CompliantContactManager;
using systems::Context;
using systems::DiscreteValues;
using systems::System;
using systems::DiagramBuilder;

namespace examples {
namespace multibody {
namespace two_acrobots_and_box {

void create_double_plant(MultibodyPlant<double>* plant,
                         const bool& dense_algebra) {
  // Load the models of acrobots and box from an sdf file 
  const std::string acrobot_file = FindResourceOrThrow(
      "drake/examples/multibody/two_acrobots_and_box/two_acrobots_and_box.sdf");
  Parser(plant).AddAllModelsFromFile(acrobot_file);
  plant->Finalize();

  // Specify the SAP solver and parameters
  auto manager = std::make_unique<CompliantContactManager<double>>();
  SapSolverParameters sap_params;
  sap_params.use_dense_algebra = dense_algebra;
  manager->set_sap_solver_parameters(sap_params);
  plant->SetDiscreteUpdateManager(std::move(manager));
}

/**
 * Run a quick simulation of the system with zero input, connected to the Drake
 * visualizer. This is just to get a general sense of what is going on in the
 * example.
 *
 * @param x0            The initial state of the system
 * @param end_time      The time (in seconds) to simulate for.
 */
void simulate_with_visualizer(const VectorX<double>& x0,
                              const double& end_time) {
  // Set up the system diagram and create the plant model
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = 1e-3;
  config.contact_model = "hydroelastic";
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  create_double_plant(&plant, false);

  // Connect to Drake visualizer
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  // Compile the system diagram
  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set initial conditions and input
  VectorX<double> u = VectorX<double>::Zero(plant.num_actuators());
  plant.get_actuation_input_port().FixValue(&plant_context, u);
  plant.SetPositionsAndVelocities(&plant_context, x0);

  // Set up and run the simulator
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(end_time);
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
std::tuple<double, VectorX<double>, MatrixX<double>> take_autodiff_steps(
    const VectorX<double>& x0, const int& num_steps,
    const bool& dense_algebra) {
  // Create a double plant and scene graph
  MultibodyPlantConfig config;
  config.time_step = 1e-3;
  config.contact_model = "hydroelastic";
  DiagramBuilder<double> builder;
  auto [plant_double, scene_graph_double] = AddMultibodyPlant(config, &builder);
  create_double_plant(&plant_double, dense_algebra);
  auto diagram_double = builder.Build();

  // Convert to autodiff
  auto diagram = systems::System<double>::ToAutoDiffXd(*diagram_double);
  auto diagram_context = diagram->CreateDefaultContext();
  const MultibodyPlant<AutoDiffXd>* plant=
      dynamic_cast<const MultibodyPlant<AutoDiffXd>*>(
          &diagram->GetSubsystemByName("plant"));
  systems::Context<AutoDiffXd>& plant_context =
        plant->GetMyMutableContextFromRoot(diagram_context.get());

  // Set initial conditions and input
  VectorX<AutoDiffXd> u = VectorX<AutoDiffXd>::Zero(plant->num_actuators());
  plant->get_actuation_input_port().FixValue(&plant_context, u);

  const VectorX<AutoDiffXd> x0_ad = math::InitializeAutoDiff(x0);
  plant->SetPositionsAndVelocities(&plant_context, x0_ad);

  // Step forward in time
  std::unique_ptr<DiscreteValues<AutoDiffXd>> state =
      plant->AllocateDiscreteVariables();

  std::chrono::duration<double> elapsed;
  auto st = std::chrono::high_resolution_clock::now();

  for (int i=0; i<num_steps; ++i) {
    plant->CalcDiscreteVariableUpdates(plant_context, state.get());
    plant_context.SetDiscreteState(state->value());
  }

  elapsed = std::chrono::high_resolution_clock::now() - st;
  VectorX<AutoDiffXd> x = state->value();

  return {elapsed.count(), math::ExtractValue(x), math::ExtractGradient(x)};
}

int do_main() {
  // Set the initial state
  VectorX<double> x0(4+4+13);
  x0 << 0.9, 1.1,           // first acrobot position
        0.9, 1.1,           // second acrobot position
        0.985, 0, 0.174, 0, // box orientation
        -1.5, 0.25, 2,      // box position
        0.5, 0.5,           // first acrobot velocity
        0.5, 0.5,           // second acrobot velocity
        0.1, -0.2, 0.1,     // box angular velocity
        0.1, 0.1, 0.2;      // box linear velocity

  // Run a full simulation
  //const double end_time = 2;
  //simulate_with_visualizer(x0, end_time);

  // Take several timesteps with autodiff
  const int num_steps = 1;
  auto [runtime, x, dx] = take_autodiff_steps(x0, num_steps, true);

  return 0;
}

}  // namespace two_acrobots_and_box
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::multibody::two_acrobots_and_box::do_main();
}
