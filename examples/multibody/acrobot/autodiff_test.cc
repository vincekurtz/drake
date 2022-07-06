#include <iostream>
#include <memory>

#include <chrono>
#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"

namespace drake {

using multibody::AddMultibodyPlant;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::Context;
using systems::DiscreteValues;
using systems::System;

namespace examples {
namespace multibody {
namespace acrobot {

// Simulate several steps and use autodiff to compute gradients with respect to the initial state. 
// Return the resulting state and gradient matrix.
std::tuple<double, VectorX<double>, MatrixX<double>> take_autodiff_steps(
    MultibodyPlantConfig plant_config, int num_steps) {
  // Create a MultibodyPlant acrobot model via SDF parsing
  systems::DiagramBuilder<double> builder;
  auto [plant_double, scene_graph_double] =
      AddMultibodyPlant(plant_config, &builder);
  const std::string file_name =
      FindResourceOrThrow("drake/multibody/benchmarks/acrobot/acrobot.sdf");
  Parser(&plant_double).AddModelFromFile(file_name);
  plant_double.Finalize();

  // Convert to autodiff after parsing
  std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant =
      System<double>::ToAutoDiffXd(plant_double);
  std::unique_ptr<Context<AutoDiffXd>> context = plant->CreateDefaultContext();

  // Set initial conditions and input
  VectorX<double> x0_val(4);
  x0_val << 0.9, 1.1, 0.1, -0.2;
  const VectorX<AutoDiffXd> x0 = math::InitializeAutoDiff(x0_val);
  plant->SetPositionsAndVelocities(context.get(), x0);

  const AutoDiffXd u = 0;
  plant->get_actuation_input_port().FixValue(context.get(), u);

  // Simulate forward for num_steps
  std::unique_ptr<DiscreteValues<AutoDiffXd>> state =
      plant->AllocateDiscreteVariables();
  VectorX<AutoDiffXd> x;

  std::chrono::duration<double> elapsed;
  auto st = std::chrono::high_resolution_clock::now();
  for (int i=0; i<num_steps; ++i)  {
    plant->CalcDiscreteVariableUpdates(*context, state.get());
    x = state->value();
    context->SetDiscreteState(x);
  }
  elapsed = std::chrono::high_resolution_clock::now() - st;

  // Return gradients and time elapsed in seconds
  return {elapsed.count(), math::ExtractValue(x), math::ExtractGradient(x)};
}

DEFINE_double(time_step, 1e-2, "Time step for discrete-time simulation");
DEFINE_string(contact_solver, "both",
              "Contact solver. Options are: 'tamsi', 'sap', or 'both'.");
DEFINE_int32(num_steps, 1, "Number of steps to simulate.");

int do_main() {
  int num_steps = FLAGS_num_steps;

  if ( FLAGS_contact_solver == "sap" or FLAGS_contact_solver == "tamsi" ) {
    // Simulate a step and print the next state and gradients

    MultibodyPlantConfig plant_config;
    plant_config.time_step = FLAGS_time_step;
    plant_config.contact_solver = FLAGS_contact_solver;

    // Simulate a step
    auto [runtime, x, dx] = take_autodiff_steps(plant_config, num_steps);

    // Get the gradients
    std::cout << "runtime: " << runtime << std::endl;
    std::cout << "x: \n" << x << std::endl;
    std::cout << "dx/dx0: \n" << dx << std::endl;
  } else if ( FLAGS_contact_solver == "both" ) {
    // Take a step with both contact solvers and compare the result

    MultibodyPlantConfig plant_config;
    plant_config.time_step = FLAGS_time_step;
    plant_config.contact_solver = "tamsi";
    auto [st_tamsi, x_tamsi, dx_tamsi] = take_autodiff_steps(plant_config, num_steps);

    plant_config.contact_solver = "sap";
    auto [st_sap, x_sap, dx_sap] = take_autodiff_steps(plant_config, num_steps);

    const VectorX<double> val_diff = x_tamsi - x_sap;
    const MatrixX<double> grad_diff = dx_tamsi - dx_sap;

    std::cout << "TAMSI time: " << st_tamsi << std::endl;
    std::cout << "SAP time: " << st_sap << std::endl;
    std::cout << "Value error: " << val_diff.norm() << std::endl;
    std::cout << "Gradient error: " << grad_diff.norm() << std::endl;

  }

  return 0;
}

}  // namespace acrobot
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::acrobot::do_main();
}
