#include <iostream>
#include <memory>

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

// Simulate a single step and use autodiff to compute gradients with respect to the initial state. 
// Return the subsequent state and gradient matrix. 
std::tuple<VectorX<double>, MatrixX<double>> take_autodiff_step(MultibodyPlantConfig plant_config) {
  // Create a MultibodyPlant acrobot model via SDF parsing
  systems::DiagramBuilder<double> builder;
  auto [plant_double, scene_graph_double] = AddMultibodyPlant(plant_config, &builder);
  const std::string file_name =
      FindResourceOrThrow("drake/multibody/benchmarks/acrobot/acrobot.sdf");
  Parser(&plant_double).AddModelFromFile(file_name);
  plant_double.Finalize();

  // Convert to autodiff after parsing
  std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant = System<double>::ToAutoDiffXd(plant_double);
  std::unique_ptr<Context<AutoDiffXd>> context = plant->CreateDefaultContext();

  // Set initial conditions and input
  VectorX<double> x0_val(4);
  x0_val << 0.9, 1.1, 0.1, -0.2;
  const VectorX<AutoDiffXd> x0 = math::InitializeAutoDiff(x0_val);
  plant->SetPositionsAndVelocities(context.get(), x0);
  
  const AutoDiffXd u = 0;
  plant->get_actuation_input_port().FixValue(context.get(), u);

  // Simulate forward one timestep
  std::unique_ptr<DiscreteValues<AutoDiffXd>> state =
      plant->AllocateDiscreteVariables();
  plant->CalcDiscreteVariableUpdates(*context, state.get());
  VectorX<AutoDiffXd> x = state->value();

  // Get the gradients
  return { math::ExtractValue(x), math::ExtractGradient(x) };

}

DEFINE_double(time_step, 1e-2, "Time step for discrete-time simulation");
DEFINE_string(contact_solver, "tamsi",
              "Contact solver. Options are: 'tamsi', 'sap'.");

int do_main() {

  if ( FLAGS_contact_solver == "sap" or FLAGS_contact_solver == "tamsi" ) {
    // Simulate a step and print the next state and gradients

    MultibodyPlantConfig plant_config;
    plant_config.time_step = FLAGS_time_step;
    plant_config.contact_solver = FLAGS_contact_solver;

    // Simulate a step
    auto [x, dx] = take_autodiff_step(plant_config);

    // Get the gradients
    std::cout << x << std::endl;
    std::cout << dx << std::endl;
  } else if ( FLAGS_contact_solver == "both" ) {
    // Take a step with both contact solvers and compare the result

    MultibodyPlantConfig plant_config;
    plant_config.time_step = FLAGS_time_step;
    plant_config.contact_solver = "tamsi";
    auto [x_tamsi, dx_tamsi] = take_autodiff_step(plant_config);

    plant_config.contact_solver = "sap";
    auto [x_sap, dx_sap] = take_autodiff_step(plant_config);

    const VectorX<double> val_diff = x_tamsi - x_sap;
    const MatrixX<double> grad_diff = dx_tamsi - dx_sap;

    std::cout << val_diff << std::endl;
    std::cout << grad_diff << std::endl;

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
