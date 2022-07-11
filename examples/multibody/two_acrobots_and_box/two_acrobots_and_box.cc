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
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using multibody::contact_solvers::internal::SapSolverParameters;
using multibody::internal::CompliantContactManager;
using systems::Context;
using systems::DiscreteValues;
using systems::System;

namespace examples {
namespace multibody {
namespace two_acrobots_and_box {

/**
 * Run a quick simulation of the system with zero input, connected to the Drake
 * visualizer. This is just to get a general sense of what is going on in the
 * example.
 *
 * @param x0        The initial state of the system.
 * @param end_time  The time (in seconds) to simulate for.
 */

void simulate_with_visualizer(const VectorX<double>& x0,
                              const double& end_time) {
  systems::DiagramBuilder<double> builder;

  // Create the multibody plant
  MultibodyPlantConfig config;
  config.time_step = 1e-3;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  const std::string file_name =
      FindResourceOrThrow("drake/multibody/benchmarks/acrobot/acrobot.sdf");
  Parser(&plant).AddModelFromFile(file_name);
  plant.Finalize();

  // Specify the SAP solver and parameters
  auto manager = std::make_unique<CompliantContactManager<double>>();
  SapSolverParameters sap_params;
  manager->set_sap_solver_parameters(sap_params);
  plant.SetDiscreteUpdateManager(std::move(manager));

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
  plant.SetPositionsAndVelocities(&plant_context, x0);
  const double u = 0;
  plant.get_actuation_input_port().FixValue(&plant_context, u);

  // Set up and run the simulator
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(end_time);
}

int do_main() {
  std::cout << "hello world" << std::endl;
  double end_time = 2.0;
  VectorX<double> x0(4);
  x0 << 0.9, 1.1, 0.1, -0.2;

  simulate_with_visualizer(x0, end_time);

  return 0;
}

}  // namespace two_acrobots_and_box
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::multibody::two_acrobots_and_box::do_main();
}
