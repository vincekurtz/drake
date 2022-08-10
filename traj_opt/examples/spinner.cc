#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace spinner {

// Command line options
DEFINE_double(time_step, 1e-2,
              "Discretization timestep for the optimizer (seconds).");
DEFINE_int32(num_steps, 200,
             "Number of timesteps in the optimization problem.");
DEFINE_double(q1_init, 0.2, "Initial angle for the first (finger) joint");
DEFINE_double(q2_init, 1.5, "Initial angle for the second (finger) joint");
DEFINE_double(q3_init, 0.0, "Initial angle for the third (spinner) joint");
DEFINE_double(v1_init, 1.0, "Initial velocity for the first (finger) joint");
DEFINE_double(v2_init, 0.0, "Initial velocity for the second (finger) joint");
DEFINE_double(v3_init, 0.0, "Initial velocity for the third (spinner) joint");

using geometry::DrakeVisualizerd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlant;
using multibody::ConnectContactResultsToDrakeVisualizer;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;
using systems::Simulator;

/**
 * Simulate the system with zero control input, connected to the visualizer.
 */
void run_passive_simulation() {
  const double dt = FLAGS_time_step;
  const int num_steps = FLAGS_num_steps;
  const double sim_time = dt * num_steps;

  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = dt;
  config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/spinner.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
  ConnectContactResultsToDrakeVisualizer(&builder, plant, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const VectorXd u = VectorXd::Zero(plant.num_actuators());
  plant.get_actuation_input_port().FixValue(&plant_context, u);

  VectorX<double> x0(6);
  x0 << FLAGS_q1_init, FLAGS_q2_init, FLAGS_q3_init, FLAGS_v1_init,
      FLAGS_v2_init, FLAGS_v3_init;

  plant.SetPositionsAndVelocities(&plant_context, x0);

  Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(sim_time);
}

int do_main() {
  run_passive_simulation();
  return 0;
}

}  // namespace spinner
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::traj_opt::examples::spinner::do_main();
}
