#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include <gflags/gflags.h>

#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/visualization/visualization_config_functions.h"

namespace drake {
namespace examples {
namespace {

DEFINE_double(simulation_time, 10.0, "Simulation duration in seconds.");
DEFINE_double(time_step, 0.001, "Simulator time step (dt) in seconds.");
DEFINE_bool(visualize, false, "Whether to visualize (true) or not (false).");
DEFINE_bool(meshes, false,
            "Whether to an example with complicated meshes (true) or just a "
            "bunch of spheres (false).");
DEFINE_double(near_rigid_threshold, 1.0, "Threshold for near-rigid contact.");
DEFINE_double(stiction_tolerance, 1.0e-4, "Stiction tolerance [m/s].");
DEFINE_double(penetration_allowance, 1.0e-3,
              "Penetration allowance for contact.");
DEFINE_double(point_stiffness, 1.0e6,
              "Stiffness for point contact [N/m].");

using geometry::SceneGraphConfig;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::Context;
using systems::DiagramBuilder;
using systems::Simulator;
using visualization::AddDefaultVisualization;
using clock = std::chrono::steady_clock;

int do_main() {
  // Set up the system diagram
  DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.sap_near_rigid_threshold = FLAGS_near_rigid_threshold;
  plant_config.stiction_tolerance = FLAGS_stiction_tolerance;
  plant_config.penetration_allowance = FLAGS_penetration_allowance; 
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  std::string url = "package://drake/examples/clutter/spheres_in_a_box.xml";
  if (FLAGS_meshes) {
    url = "package://drake/examples/clutter/meshes_in_a_box.xml";
  }
  Parser(&plant).AddModelsFromUrl(url);
  plant.Finalize();

  SceneGraphConfig sg_config;
  sg_config.default_proximity_properties.point_stiffness = FLAGS_point_stiffness;
  scene_graph.set_config(sg_config);

  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  if (FLAGS_visualize) {
    AddDefaultVisualization(&builder, meshcat);
  }

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  // Set up the simulator with the specified integration scheme.
  auto simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));
  simulator->Initialize();
  if (FLAGS_visualize) {
    std::cout << "Press any key to continue ...\n";
    getchar();
  }

  const double recording_frames_per_second =
      FLAGS_time_step == 0 ? 32 : 1.0 / FLAGS_time_step;
  meshcat->StartRecording(recording_frames_per_second);
  clock::time_point sim_start_time = clock::now();
  simulator->AdvanceTo(FLAGS_simulation_time);
  clock::time_point sim_end_time = clock::now();
  const double sim_time =
      std::chrono::duration<double>(sim_end_time - sim_start_time).count();
  std::cout << "AdvanceTo() time [sec]: " << sim_time << std::endl;
  meshcat->StopRecording();
  meshcat->PublishRecording();

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "\nSimulation of a bunch of spheres falling into a box.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
