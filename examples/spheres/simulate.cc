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
DEFINE_double(time_step, 0.005, "Simulator time step (dt) in seconds.");
DEFINE_bool(visualize, false, "Whether to visualize (true) or not (false).");

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
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  Parser(&plant).AddModelsFromUrl(
      "package://drake/examples/spheres/spheres_in_a_box.xml");
  plant.Finalize();

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
