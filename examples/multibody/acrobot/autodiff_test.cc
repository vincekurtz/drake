#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {

using multibody::AddMultibodyPlantSceneGraph;
using multibody::Parser;
using systems::Context;
using systems::DiscreteValues;

namespace examples {
namespace multibody {
namespace acrobot {

DEFINE_double(time_step, 1e-2, "Time step for discrete-time simulation");

int do_main() {
  const double dt = FLAGS_time_step;

  // Create a MultibodyPlant acrobot model via SDF parsing
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, dt);
  const std::string file_name =
      FindResourceOrThrow("drake/multibody/benchmarks/acrobot/acrobot.sdf");
  Parser parser(&plant, &scene_graph);
  parser.AddModelFromFile(file_name);
  plant.Finalize();

  // Set initial conditions and input
  std::unique_ptr<Context<double>> context = plant.CreateDefaultContext();
  VectorX<double> x0(4);
  x0 << 0.9, 1.1, 0.1, -0.2;
  plant.SetPositionsAndVelocities(context.get(), x0);
  
  const double u = 0;
  plant.get_actuation_input_port().FixValue(context.get(), u);

  // Simulate forward one timestep
  std::unique_ptr<DiscreteValues<double>> state =
      plant.AllocateDiscreteVariables();
  plant.CalcDiscreteVariableUpdates(*context, state.get());
  VectorX<double> x = state->value();

  std::cout << x << std::endl;

  // Get the gradients

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
