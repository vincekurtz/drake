#include <chrono>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {

using multibody::AddMultibodyPlant;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using multibody::contact_solvers::internal::SapSolverParameters;
using multibody::internal::CompliantContactManager;
using systems::Context;
using systems::DiscreteValues;
using systems::System;

namespace examples {
namespace multibody {
namespace acrobot {

// Simulate several steps and use autodiff to compute gradients with respect to
// the initial state. Return the resulting state and gradient matrix.
std::tuple<double, VectorX<double>, MatrixX<double>> take_autodiff_steps(
    MultibodyPlantConfig plant_config, int num_steps, bool dense_algebra) {
  // Create a MultibodyPlant acrobot model via SDF parsing
  systems::DiagramBuilder<double> builder;
  auto [plant_double, scene_graph_double] =
      AddMultibodyPlant(plant_config, &builder);
  const std::string file_name =
      FindResourceOrThrow("drake/multibody/benchmarks/acrobot/acrobot.sdf");
  Parser(&plant_double).AddModelFromFile(file_name);
  plant_double.Finalize();

  // Specify the SAP solver and parameters
  auto manager = std::make_unique<CompliantContactManager<double>>();
  SapSolverParameters sap_params;
  sap_params.use_dense_algebra = dense_algebra;
  manager->set_sap_solver_parameters(sap_params);
  plant_double.SetDiscreteUpdateManager(std::move(manager));

  // Convert to autodiff
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
  for (int i = 0; i < num_steps; ++i) {
    plant->CalcDiscreteVariableUpdates(*context, state.get());
    x = state->value();
    context->SetDiscreteState(x);
  }
  elapsed = std::chrono::high_resolution_clock::now() - st;

  // Return gradients and time elapsed in seconds
  return {elapsed.count(), math::ExtractValue(x), math::ExtractGradient(x)};
}

DEFINE_double(time_step, 1e-2, "Time step for discrete-time simulation");
DEFINE_string(
    algebra, "both",
    "Type of algebra to use. Options are: 'sparse', 'dense', or 'both'.");
DEFINE_int32(num_steps, 1, "Number of steps to simulate.");

int do_main() {
  int num_steps = FLAGS_num_steps;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;

  if (FLAGS_algebra == "sparse" || FLAGS_algebra == "dense") {
    // Simulate a step and print the next state and gradients

    auto [runtime, x, dx] = take_autodiff_steps(plant_config, num_steps,
                                                (FLAGS_algebra == "dense"));

    std::cout << "runtime: " << runtime << std::endl;
    std::cout << "x: \n" << x << std::endl;
    std::cout << "dx/dx0: \n" << dx << std::endl;

  } else if (FLAGS_algebra == "both") {
    // Take a step with both sparse and dense algebra and compare the results

    auto [st_dense, x_dense, dx_dense] =
        take_autodiff_steps(plant_config, num_steps, true);
    auto [st_sparse, x_sparse, dx_sparse] =
        take_autodiff_steps(plant_config, num_steps, false);

    const VectorX<double> val_diff = x_dense - x_sparse;
    const MatrixX<double> grad_diff = dx_dense - dx_sparse;

    std::cout << "Baseline (dense algebra) time: " << st_dense << std::endl;
    std::cout << "SAP (sparse algebra) time: " << st_sparse << std::endl;
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
