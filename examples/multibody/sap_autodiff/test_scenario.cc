#include "drake/examples/multibody/sap_autodiff/test_scenario.h"

#include <iostream>
#include <chrono>

namespace drake {

using multibody::AddMultibodyPlant;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
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

void SapAutodiffTestScenario::RunTests(
    const SapAutodiffTestParameters& params) {
  // Update the stored parameters
  params_ = params;

  // Set the initial condition
  VectorX<double> x0;
  if (params.constraint) {
    x0 = get_x0_constrained();
  } else {
    x0 = get_x0_unconstrained();
  }

  // Do whatever we asked for
  if (params.simulate) {
    SimulateWithVisualizer(x0);
  }

  if (params.test_autodiff) {
    if (params.algebra == kAlgebraType::Both) {
      // Simulate several steps with both sparse (fancy) and dense (baseline)
      // methods and compare the results

      auto [st_dense, x_dense, dx_dense] =
          TakeAutodiffSteps(x0, params.num_steps, true);
      auto [st_sparse, x_sparse, dx_sparse] =
          TakeAutodiffSteps(x0, params.num_steps, false);

      const VectorX<double> val_diff = x_dense - x_sparse;
      const MatrixX<double> grad_diff = dx_dense - dx_sparse;

      std::cout << "Baseline (dense algebra) time: " << st_dense << std::endl;
      std::cout << "SAP (sparse algebra) time: " << st_sparse << std::endl;
      std::cout << "Value error: " << val_diff.norm() << std::endl;
      std::cout << "Gradient error: " << grad_diff.norm() << std::endl;
    
    } else {
      // Simulate several steps, then print the final state and gradients
      auto [runtime, x, dx] = TakeAutodiffSteps(
          x0, params.num_steps, (params.algebra == kAlgebraType::Dense));

      std::cout << "runtime: " << runtime << std::endl;
      std::cout << "x: \n" << x << std::endl;
      std::cout << "dx/dx0: \n" << dx << std::endl;
    }
  }
}

void SapAutodiffTestScenario::SimulateWithVisualizer(
    const VectorX<double>& x0) const {
  // Set up a system diagram
  MultibodyPlantConfig config;
  config.time_step = params_.time_step;
  config.contact_model = "hydroelastic";
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  // Create the plant model (user defined)
  CreateDoublePlant(&plant);

  // Use the SAP solver and set its parameters
  auto manager = std::make_unique<CompliantContactManager<double>>();
  SapSolverParameters sap_params;
  sap_params.use_dense_algebra = false;
  sap_params.line_search_type =
      SapSolverParameters::LineSearchType::kBackTracking;
  manager->set_sap_solver_parameters(sap_params);
  plant.SetDiscreteUpdateManager(std::move(manager));

  // Connect to the Drake visualizer
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  // Build the diagram
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, context.get());

  // Set initial conditions and control inputs
  plant.SetPositionsAndVelocities(&plant_context, x0);
  if (plant.num_actuators() > 0) {
    VectorX<double> u = VectorX<double>::Zero(plant.num_actuators());
    plant.get_actuation_input_port().FixValue(&plant_context, u);
  }

  // Set up and run the simulator
  systems::Simulator<double> simulator(*diagram, std::move(context));
  simulator.set_target_realtime_rate(params_.realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(params_.simulation_time);
}

std::tuple<double, VectorX<double>, MatrixX<double>>
SapAutodiffTestScenario::TakeAutodiffSteps(const VectorX<double>& x0,
                                           const int num_steps,
                                           const bool dense_algebra) const {
  // Set up a system diagram
  MultibodyPlantConfig config;
  config.time_step = params_.time_step;
  config.contact_model = "hydroelastic";
  DiagramBuilder<double> builder;
  auto [plant_double, scene_graph_double] = AddMultibodyPlant(config, &builder);

  // Create the plant model (user defined)
  CreateDoublePlant(&plant_double);

  // Use the SAP solver and set its parameters
  auto manager = std::make_unique<CompliantContactManager<double>>();
  SapSolverParameters sap_params;
  sap_params.use_dense_algebra = dense_algebra;
  sap_params.line_search_type =
      SapSolverParameters::LineSearchType::kBackTracking;
  manager->set_sap_solver_parameters(sap_params);
  plant_double.SetDiscreteUpdateManager(std::move(manager));

  // Build the system diagram and convert to autodiff
  auto diagram_double = builder.Build();
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

}  // namespace sap_autodiff
}  // namespace multibody
}  // namespace examples
}  // namespace drake