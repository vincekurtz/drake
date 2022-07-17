#include <iostream>

#include "drake/examples/multibody/sap_autodiff/test_scenario.h"

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
        // Compare both methods
        std::cout << "comparing both methods" << std::endl;
    } else {
        // Just use one of the methods and print the result
        std::cout << "using just one method" << std::endl;
    }
  }
}

void SapAutodiffTestScenario::SimulateWithVisualizer(const VectorX<double>& x0) const {

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
  sap_params.line_search_type = SapSolverParameters::LineSearchType::kBackTracking;
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

}  // namespace sap_autodiff
}  // namespace multibody
}  // namespace examples
}  // namespace drake