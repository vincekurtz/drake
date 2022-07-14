#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/multibody/tree/prismatic_joint.h"

/**
 * A super simple example system which contains one degree of freedom and one
 * constraint.
 */

namespace drake {

using multibody::AddMultibodyPlant;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::PrismaticJoint;
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
namespace {

int do_main() {

  // Some parameters
  const double time_step = 1e-3;
  const double radius = 0.1;
  const double mass = 0.8;
  const double lower_limit = 0.1;

  // Set up a system diagram, complete with plant and scene graph
  systems::DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = time_step;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  // Create the plant model
  UnitInertia<double> G_Bo = UnitInertia<double>::SolidSphere(radius);
  SpatialInertia<double> M_Bo(mass, Vector3<double>::Zero(), G_Bo);
  const RigidBody<double>& sphere_body = plant.AddRigidBody("sphere_body", M_Bo);
  const math::RigidTransform<double> X;
  plant.RegisterVisualGeometry(sphere_body, X, geometry::Sphere(radius), "sphere_body");
  plant.AddJoint<PrismaticJoint>(
      "joint", plant.world_body(), std::nullopt, sphere_body, std::nullopt,
      Vector3<double>::UnitZ(), lower_limit);
  plant.Finalize();

  // Connect to Drake visualizer and compile the diagram
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, context.get());

  // Set initial conditions
  VectorX<double> x0(2);
  x0 << 0.2, 0;
  plant.SetPositionsAndVelocities(&plant_context, x0);

  // Set up and run the simulator
  systems::Simulator<double> simulator(*diagram, std::move(context));
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(1.0);

  return 0;
}

}  // namespace
}  // namespace sap_autodiff
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::sap_autodiff::do_main();
}
