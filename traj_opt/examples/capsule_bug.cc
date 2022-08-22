#include <iostream>

#include "drake/common/find_resource.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace capsule_geometry_bug {

using Eigen::VectorXd;
using geometry::SignedDistancePair;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::Context;
using systems::DiagramBuilder;

void compute_change_in_signed_distance(const std::string urdf_file,
                                       const double epsilon) {
  // Create a plant from the given urdf
  MultibodyPlantConfig config;
  config.time_step = 5e-2;
  DiagramBuilder<double> builder;

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  Parser(&plant).AddAllModelsFromFile(FindResourceOrThrow(urdf_file));
  plant.Finalize();

  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set positions to a point we know is a problem
  VectorXd q(2);
  q << 1.08, -0.12;
  plant.SetPositions(&plant_context, q);

  // Compute signed distances at this point
  const geometry::QueryObject<double>& query_object =
      plant.get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<double>>(plant_context);
  const std::vector<SignedDistancePair<double>>& signed_distance_pairs =
      query_object.ComputeSignedDistancePairwiseClosestPoints();
  const double phi = signed_distance_pairs[0].distance;

  // Perturb positions by epsilon and compute signed distances again
  q(0) += epsilon;
  plant.SetPositions(&plant_context, q);

  const geometry::QueryObject<double>& query_object_eps =
      plant.get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<double>>(plant_context);
  const std::vector<SignedDistancePair<double>>& signed_distance_pairs_eps =
      query_object_eps.ComputeSignedDistancePairwiseClosestPoints();
  const double phi_eps = signed_distance_pairs_eps[0].distance;

  // Approximate ∂ϕ/∂q₁, the derivative of signed distance w.r.t q1:
  const double dphi = (phi_eps - phi) / epsilon;

  std::cout << fmt::format("ϕ(q)   : {}", phi) << std::endl;
  std::cout << fmt::format("ϕ(q+ε) : {}", phi_eps) << std::endl;
  std::cout << fmt::format("∂ϕ/∂q₁ : {}", dphi) << std::endl;
}

int do_main() {
  const double eps = sqrt(std::numeric_limits<double>::epsilon());

  std::cout << "Sphere-Capulse Collisions: " << std::endl;
  compute_change_in_signed_distance(
      "drake/traj_opt/examples/2dof_spinner.urdf", eps);
  std::cout << std::endl;
  
  std::cout << "Capsule-Capsule Collisions: " << std::endl;
  compute_change_in_signed_distance(
      "drake/traj_opt/examples/2dof_spinner_capsule.urdf", eps);

  return 0;
}

}  // namespace capsule_geometry_bug
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  return drake::traj_opt::examples::capsule_geometry_bug::do_main();
}