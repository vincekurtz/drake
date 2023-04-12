#include "drake/common/find_resource.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace jaco {

using Eigen::Vector3d;
using geometry::AddContactMaterial;
using geometry::AddCompliantHydroelasticProperties;
using geometry::Box;
using geometry::ProximityProperties;
using math::RigidTransformd;
using math::RollPitchYaw;
using multibody::CoulombFriction;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::Parser;

class JacoExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    // Add a jaco arm without gravity
    std::string robot_file = FindResourceOrThrow(
        "drake/traj_opt/examples/models/j2s7s300_arm_sphere_collision_v2.sdf");
    ModelInstanceIndex jaco = Parser(plant).AddModelFromFile(robot_file);
    RigidTransformd X_jaco(RollPitchYaw<double>(0, 0, M_PI_2),
                           Vector3d(0, 0.27, 0.11));
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base"),
                      X_jaco);
    plant->disable_gravity(jaco);

    // Add a manipuland with sphere contact
    std::string manipuland_file =
        FindResourceOrThrow("drake/traj_opt/examples/models/box_15cm.sdf");
    Parser(plant).AddAllModelsFromFile(manipuland_file);

    // Add the ground
    const Vector4<double> tan(0.87, 0.7, 0.5, 1.0);
    const Vector4<double> green(0.3, 0.6, 0.4, 1.0);
    RigidTransformd X_ground(Vector3d(0.0, 0.0, -0.5));
    RigidTransformd X_table(Vector3d(0.6, 0.0, -0.499));
    plant->RegisterVisualGeometry(plant->world_body(), X_ground, Box(25, 25, 1),
                                  "ground", green);
    plant->RegisterVisualGeometry(plant->world_body(), X_table,
                                  Box(1.5, 1.5, 1), "table", tan);
    plant->RegisterCollisionGeometry(plant->world_body(), X_ground,
                                     Box(25, 25, 1), "ground",
                                     CoulombFriction<double>(0.05, 0.05));
    // N.B. When combined with the friction coefficient of the box according to
    // μ = 2μₘμₙ/(μₘ + μₙ), this gives a friction coefficient of roughly 0.1
    // between the box and the ground.
  }

  void CreatePlantModelForSimulation(
      MultibodyPlant<double>* plant) const final {
    // Use hydroelastic contact, and throw instead of point contact fallback
    plant->set_contact_model(multibody::ContactModel::kHydroelastic);
      
    // Add a jaco arm, including gravity, with rigid hydroelastic contact
    std::string robot_file = FindResourceOrThrow(
        "drake/traj_opt/examples/models/j2s7s300_arm_hydro_collision.sdf");
    Parser(plant).AddModelFromFile(robot_file);
    RigidTransformd X_jaco(RollPitchYaw<double>(0, 0, M_PI_2),
                           Vector3d(0, 0.27, 0.11));
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base"),
                      X_jaco);

    // Add a manipuland with compliant hydroelastic contact
    std::string manipuland_file = FindResourceOrThrow(
        "drake/traj_opt/examples/models/box_15cm_hydro.sdf");
    Parser(plant).AddAllModelsFromFile(manipuland_file);

    // Add the ground with compliant hydroelastic contact
    const Vector4<double> tan(0.87, 0.7, 0.5, 1.0);
    const Vector4<double> green(0.3, 0.6, 0.4, 1.0);
    RigidTransformd X_ground(Vector3d(0.0, 0.0, -0.5));
    RigidTransformd X_table(Vector3d(0.6, 0.0, -0.499));
    plant->RegisterVisualGeometry(plant->world_body(), X_ground, Box(25, 25, 1),
                                  "ground", green);
    plant->RegisterVisualGeometry(plant->world_body(), X_table,
                                  Box(1.5, 1.5, 1), "table", tan);

    ProximityProperties ground_proximity;
    AddContactMaterial({}, {}, CoulombFriction<double>(0.05, 0.05),
                       &ground_proximity);
    AddCompliantHydroelasticProperties(0.1, 5e7, &ground_proximity);
    plant->RegisterCollisionGeometry(plant->world_body(), X_ground,
                                     Box(25, 25, 1), "ground",
                                     ground_proximity);
  }
};

}  // namespace jaco
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::jaco::JacoExample example;
  example.RunExample("drake/traj_opt/examples/jaco.yaml");
  return 0;
}
