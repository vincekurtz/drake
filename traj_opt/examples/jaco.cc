#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace jaco {

using Eigen::Vector3d;
using geometry::Box;
using math::RigidTransformd;
using multibody::CoulombFriction;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::Parser;

class JacoExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    const Vector4<double> blue(0.1, 0.3, 0.5, 1.0);
    const Vector4<double> green(0.3, 0.6, 0.4, 1.0);
    const Vector4<double> black(0.0, 0.0, 0.0, 1.0);

    // Add a jaco arm
    std::string robot_file = FindResourceOrThrow(
        "drake/traj_opt/examples/models/j2s7s300_arm_sphere_collision_v2.sdf");
    ModelInstanceIndex jaco = Parser(plant).AddModelFromFile(robot_file);
    RigidTransformd X_jaco(Vector3d(0, -0.27, 0.11));
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base"),
                      X_jaco);
    plant->disable_gravity(jaco);

    // Add a manipuland
    std::string manipuland_file =
        FindResourceOrThrow("drake/traj_opt/examples/models/box_intel_nuc.sdf");
    Parser(plant).AddAllModelsFromFile(manipuland_file);

    // Add the ground
    RigidTransformd X_ground(Vector3d(0.0, 0.0, -0.5));
    plant->RegisterVisualGeometry(plant->world_body(), X_ground, Box(25, 25, 1),
                                  "ground", green);
    plant->RegisterCollisionGeometry(plant->world_body(), X_ground,
                                     Box(25, 25, 1), "ground",
                                     CoulombFriction<double>(0.5, 0.5));
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
