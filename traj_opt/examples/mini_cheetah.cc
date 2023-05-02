#include <drake/multibody/tree/prismatic_joint.h>

#include "drake/common/find_resource.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace mini_cheetah {

using Eigen::Vector3d;
using geometry::Box;
using geometry::Cylinder;
using math::RigidTransformd;
using math::RollPitchYawd;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::Parser;

/**
 * A model of the MIT mini cheetah quadruped.
 */
class MiniCheetahExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const {
    const Vector4<double> green(0.3, 0.6, 0.4, 0.5);

    // Add the robot
    std::string urdf_file = FindResourceOrThrow(
        "drake/traj_opt/examples/models/mini_cheetah_mesh.urdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);

    // Add collision with the ground
    RigidTransformd X_ground(Vector3d(0.0, 0.0, -5.0));
    plant->RegisterVisualGeometry(plant->world_body(), X_ground,
                                  Box(25, 25, 10), "ground", green);
    plant->RegisterCollisionGeometry(plant->world_body(), X_ground,
                                     Box(25, 25, 10), "ground",
                                     CoulombFriction<double>(0.5, 0.5));

    // Add some extra terrain
    RigidTransformd X_terrain(RollPitchYawd(M_PI_2, 0, 0),
                              Vector3d(2.0, 0.0, -0.9));
    plant->RegisterVisualGeometry(plant->world_body(), X_terrain,
                                  Cylinder(1, 25), "hill", green);
    plant->RegisterCollisionGeometry(plant->world_body(), X_terrain,
                                     Cylinder(1, 25), "hill",
                                     CoulombFriction<double>(0.5, 0.5));
  }
};

}  // namespace mini_cheetah
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::mini_cheetah::MiniCheetahExample example;
  example.RunExample("drake/traj_opt/examples/mini_cheetah.yaml");
  return 0;
}
