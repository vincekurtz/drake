#include <drake/multibody/tree/prismatic_joint.h>

#include "drake/common/find_resource.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace hopper {

using Eigen::Vector3d;
using geometry::Box;
using geometry::Sphere;
using math::RigidTransformd;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::PrismaticJoint;
using multibody::Parser;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;

/**
 * A simple planar hopper, inspired by https://youtu.be/uWADBSmHebA?t=893.
 */
class HopperExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const {
    std::string urdf_file =
        FindResourceOrThrow("drake/traj_opt/examples/hopper.urdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);
  }
};

}  // namespace hopper
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::hopper::HopperExample example;
  example.SolveTrajectoryOptimization("drake/traj_opt/examples/hopper.yaml");
  return 0;
}
