#include <iostream>

#include "drake/common/find_resource.h"
#include "drake/common/profiler.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace airhockey {

using Eigen::Vector3d;
using geometry::Box;
using geometry::Cylinder;
using geometry::Sphere;
using math::RigidTransformd;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::PlanarJoint;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;

class AirHockeyExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    // Colors that we'll use
    const Vector4<double> red(0.9, 0.1, 0.0, 1.0);
    const Vector4<double> blue(0.1, 0.3, 0.5, 1.0);
    const Vector4<double> black(0.0, 0.0, 0.0, 1.0);

    // Puck parameters
    const double mass = 0.1;
    const double radius = 0.1;
    const double height = 0.05;

    const SpatialInertia<double> I(
        mass, Vector3d::Zero(),
        UnitInertia<double>::SolidCylinder(radius, height));

    // Create the pusher
    const RigidBody<double>& pusher = plant->AddRigidBody("pusher", I);
    plant->RegisterVisualGeometry(pusher, RigidTransformd(),
                                  Cylinder(radius, height), "pusher", red);
    plant->RegisterVisualGeometry(
        pusher, RigidTransformd(Vector3d(0.0, 0.0, height)),
        Box(radius / 2, radius / 2, height), "handle", red);
    plant->RegisterCollisionGeometry(pusher, RigidTransformd::Identity(), Sphere(radius),
                                     "pusher_collision", CoulombFriction<double>());

    plant->AddJoint<PlanarJoint>("pusher_joint", plant->world_body(),
                                 RigidTransformd(), pusher, {},
                                 Vector3d::Zero());

    // Create the puck
    const RigidBody<double>& puck = plant->AddRigidBody("puck", I);
    plant->RegisterVisualGeometry(puck, RigidTransformd(),
                                  Cylinder(radius, height), "puck", blue);
    plant->RegisterVisualGeometry(puck, RigidTransformd(),
                                  Box(radius / 2, radius / 2, 1.01 * height),
                                  "marker", black);
    plant->RegisterCollisionGeometry(puck, RigidTransformd::Identity(), Sphere(radius),
                                     "puck_collision", CoulombFriction<double>());

    plant->AddJoint<PlanarJoint>("puck_joint", plant->world_body(),
                                 RigidTransformd(), puck, {}, Vector3d::Zero());
  }
};

}  // namespace airhockey
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::airhockey::AirHockeyExample example;
  example.SolveTrajectoryOptimization("drake/traj_opt/examples/airhockey.yaml");
  return 0;
}
