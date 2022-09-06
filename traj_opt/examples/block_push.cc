#include "drake/common/find_resource.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"
#include <drake/multibody/tree/prismatic_joint.h>

namespace drake {
namespace traj_opt {
namespace examples {
namespace block_push {

using Eigen::Vector3d;
using geometry::Box;
using geometry::Sphere;
using math::RigidTransformd;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::PrismaticJoint;
using multibody::UnitInertia;

/**
 * The simplest possible system with a floating base: a box that floats through
 * the air.
 */
class BlockPushExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const {
    // Colors that we'll use
    const Vector4<double> red(0.9, 0.1, 0.0, 1.0);
    const Vector4<double> green(0.3, 0.6, 0.4, 0.5);
    const Vector4<double> blue(0.1, 0.3, 0.5, 1.0);

    // Block (unactuated) that we'll push
    const double l = 0.2;
    const double w = 0.2;
    const double h = 0.2;
    const double block_mass = 0.1;
    const SpatialInertia<double> I_block(
        block_mass, Vector3d::Zero(),
        UnitInertia<double>::SolidBox(l, w, h));
    const RigidBody<double>& block =
        plant->AddRigidBody("block", I_block);
    plant->RegisterVisualGeometry(block, RigidTransformd::Identity(),
                                  Box(l, w, h), "box_visual", blue);
    plant->RegisterCollisionGeometry(block, RigidTransformd::Identity(),
                                     Box(l, w, h), "box_collision",
                                     CoulombFriction<double>());

    // Ground is modeled as a large box
    RigidTransformd X_ground(Vector3d(0.0, 0.0, -5.0));
    plant->RegisterVisualGeometry(plant->world_body(), X_ground,
                                  Box(25, 25, 10), "ground", green);
    plant->RegisterCollisionGeometry(plant->world_body(), X_ground,
                                     Box(25, 25, 10), "ground",
                                     CoulombFriction<double>());

    // Pusher is a sphere that can move in 3d, but doesn't rotate
    //const double radius = 0.05;
    //const double pusher_mass = 0.05;
    //const SpatialInertia<double> I_pusher(
    //    pusher_mass, Vector3d::Zero(),
    //    UnitInertia<double>::SolidSphere(radius));
    //const SpatialInertia<double> I_dummy(
    //    0.0, Vector3d::Zero(),
    //    UnitInertia<double>::SolidSphere(radius));
    //const RigidBody<double>& pusher = plant->AddRigidBody("pusher", I_pusher);
    //const RigidBody<double>& dummy_one = plant->AddRigidBody("dummy_one", I_dummy);
    //const RigidBody<double>& dummy_two = plant->AddRigidBody("dummy_two", I_dummy);
    //plant->RegisterVisualGeometry(pusher, RigidTransformd::Identity(),
    //                              Sphere(radius), "pusher_visual", red);

    //plant->AddJoint<PrismaticJoint>("pusher_x", plant->world_body(), {}, dummy_two,
    //                                {}, Vector3d(1, 0, 0));
    //plant->AddJoint<PrismaticJoint>("pusher_y", dummy_one, {}, dummy_two,
    //                                {}, Vector3d(0, 1, 0));
    //plant->AddJoint<PrismaticJoint>("pusher_joint", plant->world_body(), {}, pusher,
    //                                {}, Vector3d(1, 0, 0));
  }
};

}  // namespace block_push
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::block_push::BlockPushExample example;
  example.SolveTrajectoryOptimization(
      "drake/traj_opt/examples/block_push.yaml");
  return 0;
}
