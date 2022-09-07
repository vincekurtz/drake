#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace punyo_hug {

using Eigen::Vector3d;
using geometry::Box;
using geometry::Cylinder;
using geometry::Sphere;
using math::RigidTransformd;
using math::RollPitchYawd;
using multibody::CoulombFriction;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;

class PunyoHugExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    const Vector4<double> blue(0.1, 0.3, 0.5, 1.0);
    const Vector4<double> black(0.0, 0.0, 0.0, 1.0);

    // Add a humanoid model
    std::string urdf_file =
        FindResourceOrThrow("drake/traj_opt/examples/punyoid.sdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base"));

    // Add a ground with contact
    RigidTransformd X_ground(Eigen::Vector3d(0.0, 0.0, -5.0));
    plant->RegisterCollisionGeometry(plant->world_body(), X_ground,
                                     Box(25, 25, 10), "ground",
                                     CoulombFriction<double>());

    // Add a free-floating ball to pick up
    ModelInstanceIndex ball_idx = plant->AddModelInstance("ball");

    const double mass = 0.1;
    const double radius = 0.2;

    const SpatialInertia<double> I(mass, Vector3d::Zero(),
                                   UnitInertia<double>::SolidSphere(radius));
    const RigidBody<double>& ball = plant->AddRigidBody("ball", ball_idx, I);

    plant->RegisterVisualGeometry(ball, RigidTransformd::Identity(),
                                  Sphere(radius), "ball_visual", blue);
    plant->RegisterCollisionGeometry(ball, RigidTransformd::Identity(),
                                     Sphere(radius), "ball_collision",
                                     CoulombFriction<double>());

    // Add some markers to the ball so we can see its rotation
    RigidTransformd X_m1(RollPitchYawd(0, 0, 0), Vector3d(0, 0, 0));
    RigidTransformd X_m2(RollPitchYawd(M_PI_2, 0, 0), Vector3d(0, 0, 0));
    RigidTransformd X_m3(RollPitchYawd(0, M_PI_2, 0), Vector3d(0, 0, 0));
    plant->RegisterVisualGeometry(ball, X_m1,
                                  Cylinder(0.1 * radius, 2 * radius),
                                  "ball_marker_one", black);
    plant->RegisterVisualGeometry(ball, X_m2,
                                  Cylinder(0.1 * radius, 2 * radius),
                                  "ball_marker_two", black);
    plant->RegisterVisualGeometry(ball, X_m3,
                                  Cylinder(0.1 * radius, 2 * radius),
                                  "ball_marker_three", black);
  }
};

}  // namespace punyo_hug
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::punyo_hug::PunyoHugExample example;
  example.SolveTrajectoryOptimization("drake/traj_opt/examples/punyo_hug.yaml");
  return 0;
}
