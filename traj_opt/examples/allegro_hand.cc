#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

DEFINE_bool(upside_down, false,
            "whether to treat the hand as upside down (by reversing gravity)");

namespace drake {
namespace traj_opt {
namespace examples {
namespace allegro_hand {

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

class AllegroHandExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    const Vector4<double> blue(0.1, 0.3, 0.5, 0.8);
    const Vector4<double> black(0.0, 0.0, 0.0, 0.5);

    // Add a model of the hand
    std::string sdf_file =
        FindResourceOrThrow("drake/traj_opt/examples/allegro_hand.sdf");
    Parser(plant).AddAllModelsFromFile(sdf_file);
    RigidTransformd X_hand(RollPitchYawd(0, -M_PI_2, 0), Vector3d(0, 0, 0));
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("hand_root"),
                      X_hand);

    // Define gravity (so we can turn the hand upside down)
    if (FLAGS_upside_down) {
      plant->mutable_gravity_field().set_gravity_vector(Vector3d(0, 0, 9.81));
    }

    // Add a free-floating ball
    ModelInstanceIndex ball_idx = plant->AddModelInstance("ball");

    const double mass = 0.05;
    const double radius = 0.06;

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

}  // namespace allegro_hand
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  drake::traj_opt::examples::allegro_hand::AllegroHandExample example;
  std::string yaml_file;
  if (FLAGS_upside_down) {
    yaml_file = "drake/traj_opt/examples/allegro_hand_upside_down.yaml";
  } else {
    yaml_file = "drake/traj_opt/examples/allegro_hand.yaml";
  }
  example.RunExample(yaml_file);

  return 0;
}
