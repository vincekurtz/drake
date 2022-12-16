#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace acrobot {

using multibody::MultibodyPlant;
using multibody::Parser;

class AcrobotExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const {
    const std::string urdf_file =
        FindResourceOrThrow("drake/examples/acrobot/Acrobot_no_collision.urdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base_link"));
  }
};

int do_main() {
  bool MPC = true;
  AcrobotExample acrobot_example;
  const std::string yaml_file = "drake/traj_opt/examples/acrobot.yaml";

  if (MPC) {
    // Use the optimizer for MPC
    const double optimizer_iters = 20;
    acrobot_example.RunModelPredictiveControl(yaml_file, optimizer_iters);

  } else {
    // Just solve for a single trajectory and play it on the visualizer
    acrobot_example.SolveTrajectoryOptimization(yaml_file);
  }

  return 0;
}

}  // namespace acrobot
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() { return drake::traj_opt::examples::acrobot::do_main(); }
