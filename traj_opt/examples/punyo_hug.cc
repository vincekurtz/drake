#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace punyo_hug {

using multibody::MultibodyPlant;
using multibody::Parser;

class PunyoHugExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    std::string urdf_file =
        FindResourceOrThrow("drake/traj_opt/examples/punyoid.sdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base"));
  }
};

}  // namespace spinner
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::punyo_hug::PunyoHugExample example;
  example.SolveTrajectoryOptimization(
      "drake/traj_opt/examples/punyo_hug.yaml");
  return 0;
}
