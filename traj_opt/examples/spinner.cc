#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace spinner {

using multibody::MultibodyPlant;
using multibody::Parser;

class SpinnerExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const {
    const std::string urdf_file =
        FindResourceOrThrow("drake/traj_opt/examples/spinner.urdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);
  }
};

}  // namespace spinner
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::spinner::SpinnerExample spinner_example;
  spinner_example.SolveTrajectoryOptimization(
      "drake/traj_opt/examples/spinner.yaml");
  return 0;
}
