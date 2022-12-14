#include <thread>

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

  if (MPC) {
    // Use the optimizer as a controller via MPC

    // Start an LCM instance
    lcm::DrakeLcm lcm_instance();

    // Simulator options
    const Eigen::Vector2d q0(0.3, 0.0);  // TODO: read from YAML
    const double sim_time_step = 1e-3;
    const double sim_time = 5.0;

    // Start the simulator, which reads control inputs and publishes the system
    // state over LCM
    std::thread sim_thread(&AcrobotExample::SimulateWithControlFromLcm,
                           &acrobot_example, q0, sim_time_step, sim_time);

    // Start the controller, which reads the system state and publishes
    // control torques over LCM
    std::thread counter_thread(&AcrobotExample::CountToTen, &acrobot_example);

    // Wait for all threads to stop
    sim_thread.join();
    counter_thread.join();

  } else {
    // Just solve for a single trajectory and play it on the visualizer
    acrobot_example.SolveTrajectoryOptimization(
        "drake/traj_opt/examples/acrobot.yaml");
  }

  return 0;
}

}  // namespace acrobot
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() { return drake::traj_opt::examples::acrobot::do_main(); }
