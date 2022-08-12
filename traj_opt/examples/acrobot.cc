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
        FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);
  }
  solver_params.max_iterations = FLAGS_max_iters;
  solver_params.max_linesearch_iterations = 50;

  // Establish an initial guess
  std::vector<VectorXd> q_guess;
  for (int t = 0; t <= num_steps; ++t) {
    q_guess.push_back(opt_prob.q_init);
  }

  // Set the indices of anactuated DOF
  opt_prob.unactuated_dof = {0};

  // Solve the optimzation problem
  TrajectoryOptimizer<double> optimizer(&plant, opt_prob, solver_params);
  TrajectoryOptimizerSolution<double> solution;
  TrajectoryOptimizerStats<double> stats;

  SolverFlag status = optimizer.Solve(q_guess, &solution, &stats);

  if (status == SolverFlag::kSuccess) {
    std::cout << "Solved in " << stats.solve_time[-1] << " seconds."
              << std::endl;
    // Report maximum torques applied to the unactuated shoulder and actuated
    // elbow.
    double max_unactuated_torque = 0;
    double max_actuated_torque = 0;
    double unactuated_torque;
    double actuated_torque;
    for (int t = 0; t < num_steps; ++t) {
      unactuated_torque = abs(solution.tau[t](0));
      actuated_torque = abs(solution.tau[t](1));
      if (unactuated_torque > max_unactuated_torque) {
        max_unactuated_torque = unactuated_torque;
      }
      if (actuated_torque > max_actuated_torque) {
        max_actuated_torque = actuated_torque;
      }
    }
    std::cout << "Maximum actuated torque: " << max_actuated_torque
              << std::endl;
    std::cout << "Maximum unactuated torque: " << max_unactuated_torque
              << std::endl;
  }

  // Save data to CSV, if requested
  if (FLAGS_save_data) {
    stats.SaveToCsv("acrobot_data.csv");
  }

  // Play back the result on the visualizer
  if (FLAGS_visualize) {
    play_back_trajectory(solution.q, time_step);
  }
}

int do_main() {
  // Solve an optimization problem to swing-up the acrobot
  solve_trajectory_optimization(FLAGS_time_step, FLAGS_num_steps);

  return 0;
}

}  // namespace acrobot
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::acrobot::AcrobotExample acrobot_example;
  acrobot_example.SolveTrajectoryOptimization(
      "drake/traj_opt/examples/acrobot.yaml");
  return 0;
}
