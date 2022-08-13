#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/examples/yaml_config.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace spinner {

using Eigen::Vector3d;
using geometry::DrakeVisualizerd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlant;
using multibody::ConnectContactResultsToDrakeVisualizer;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;
using systems::Simulator;

/**
 * Play back the given trajectory on the Drake visualizer.
 *
 * @param q sequence of generalized positions defining the trajectory
 * @param time_step time step (seconds) for the discretization
 */
void play_back_trajectory(std::vector<VectorXd> q, double time_step) {
  // TODO(vincekurtz): verify size of q
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = time_step;

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/spinner.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const VectorXd u = VectorXd::Zero(plant.num_actuators());
  plant.get_actuation_input_port().FixValue(&plant_context, u);

  const int N = q.size();
  for (int t = 0; t < N; ++t) {
    diagram_context->SetTime(t * time_step);
    plant.SetPositions(&plant_context, q[t]);
    diagram->Publish(*diagram_context);

    // Hack to make the playback roughly realtime
    std::this_thread::sleep_for(std::chrono::duration<double>(time_step));
  }
}

/**
 * Solve a trajectory optimization problem to spin the spinner.
 *
 * Then play back an animation of the optimal trajectory using the Drake
 * visualizer.
 */
void solve_trajectory_optimization() {
  // DEBUG: load parameters from file
  const TrajOptExampleParams options = yaml::LoadYamlFile<TrajOptExampleParams>(
      FindResourceOrThrow("drake/traj_opt/examples/spinner.yaml"));

  // Create a system model
  // N.B. we need a whole diagram, including scene_graph, to handle contact
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = options.time_step;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/spinner.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set up an optimization problem
  ProblemDefinition opt_prob;
  opt_prob.num_steps = options.num_steps;
  opt_prob.q_init = Vector3d(options.q_init.data());
  opt_prob.v_init = Vector3d(options.v_init.data());
  opt_prob.Qq = Vector3d(options.Qq.data()).asDiagonal();
  opt_prob.Qv = Vector3d(options.Qv.data()).asDiagonal();
  opt_prob.Qf_q = Vector3d(options.Qfq.data()).asDiagonal();
  opt_prob.Qf_v = Vector3d(options.Qfv.data()).asDiagonal();
  opt_prob.R = Vector3d(options.R.data()).asDiagonal();
  opt_prob.q_nom = Vector3d(options.q_nom.data());
  opt_prob.v_nom = Vector3d(options.v_nom.data());

  // Set our solver parameters
  SolverParameters solver_params;
  if (options.linesearch == "backtracking") {
    solver_params.linesearch_method = LinesearchMethod::kBacktracking;
  } else {
    solver_params.linesearch_method = LinesearchMethod::kArmijo;
  }
  solver_params.max_iterations = options.max_iters;
  solver_params.max_linesearch_iterations = 60;
  solver_params.print_debug_data = options.print_debug_data;
  solver_params.linesearch_plot_every_iteration =
      options.linesearch_plot_every_iteration;

  // Set contact parameters
  // TODO(vincekurtz): figure out a better place to set these
  solver_params.F = options.F;
  solver_params.delta = options.delta;
  solver_params.n = options.n;

  // Establish an initial guess
  const VectorXd qT_guess = Vector3d(options.q_guess.data());
  std::vector<VectorXd> q_guess;
  double lambda = 0;
  for (int t = 0; t <= options.num_steps; ++t) {
    lambda = (1.0 * t) / (1.0 * options.num_steps);
    q_guess.push_back((1 - lambda) * opt_prob.q_init + lambda * qT_guess);
  }
  if (options.play_initial_guess) {
    play_back_trajectory(q_guess, options.time_step);
  }

  // Solve the optimzation problem
  TrajectoryOptimizer<double> optimizer(&plant, &plant_context, opt_prob,
                                        solver_params);

  TrajectoryOptimizerSolution<double> solution;
  TrajectoryOptimizerStats<double> stats;
  SolverFlag status = optimizer.Solve(q_guess, &solution, &stats);
  DRAKE_ASSERT(status == SolverFlag::kSuccess);
  std::cout << "Solved in " << stats.solve_time << " seconds." << std::endl;
  // Report maximum torques on the unactuated and actuated joints
  double tau_max_f1 = 0;
  double tau_max_f2 = 0;
  double tau_max_s = 0;
  for (int t = 0; t < options.num_steps; ++t) {
    if (abs(solution.tau[t](0)) > tau_max_f1) {
      tau_max_f1 = abs(solution.tau[t](0));
    }
    if (abs(solution.tau[t](1)) > tau_max_f2) {
      tau_max_f2 = abs(solution.tau[t](1));
    }
    if (abs(solution.tau[t](2)) > tau_max_s) {
      tau_max_s = abs(solution.tau[t](2));
    }
  }
  std::cout << std::endl;
  std::cout << "Max finger one torque : " << tau_max_f1 << std::endl;
  std::cout << "Max finger two torque : " << tau_max_f2 << std::endl;
  std::cout << "Max spinner torque    : " << tau_max_s << std::endl;
  // Report desired and final state
  std::cout << std::endl;
  std::cout << "q_nom : " << opt_prob.q_nom.transpose() << std::endl;
  std::cout << "q[T]  : " << solution.q[options.num_steps].transpose()
            << std::endl;
  std::cout << std::endl;
  std::cout << "v_nom : " << opt_prob.v_nom.transpose() << std::endl;
  std::cout << "v[T]  : " << solution.v[options.num_steps].transpose()
            << std::endl;

  // Play back the result on the visualizer
  if (options.play_optimal_trajectory) {
    play_back_trajectory(solution.q, options.time_step);
  }
}

}  // namespace spinner
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::spinner::solve_trajectory_optimization();
  return 0;
}
