#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include "drake/common/find_resource.h"
#include "drake/common/yaml/yaml_io.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace spinner {

// Options from YAML, see spinner.yaml for explanations
struct SpinnerParams {
  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(q_init));
    a->Visit(DRAKE_NVP(v_init));
    a->Visit(DRAKE_NVP(q_nom));
    a->Visit(DRAKE_NVP(v_nom));
    a->Visit(DRAKE_NVP(q_guess));
    a->Visit(DRAKE_NVP(Qq));
    a->Visit(DRAKE_NVP(Qv));
    a->Visit(DRAKE_NVP(R));
    a->Visit(DRAKE_NVP(Qfq));
    a->Visit(DRAKE_NVP(Qfv));
    a->Visit(DRAKE_NVP(time_step));
    a->Visit(DRAKE_NVP(num_steps));
    a->Visit(DRAKE_NVP(max_iters));
    a->Visit(DRAKE_NVP(linesearch));
    a->Visit(DRAKE_NVP(play_optimal_trajectory));
    a->Visit(DRAKE_NVP(play_initial_guess));
  }
  std::vector<double> q_init;
  std::vector<double> v_init;

  std::vector<double> q_nom;
  std::vector<double> v_nom;

  std::vector<double> q_guess;

  std::vector<double> Qq;
  std::vector<double> Qv;
  std::vector<double> R;

  std::vector<double> Qfq;
  std::vector<double> Qfv;

  double time_step;
  int num_steps;

  int max_iters;
  std::string linesearch;

  bool play_optimal_trajectory;
  bool play_initial_guess;
};

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
  const SpinnerParams options = yaml::LoadYamlFile<SpinnerParams>(
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
  solver_params.max_linesearch_iterations = 50;

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
  std::cout << "Solved in " << stats.solve_time << " seconds."
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
