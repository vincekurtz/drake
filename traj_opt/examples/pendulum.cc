#include <chrono>
#include <iostream>
#include <thread>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace pendulum {

using geometry::DrakeVisualizerd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;
using systems::Simulator;

/**
 * Just run a simple passive simulation of the pendulum, connected to the Drake
 * visualizer.
 *
 * @param time_step Time step for discretization (seconds)
 * @param sim_time How long to simulate for (seconds)
 */
void run_passive_simulation(double time_step, double sim_time) {
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = time_step;
  config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  const std::string urdf_file =
      FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const double u = 0;
  VectorX<double> x0(2);
  x0 << 0.5, 0.1;
  plant.get_actuation_input_port().FixValue(&plant_context, u);
  plant.SetPositionsAndVelocities(&plant_context, x0);

  Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(sim_time);
}

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
      FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const double u = 0;
  plant.get_actuation_input_port().FixValue(&plant_context, u);

  const int N = q.size();
  for (int t = 0; t < N; ++t) {
    diagram_context->SetTime(t*time_step);
    plant.SetPositions(&plant_context, q[t]);
    diagram->Publish(*diagram_context);

    // Hack to make the playback roughly realtime
    std::this_thread::sleep_for(std::chrono::duration<double>(time_step));
  }
}

/**
 * Solve a trajectory optimization problem that swings the pendulum upright.
 *
 * Then play back an animation of the optimal trajectory using the Drake
 * visualizer.
 *
 * @param time_step Time step for discretization (seconds)
 * @param num_steps Number of steps in the optimization problem.
 */
void solve_trajectory_optimization(double time_step, int num_steps) {
  // Create a system model
  MultibodyPlant<double> plant(time_step);
  const std::string urdf_file =
      FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  // Set up an optimization problem
  ProblemDefinition opt_prob;
  opt_prob.num_steps = num_steps;
  opt_prob.q_init = Vector1d(0.0);
  opt_prob.v_init = Vector1d(0.0);
  opt_prob.Qq = 1.0 * MatrixXd::Identity(1, 1);
  opt_prob.Qv = 1.0 * MatrixXd::Identity(1, 1);
  opt_prob.Qf_q = 1.0 * MatrixXd::Identity(1, 1);
  opt_prob.Qf_v = 1.0 * MatrixXd::Identity(1, 1);
  opt_prob.R = 1.0 * MatrixXd::Identity(1, 1);
  opt_prob.q_nom = Vector1d(1.5);
  opt_prob.v_nom = Vector1d(-0.1);

  // Establish an initial guess
  std::vector<VectorXd> q_guess;
  q_guess.push_back(opt_prob.q_init);
  for (int t = 1; t <= num_steps; ++t) {
    q_guess.push_back(q_guess[t-1] + 0.1 * time_step * Vector1d(1.0));
  }

  // Solve the optimzation problem
  TrajectoryOptimizer<double> optimizer(&plant, opt_prob);
  TrajectoryOptimizerSolution<double> solution;

  SolverFlag status = optimizer.Solve(q_guess, &solution);
  (void) status;

  // Play back the result on the visualizer
  play_back_trajectory(solution.q, time_step);
}

int do_main() {
  // For now we'll just run a simple passive simulation of the pendulum
  // run_passive_simulation(1e-2, 2.0);

  // Solve an optimization problem to swing-up the pendulum
  solve_trajectory_optimization(1e-2, 20);

  return 0;
}

}  // namespace pendulum
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() { return drake::traj_opt::examples::pendulum::do_main(); }
