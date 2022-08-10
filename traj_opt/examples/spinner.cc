#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
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

// Command line options
DEFINE_double(time_step, 1e-2,
              "Discretization timestep for the optimizer (seconds).");
DEFINE_int32(num_steps, 200,
             "Number of timesteps in the optimization problem.");
DEFINE_int32(max_iters, 100,
             "Maximum number of Gauss-Newton iterations to take.");

DEFINE_bool(visualize, true, "Flag for displaying the optimal solution.");
DEFINE_string(linesearch, "armijo",
              "Linesearch strategy, {backtracking} or {armijo}.");

DEFINE_double(q1_init, 0.2, "Initial angle for the first (finger) joint");
DEFINE_double(q2_init, 1.5, "Initial angle for the second (finger) joint");
DEFINE_double(q3_init, 0.0, "Initial angle for the third (spinner) joint");
DEFINE_double(v1_init, 0.0, "Initial velocity for the first (finger) joint");
DEFINE_double(v2_init, 0.0, "Initial velocity for the second (finger) joint");
DEFINE_double(v3_init, 0.0, "Initial velocity for the third (spinner) joint");

DEFINE_double(q1_nom, 0.2, "Target angle for the first (finger) joint");
DEFINE_double(q2_nom, 1.5, "Target angle for the second (finger) joint");
DEFINE_double(q3_nom, 0.0, "Target angle for the third (spinner) joint");
DEFINE_double(v1_nom, 0.0, "Target velocity for the first (finger) joint");
DEFINE_double(v2_nom, 0.0, "Target velocity for the second (finger) joint");
DEFINE_double(v3_nom, 0.0, "Target velocity for the third (spinner) joint");

DEFINE_double(Qq1, 0.0,
              "Running cost weight on angle for the first (finger) joint");
DEFINE_double(Qq2, 0.0,
              "Running cost weight on angle for the second (finger) joint");
DEFINE_double(Qq3, 0.0,
              "Running cost weight on angle for the third (spinner) joint");
DEFINE_double(Qv1, 0.1,
              "Running cost weight on velocity for the first (finger) joint");
DEFINE_double(Qv2, 0.1,
              "Running cost weight on velocity for the second (finger) joint");
DEFINE_double(Qv3, 0.1,
              "Running cost weight on velocity for the third (spinner) joint");

DEFINE_double(R1, 0.1,
              "Running cost weight on torques for the first (finger) joint");
DEFINE_double(R2, 0.1,
              "Running cost weight on torques for the second (finger) joint");
DEFINE_double(R3, 0.1,
              "Running cost weight on torques for the third (spinner) joint");

DEFINE_double(Qfq1, 10.0,
              "Terminal cost weight on angle for the first (finger) joint");
DEFINE_double(Qfq2, 10.0,
              "Terminal cost weight on angle for the second (finger) joint");
DEFINE_double(Qfq3, 10.0,
              "Terminal cost weight on angle for the third (spinner) joint");
DEFINE_double(Qfv1, 10.0,
              "Terminal cost weight on velocity for the first (finger) joint");
DEFINE_double(Qfv2, 10.0,
              "Terminal cost weight on velocity for the second (finger) joint");
DEFINE_double(Qfv3, 10.0,
              "Terminal cost weight on velocity for the third (spinner) joint");

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
  const double time_step = FLAGS_time_step;
  const int num_steps = FLAGS_num_steps;

  // Create a system model
  // N.B. we need a whole diagram, including scene_graph, to handle contact
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = time_step;
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
  opt_prob.num_steps = num_steps;
  opt_prob.q_init = Vector3d(FLAGS_q1_init, FLAGS_q2_init, FLAGS_q3_init);
  opt_prob.v_init = Vector3d(FLAGS_v1_init, FLAGS_v2_init, FLAGS_v3_init);
  opt_prob.Qq = Vector3d(FLAGS_Qq1, FLAGS_Qq2, FLAGS_Qq3).asDiagonal();
  opt_prob.Qv = Vector3d(FLAGS_Qv1, FLAGS_Qv2, FLAGS_Qv3).asDiagonal();
  opt_prob.Qf_q = Vector3d(FLAGS_Qfq1, FLAGS_Qfq2, FLAGS_Qfq3).asDiagonal();
  opt_prob.Qf_v = Vector3d(FLAGS_Qfv1, FLAGS_Qfv2, FLAGS_Qfv3).asDiagonal();
  opt_prob.R = Vector3d(FLAGS_R1, FLAGS_R2, FLAGS_R3).asDiagonal();
  opt_prob.q_nom = Vector3d(FLAGS_q1_nom, FLAGS_q2_nom, FLAGS_q3_nom);
  opt_prob.v_nom = Vector3d(FLAGS_v1_nom, FLAGS_v2_nom, FLAGS_v3_nom);

  // Set our solver options
  SolverParameters solver_params;
  if (FLAGS_linesearch == "backtracking") {
    solver_params.linesearch_method = LinesearchMethod::kBacktracking;
  } else {
    solver_params.linesearch_method = LinesearchMethod::kArmijo;
  }
  solver_params.max_iterations = FLAGS_max_iters;
  solver_params.max_linesearch_iterations = 50;

  // Establish an initial guess
  std::vector<VectorXd> q_guess;
  for (int t = 0; t <= num_steps; ++t) {
    q_guess.push_back(opt_prob.q_init);
  }

  // Solve the optimzation problem
  TrajectoryOptimizer<double> optimizer(&plant, &plant_context, opt_prob,
                                        solver_params);
  Solution<double> solution;
  SolutionData<double> solution_data;

  SolverFlag status = optimizer.Solve(q_guess, &solution, &solution_data);
  DRAKE_ASSERT(status == SolverFlag::kSuccess);
  std::cout << "Solved in " << solution_data.solve_time << " seconds."
            << std::endl;

  solution.q = q_guess;
  // Play back the result on the visualizer
  if (FLAGS_visualize) {
    play_back_trajectory(solution.q, time_step);
  }
}

/**
 * Simulate the system with zero control input, connected to the visualizer.
 */
void run_passive_simulation() {
  const double dt = FLAGS_time_step;
  const int num_steps = FLAGS_num_steps;
  const double sim_time = dt * num_steps;

  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = dt;
  config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/spinner.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
  ConnectContactResultsToDrakeVisualizer(&builder, plant, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const VectorXd u = VectorXd::Zero(plant.num_actuators());
  plant.get_actuation_input_port().FixValue(&plant_context, u);

  VectorX<double> x0(6);
  x0 << FLAGS_q1_init, FLAGS_q2_init, FLAGS_q3_init, FLAGS_v1_init,
      FLAGS_v2_init, FLAGS_v3_init;

  plant.SetPositionsAndVelocities(&plant_context, x0);

  Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(sim_time);
}

int do_main() {
  // run_passive_simulation();
  solve_trajectory_optimization();
  return 0;
}

}  // namespace spinner
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::traj_opt::examples::spinner::do_main();
}
