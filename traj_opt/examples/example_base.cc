#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {

void TrajOptExample::SolveTrajectoryOptimization(
    const std::string options_file) const {
  // Load parameters from file
  TrajOptExampleParams default_options;
  TrajOptExampleParams options = yaml::LoadYamlFile<TrajOptExampleParams>(
      FindResourceOrThrow(options_file), {}, default_options);

  // Create a system model
  // N.B. we need a whole diagram, including scene_graph, to handle contact
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = options.time_step;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  CreatePlantModel(&plant);
  plant.Finalize();
  const int nv = plant.num_positions();

  auto diagram = builder.Build();

  // Define the optimization problem
  ProblemDefinition opt_prob;
  SetProblemDefinition(options, &opt_prob);

  // Set our solver parameters
  SolverParameters solver_params;
  SetSolverParameters(options, &solver_params);

  // Establish an initial guess
  std::vector<VectorXd> q_guess = MakeLinearInterpolation(
      opt_prob.q_init, options.q_guess, opt_prob.num_steps + 1);

  // Visualize the target trajectory and initial guess, if requested
  if (options.play_target_trajectory) {
    PlayBackTrajectory(opt_prob.q_nom, options.time_step);
  }
  if (options.play_initial_guess) {
    PlayBackTrajectory(q_guess, options.time_step);
  }

  // Solve the optimzation problem
  TrajectoryOptimizer<double> optimizer(diagram.get(), &plant, opt_prob,
                                        solver_params);

  TrajectoryOptimizerSolution<double> solution;
  TrajectoryOptimizerStats<double> stats;
  SolverFlag status = optimizer.Solve(q_guess, &solution, &stats);
  if (status != SolverFlag::kSuccess) {
    std::cout << "Solver failed!" << std::endl;
  } else {
    std::cout << "Solved in " << stats.solve_time << " seconds." << std::endl;
  }

  // Report maximum torques on all DoFs
  VectorXd tau_max = VectorXd::Zero(nv);
  VectorXd abs_tau_t = VectorXd::Zero(nv);
  for (int t = 0; t < options.num_steps; ++t) {
    abs_tau_t = solution.tau[t].cwiseAbs();
    for (int i = 0; i < nv; ++i) {
      if (abs_tau_t(i) > tau_max(i)) {
        tau_max(i) = abs_tau_t(i);
      }
    }
  }
  std::cout << std::endl;
  std::cout << "Max torques: " << tau_max.transpose() << std::endl;

  // Report desired and final state
  std::cout << std::endl;
  std::cout << "q_nom[t] : " << opt_prob.q_nom[options.num_steps].transpose()
            << std::endl;
  std::cout << "q[T]     : " << solution.q[options.num_steps].transpose()
            << std::endl;
  std::cout << std::endl;
  std::cout << "v_nom[t] : " << opt_prob.v_nom[options.num_steps].transpose()
            << std::endl;
  std::cout << "v[T]     : " << solution.v[options.num_steps].transpose()
            << std::endl;

  // Save stats to CSV for later plotting
  if (options.save_solver_stats_csv) {
    stats.SaveToCsv("solver_stats.csv");
  }

  // Play back the result on the visualizer
  if (options.play_optimal_trajectory) {
    PlayBackTrajectory(solution.q, options.time_step);
  }
}

void TrajOptExample::PlayBackTrajectory(const std::vector<VectorXd>& q,
                                        const double time_step) const {
  // Create a system diagram that includes the plant and is connected to
  // the Drake visualizer
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = time_step;

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  CreatePlantModel(&plant);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const VectorXd u = VectorXd::Zero(plant.num_actuators());
  plant.get_actuation_input_port().FixValue(&plant_context, u);

  // Step through q, setting the plant positions at each step accordingly
  const int N = q.size();
  for (int t = 0; t < N; ++t) {
    diagram_context->SetTime(t * time_step);
    plant.SetPositions(&plant_context, q[t]);
    diagram->Publish(*diagram_context);

    // Hack to make the playback roughly realtime
    // TODO(vincekurtz): add realtime rate option?
    std::this_thread::sleep_for(std::chrono::duration<double>(time_step));
  }
}

void TrajOptExample::SetProblemDefinition(const TrajOptExampleParams& options,
                                          ProblemDefinition* opt_prob) const {
  opt_prob->num_steps = options.num_steps;

  // Initial state
  opt_prob->q_init = options.q_init;
  opt_prob->v_init = options.v_init;

  // Cost weights
  opt_prob->Qq = options.Qq.asDiagonal();
  opt_prob->Qv = options.Qv.asDiagonal();
  opt_prob->Qf_q = options.Qfq.asDiagonal();
  opt_prob->Qf_v = options.Qfv.asDiagonal();
  opt_prob->R = options.R.asDiagonal();

  // Target state at each timestep
  opt_prob->q_nom = MakeLinearInterpolation(
      options.q_nom_start, options.q_nom_end, options.num_steps + 1);

  opt_prob->v_nom.push_back(opt_prob->v_init);
  for (int t = 1; t <= options.num_steps; ++t) {
    // TODO(vincekurtz): handle quaternion DoFs
    opt_prob->v_nom.push_back((opt_prob->q_nom[t] - opt_prob->q_nom[t - 1]) /
                              options.time_step);
  }
}

void TrajOptExample::SetSolverParameters(
    const TrajOptExampleParams& options,
    SolverParameters* solver_params) const {
  if (options.linesearch == "backtracking") {
    solver_params->linesearch_method = LinesearchMethod::kBacktracking;
  } else if (options.linesearch == "armijo") {
    solver_params->linesearch_method = LinesearchMethod::kArmijo;
  } else {
    throw std::runtime_error(
        fmt::format("Unknown linesearch method '{}'", options.linesearch));
  }

  if (options.gradients_method == "forward_differences") {
    solver_params->gradients_method = GradientsMethod::kForwardDifferences;
  } else if (options.gradients_method == "central_differences") {
    solver_params->gradients_method = GradientsMethod::kCentralDifferences;
  } else if (options.gradients_method == "central_differences4") {
    solver_params->gradients_method = GradientsMethod::kCentralDifferences4;
  } else if (options.gradients_method == "autodiff") {
    solver_params->gradients_method = GradientsMethod::kAutoDiff;
  } else {
    throw std::runtime_error(
        fmt::format("Unknown gradient method '{}'", options.gradients_method));
  }

  if (options.method == "linesearch") {
    solver_params->method = SolverMethod::kLinesearch;
  } else if (options.method == "trust_region") {
    solver_params->method = SolverMethod::kTrustRegion;
  } else {
    throw std::runtime_error(
        fmt::format("Unknown solver method '{}'", options.method));
  }

  solver_params->max_iterations = options.max_iters;
  solver_params->max_linesearch_iterations = 60;
  solver_params->print_debug_data = options.print_debug_data;
  solver_params->linesearch_plot_every_iteration =
      options.linesearch_plot_every_iteration;

  solver_params->proximal_operator = options.proximal_operator;
  solver_params->rho_proximal = options.rho_proximal;

  // Set contact parameters
  // TODO(vincekurtz): figure out a better place to set these
  solver_params->F = options.F;
  solver_params->delta = options.delta;
  solver_params->stiffness_exponent = options.stiffness_exponent;
  solver_params->dissipation_velocity = options.dissipation_velocity;
  solver_params->dissipation_exponent = options.dissipation_exponent;
  solver_params->friction_coefficient = options.friction_coefficient;
  solver_params->stiction_velocity = options.stiction_velocity;

  // Set parameters for making contour plot of the first two variables
  solver_params->save_contour_data = options.save_contour_data;
  solver_params->contour_q1_min = options.contour_q1_min;
  solver_params->contour_q1_max = options.contour_q1_max;
  solver_params->contour_q2_min = options.contour_q2_min;
  solver_params->contour_q2_max = options.contour_q2_max;

  // Parameters for making line plots of the first variable
  solver_params->save_lineplot_data = options.save_lineplot_data;
  solver_params->lineplot_q_min = options.lineplot_q_min;
  solver_params->lineplot_q_max = options.lineplot_q_max;
}

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
