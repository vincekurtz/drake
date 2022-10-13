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
  const int nv = plant.num_velocities();

  auto diagram = builder.Build();

  // Define the optimization problem
  ProblemDefinition opt_prob;
  SetProblemDefinition(options, &opt_prob, plant);

  // Set our solver parameters
  SolverParameters solver_params;
  SetSolverParameters(options, &solver_params, opt_prob.unactuated_dof);

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
  ConvergenceReason reason;
  SolverFlag status = optimizer.Solve(q_guess, &solution, &stats, &reason);
  if (status == SolverFlag::kSuccess) {
    std::cout << "Solved in " << stats.solve_time << " s and "
              << static_cast<int>(stats.iteration_times.size())
              << " Gauss-Newton iterations." << std::endl;
  } else if (status == SolverFlag::kMaxIterationsReached) {
    std::cout << "Maximum iterations reached in " << stats.solve_time
              << " seconds." << std::endl;
  } else {
    std::cout << "Solver failed!" << std::endl;
  }
  std::cout << std::endl;

  if (options.augmented_lagrangian) {
    for (int i = 0; i < static_cast<int>(stats.major_iteration_times.size());
         ++i) {
      std::cout << "Major iteration " << i << " was solved in "
                << stats.major_iteration_times[i] << " s and in "
                << stats.major_num_gn_iterations[i]
                << " minor iterations.\n\tmax. violation: "
                << stats.major_max_violations[i]
                << ", final position cost: " << stats.major_final_pos_costs[i]
                << "\n\tconvergence reason: "
                << DecodeConvergenceReasons(stats.major_convergence_reasons[i])
                << std::endl;
    }
  } else {
    std::cout << "Convergence reason: "
              << DecodeConvergenceReasons(reason) + ".\n";
  }

  // Save the major iteration data into a CSV file
  stats.SaveMajorToCsv("al_solver_stats.csv");

  // Report maximum torques on all DoF
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
  std::cout << "Max. joint forces : " << tau_max.transpose() << std::endl;

  // Report desired and final state
  std::cout << std::endl;
  std::cout << "q_nom[T] : " << opt_prob.q_nom[options.num_steps].transpose()
            << std::endl;
  std::cout << "q[T]     : " << solution.q[options.num_steps].transpose()
            << std::endl;
  std::cout << std::endl;
  std::cout << "v_nom[T] : " << opt_prob.v_nom[options.num_steps].transpose()
            << std::endl;
  std::cout << "v[T]     : " << solution.v[options.num_steps].transpose()
            << std::endl;

  // Print speed profiling info
  std::cout << std::endl;
  std::cout << TableOfAverages() << std::endl;

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

  geometry::DrakeVisualizerParams vis_params;
  vis_params.role = geometry::Role::kIllustration;
  DrakeVisualizerd::AddToBuilder(&builder, scene_graph, {}, vis_params);

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

void TrajOptExample::SetProblemDefinition(
    const TrajOptExampleParams& options, ProblemDefinition* opt_prob,
    const MultibodyPlant<double>& plant) const {
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
    if (options.q_init.size() == options.v_init.size()) {
      // No quaternion DoF, so compute v_nom from q_nom
      opt_prob->v_nom.push_back((opt_prob->q_nom[t] - opt_prob->q_nom[t - 1]) /
                                options.time_step);
    } else {
      // Set v_nom = v_init for systems with quaternion DoF
      // TODO(vincekurtz): enable better specification of v_nom for
      // floating-base systems
      opt_prob->v_nom.push_back(opt_prob->v_init);
    }
  }

  // Use the provided unactuated DoF indices if available
  // Otherwise, find the indices for the unactuated DoF
  if (options.overwrite_unactuated_dof) {
    opt_prob->unactuated_dof = options.unactuated_dof_indices;
  } else {
    // Get the actuation matrix for the plant
    // TODO(vincekurtz): deal with the fact that B is not well-defined for
    // some systems, such as the block pusher and floating box examples.
    MatrixXd B = plant.MakeActuationMatrix();
    // Derive the indices of the unactuated DoF from the actuation matrix
    for (int i = 0; i < plant.num_velocities(); ++i) {
      // Get the index if the actuation matrix does not select any actuators
      if (B.row(i).sum() == 0) {
        opt_prob->unactuated_dof.push_back(i);
      }
    }
  }
  // Print out the resulting indices, if any
  if (options.verbose && !opt_prob->unactuated_dof.empty()) {
    std::cout << "Unactuated DoF indices: ";
    for (int i = 0; i < static_cast<int>(opt_prob->unactuated_dof.size()); ++i)
      std::cout << opt_prob->unactuated_dof[i] << " ";
    std::cout << std::endl;
  }

  // Suppress the effort penalties for unactuated DoF when the augmented
  // Lagrangian solver is enabled
  if (options.augmented_lagrangian) {
    for (auto i : opt_prob->unactuated_dof) {
      if (opt_prob->R(i, i) > 0) {
        if (options.verbose) {
          std::cout << "Suppressing the R term for the unactuated DoF " << i
                    << " to accommodate the augmented Lagrangian solver\n";
        }
        opt_prob->R(i, i) = 0.0;
      }
    }
  }
}

void TrajOptExample::SetSolverParameters(
    const TrajOptExampleParams& options, SolverParameters* solver_params,
    const std::vector<int>& unactuated_dof) const {
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

  if (options.linear_solver == "pentadiagonal_lu") {
    solver_params->linear_solver =
        SolverParameters::LinearSolverType::kPentaDiagonalLu;
  } else if (options.linear_solver == "dense_ldlt") {
    solver_params->linear_solver =
        SolverParameters::LinearSolverType::kDenseLdlt;
  } else if (options.linear_solver == "petsc") {
    solver_params->linear_solver = SolverParameters::LinearSolverType::kPetsc;
  } else {
    throw std::runtime_error(
        fmt::format("Unknown linear solver '{}'", options.linear_solver));
  }

  solver_params->petsc_parameters.relative_tolerance =
      options.petsc_rel_tolerance;

  // PETSc solver type.
  if (options.petsc_solver == "cg") {
    solver_params->petsc_parameters.solver_type =
        SolverParameters::PetscSolverPatameters::SolverType::kConjugateGradient;
  } else if (options.petsc_solver == "direct") {
    solver_params->petsc_parameters.solver_type =
        SolverParameters::PetscSolverPatameters::SolverType::kDirect;
  } else if (options.petsc_solver == "minres") {
    solver_params->petsc_parameters.solver_type =
        SolverParameters::PetscSolverPatameters::SolverType::kMINRES;
  } else {
    throw std::runtime_error(
        fmt::format("Unknown PETSc solver '{}'", options.petsc_solver));
  }

  // PETSc preconditioner.
  if (options.petsc_preconditioner == "none") {
    solver_params->petsc_parameters.preconditioner_type =
        SolverParameters::PetscSolverPatameters::PreconditionerType::kNone;
  } else if (options.petsc_preconditioner == "chol") {
    solver_params->petsc_parameters.preconditioner_type =
        SolverParameters::PetscSolverPatameters::PreconditionerType::kCholesky;
  } else if (options.petsc_preconditioner == "ichol") {
    solver_params->petsc_parameters.preconditioner_type = SolverParameters::
        PetscSolverPatameters::PreconditionerType::kIncompleteCholesky;
  } else {
    throw std::runtime_error(fmt::format("Unknown PETSc preconditioner '{}'",
                                         options.petsc_preconditioner));
  }

  solver_params->max_iterations = options.max_iters;
  solver_params->max_linesearch_iterations = 60;
  solver_params->print_debug_data = options.print_debug_data;
  solver_params->linesearch_plot_every_iteration =
      options.linesearch_plot_every_iteration;

  solver_params->convergence_tolerances = options.tolerances;

  solver_params->proximal_operator = options.proximal_operator;
  solver_params->rho_proximal = options.rho_proximal;

  // Augmented Lagrangian solver parameters
  solver_params->augmented_lagrangian = options.augmented_lagrangian;
  // Require underactuation if the solver is enabled
  if (solver_params->augmented_lagrangian) {
    DRAKE_DEMAND(!unactuated_dof.empty());
  }
  solver_params->update_init_guess = options.update_init_guess;
  solver_params->max_major_iterations = options.max_major_iterations;
  // Overwrite the maximum number of major iterations if the solver is disabled
  if (!options.augmented_lagrangian) {
    solver_params->max_major_iterations = 1;
  }
  solver_params->lambda0 = options.lambda0;
  solver_params->mu0 = options.mu0;
  solver_params->mu_expand_coef = options.mu_expand_coef;
  solver_params->constraint_tol = options.constraint_tol;

  // Set contact parameters
  // TODO(vincekurtz): figure out a better place to set these
  solver_params->F = options.F;
  solver_params->delta = options.delta;
  solver_params->stiffness_exponent = options.stiffness_exponent;
  solver_params->dissipation_velocity = options.dissipation_velocity;
  solver_params->dissipation_exponent = options.dissipation_exponent;
  solver_params->friction_coefficient = options.friction_coefficient;
  solver_params->stiction_velocity = options.stiction_velocity;

  solver_params->force_at_a_distance = options.force_at_a_distance;
  solver_params->smoothing_factor = options.smoothing_factor;

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

  // Flag for printing iteration data
  solver_params->verbose = options.verbose;
}

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
