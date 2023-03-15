#include "drake/traj_opt/examples/example_base.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "drake/examples/acrobot/acrobot_lcm.h"
#include "drake/lcmt_acrobot_u.hpp"
#include "drake/lcmt_traj_opt_u.hpp"
#include "drake/lcmt_traj_opt_x.hpp"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/traj_opt/examples/lcm_interfaces.h"

namespace drake {
namespace traj_opt {
namespace examples {

using systems::lcm::LcmInterfaceSystem;
using systems::lcm::LcmPublisherSystem;
using systems::lcm::LcmSubscriberSystem;

void TrajOptExample::RunExample(const std::string options_file) const {
  // Load parameters from file
  TrajOptExampleParams default_options;
  TrajOptExampleParams options = yaml::LoadYamlFile<TrajOptExampleParams>(
      FindResourceOrThrow(options_file), {}, default_options);

  if (options.mpc) {
    // Do MPC with a simulator in one thread and a controller in another,
    // communicating over LCM
    RunModelPredictiveControl(options);
  } else {
    // Solve a single instance of the optimization problem and play back the
    // result on the visualizer
    SolveTrajectoryOptimization(options);
  }
}

void TrajOptExample::RunModelPredictiveControl(
    const TrajOptExampleParams& options) const {
  // Perform a full solve to convergence (as defined by YAML parameters) to
  // warm-start the first MPC iteration. Subsequent MPC iterations will be
  // warm-started based on the prior MPC iteration.
  TrajectoryOptimizerSolution<double> initial_solution =
      SolveTrajectoryOptimization(options);

  // Start an LCM instance
  lcm::DrakeLcm lcm_instance();

  // Start the simulator, which reads control inputs and publishes the system
  // state over LCM
  std::thread sim_thread(&TrajOptExample::SimulateWithControlFromLcm, this,
                         options);

  // Start the controller, which reads the system state and publishes
  // control torques over LCM
  std::thread ctrl_thread(&TrajOptExample::ControlWithStateFromLcm, this,
                          options, initial_solution.q);

  // Wait for all threads to stop
  sim_thread.join();
  ctrl_thread.join();

  // Print profiling info
  std::cout << TableOfAverages() << std::endl;
}

void TrajOptExample::ControlWithStateFromLcm(
    const TrajOptExampleParams& options,
    const std::vector<VectorXd>& q_guess) const {
  // Create a system model for the controller
  DiagramBuilder<double> builder_ctrl;
  MultibodyPlantConfig config;
  config.time_step = options.time_step;
  config.default_floating_joint_type = options.floating_joint_type;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder_ctrl);
  CreatePlantModel(&plant);
  plant.Finalize();
  auto diagram_ctrl = builder_ctrl.Build();

  // Define the optimization problem
  ProblemDefinition opt_prob;
  SetProblemDefinition(options, &opt_prob);

  // Set our solver parameters
  SolverParameters solver_params;
  SetSolverParameters(options, &solver_params);
  solver_params.max_iterations = options.mpc_iters;

  // Here we'll set up a whole separate system diagram with LCM reciever,
  // controller, and LCM publisher:
  //
  //    state_subscriber -> controller -> command_publisher
  //
  DiagramBuilder<double> builder;
  auto lcm = builder.AddSystem<LcmInterfaceSystem>();

  auto state_subscriber = builder.AddSystem(
      LcmSubscriberSystem::Make<lcmt_traj_opt_x>("traj_opt_x", lcm));

  auto controller = builder.AddSystem<TrajOptLcmController>(
      diagram_ctrl.get(), &plant, opt_prob, q_guess, solver_params);

  auto command_publisher =
      builder.AddSystem(LcmPublisherSystem::Make<lcmt_traj_opt_u>(
          "traj_opt_u", lcm, 1. / options.controller_frequency));

  builder.Connect(state_subscriber->get_output_port(),
                  controller->get_input_port());
  builder.Connect(controller->get_output_port(),
                  command_publisher->get_input_port());

  // Run this system diagram, which recieves states over LCM and publishes
  // controls over LCM
  auto diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(options.sim_time / options.sim_realtime_rate);
}

void TrajOptExample::SimulateWithControlFromLcm(
    const TrajOptExampleParams& options) const {
  // Set up the system diagram for the simulator
  DiagramBuilder<double> builder;
  auto lcm = builder.AddSystem<LcmInterfaceSystem>();

  // Construct the multibody plant system model
  MultibodyPlantConfig config;
  config.time_step = options.sim_time_step;
  config.default_floating_joint_type = options.floating_joint_type;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  CreatePlantModel(&plant);
  plant.Finalize();

  // Connect to the visualizer
  geometry::DrakeVisualizerParams vis_params;
  vis_params.role = geometry::Role::kIllustration;
  DrakeVisualizerd::AddToBuilder(&builder, scene_graph, {}, vis_params);
  multibody::ConnectContactResultsToDrakeVisualizer(&builder, plant,
                                                    scene_graph);

  // Recieve control inputs from LCM
  auto command_subscriber = builder.AddSystem(
      LcmSubscriberSystem::Make<lcmt_traj_opt_u>("traj_opt_u", lcm));

  // Set PD gains for lower level controller to zero if they aren't defined
  VectorXd Kp = options.Kp;
  VectorXd Kd = options.Kd;
  if (Kp.size() != options.q_init.size()) {
    Kp.setZero(options.q_init.size());
  }
  if (Kd.size() != options.v_init.size()) {
    Kd.setZero(options.v_init.size());
  }

  // Connect the low-level controller
  auto controller =
      builder.AddSystem<LowLevelController>(&plant, Kp, Kd);
  builder.Connect(command_subscriber->get_output_port(),
                  controller->get_control_input_port());
  builder.Connect(plant.get_state_output_port(),
                  controller->get_state_estimate_input_port());
  builder.Connect(controller->get_output_port(),
                  plant.get_actuation_input_port());

  // Send state estimates out over LCM
  auto state_sender = builder.AddSystem<StateSender>(plant.num_positions(),
                                                     plant.num_velocities());
  auto state_publisher = builder.AddSystem(
      LcmPublisherSystem::Make<lcmt_traj_opt_x>("traj_opt_x", lcm));
  builder.Connect(plant.get_state_output_port(),
                  state_sender->get_input_port());
  builder.Connect(state_sender->get_output_port(),
                  state_publisher->get_input_port());

  // Compile the diagram
  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Run the simulation
  plant.SetPositions(&plant_context, options.q_init);
  plant.SetVelocities(&plant_context, options.v_init);
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(options.sim_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(options.sim_time);
}

TrajectoryOptimizerSolution<double> TrajOptExample::SolveTrajectoryOptimization(
    const TrajOptExampleParams& options) const {
  // Create a system model
  // N.B. we need a whole diagram, including scene_graph, to handle contact
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = options.time_step;
  config.default_floating_joint_type = options.floating_joint_type;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  CreatePlantModel(&plant);
  plant.Finalize();
  const int nv = plant.num_velocities();

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
    PlayBackTrajectory(opt_prob.q_nom, options);
  }
  if (options.play_initial_guess) {
    PlayBackTrajectory(q_guess, options);
  }

  // Solve the optimzation problem
  TrajectoryOptimizer<double> optimizer(diagram.get(), &plant, opt_prob,
                                        solver_params);
  TrajectoryOptimizerSolution<double> solution;
  TrajectoryOptimizerStats<double> stats;
  ConvergenceReason reason;
  SolverFlag status = optimizer.Solve(q_guess, &solution, &stats, &reason);
  if (status == SolverFlag::kSuccess) {
    std::cout << "Solved in " << stats.solve_time << " seconds." << std::endl;
  } else if (status == SolverFlag::kMaxIterationsReached) {
    std::cout << "Maximum iterations reached in " << stats.solve_time
              << " seconds." << std::endl;
  } else {
    std::cout << "Solver failed!" << std::endl;
  }

  std::cout << "Convergence reason: "
            << DecodeConvergenceReasons(reason) + ".\n";

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

  // Report maximum actuated and unactuated torques
  // TODO(vincekurtz): deal with the fact that B is not well-defined for some
  // systems, such as the block pusher and floating box examples.
  const MatrixXd B = plant.MakeActuationMatrix();
  double tau_max_unactuated = 0;
  double tau_max_actuated = 0;
  for (int i = 0; i < nv; ++i) {
    if (B.row(i).sum() == 0) {
      if (tau_max(i) > tau_max_unactuated) {
        tau_max_unactuated = tau_max(i);
      }
    } else {
      if (tau_max(i) > tau_max_actuated) {
        tau_max_actuated = tau_max(i);
      }
    }
  }

  std::cout << std::endl;
  std::cout << "Max actuated torque   : " << tau_max_actuated << std::endl;
  std::cout << "Max unactuated torque : " << tau_max_unactuated << std::endl;

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
    PlayBackTrajectory(solution.q, options);
  }

  return solution;
}

void TrajOptExample::PlayBackTrajectory(const std::vector<VectorXd>& q,
                                        const TrajOptExampleParams& options) const {
  // Create a system diagram that includes the plant and is connected to
  // the Drake visualizer
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = options.time_step;
  config.default_floating_joint_type = options.floating_joint_type;

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
    diagram_context->SetTime(t * options.time_step);
    plant.SetPositions(&plant_context, q[t]);
    diagram->ForcedPublish(*diagram_context);

    // Hack to make the playback roughly realtime
    // TODO(vincekurtz): add realtime rate option?
    std::this_thread::sleep_for(
        std::chrono::duration<double>(options.time_step));
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
    if (options.q_init.size() == options.v_init.size()) {
      // No quaternion DoFs, so compute v_nom from q_nom
      opt_prob->v_nom.push_back((opt_prob->q_nom[t] - opt_prob->q_nom[t - 1]) /
                                options.time_step);
    } else {
      // Set v_nom = v_init for systems with quaternion DoFs
      // TODO(vincekurtz): enable better specification of v_nom for
      // floating-base systems
      opt_prob->v_nom.push_back(opt_prob->v_init);
    }
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

  // Set contact parameters
  // TODO(vincekurtz): figure out a better place to set these
  solver_params->F = options.F;
  solver_params->delta = options.delta;
  solver_params->dissipation_velocity = options.dissipation_velocity;
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

  // Whether to normalize quaterions between iterations
  solver_params->normalize_quaternions = options.normalize_quaternions;

  // Type of Hessian approximation
  solver_params->exact_hessian = options.exact_hessian;

  // Hessian rescaling
  solver_params->scaling = options.scaling;

  if (options.scaling_method == "sqrt") {
    solver_params->scaling_method = ScalingMethod::kSqrt;
  } else if (options.scaling_method == "adaptive_sqrt") {
    solver_params->scaling_method = ScalingMethod::kAdaptiveSqrt;
  } else if (options.scaling_method == "double_sqrt") {
    solver_params->scaling_method = ScalingMethod::kDoubleSqrt;
  } else if (options.scaling_method == "adaptive_double_sqrt") {
    solver_params->scaling_method = ScalingMethod::kAdaptiveDoubleSqrt;
  } else {
    throw std::runtime_error(
        fmt::format("Unknown scaling method '{}'", options.scaling_method));
  }

  // Equality constriant (unactuated DoF torques) enforcement
  solver_params->equality_constraints = options.equality_constraints;

  // Maximum and initial trust region radius
  solver_params->Delta0 = options.Delta0;
  solver_params->Delta_max = options.Delta_max;
}

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
