#pragma once

#include <string>

#include "drake/common/eigen_types.h"
#include "drake/common/yaml/yaml_io.h"
#include "drake/traj_opt/convergence_criteria_tolerances.h"

namespace drake {
namespace traj_opt {
namespace examples {

using Eigen::VectorXd;

/**
 * A simple object which stores parameters that define an optimization problem
 * and various options, and can be loaded from a YAML file.
 *
 * See, e.g., spinner.yaml for an explanation of each field, and
 * https://drake.mit.edu/doxygen_cxx/group__yaml__serialization.html for details
 * on loading options from YAML.
 */
struct TrajOptExampleParams {
  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(q_init));
    a->Visit(DRAKE_NVP(v_init));
    a->Visit(DRAKE_NVP(q_nom_start));
    a->Visit(DRAKE_NVP(q_nom_end));
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
    a->Visit(DRAKE_NVP(gradients_method));
    a->Visit(DRAKE_NVP(method));
    a->Visit(DRAKE_NVP(proximal_operator));
    a->Visit(DRAKE_NVP(rho_proximal));
    a->Visit(DRAKE_NVP(play_optimal_trajectory));
    a->Visit(DRAKE_NVP(play_initial_guess));
    a->Visit(DRAKE_NVP(play_target_trajectory));
    a->Visit(DRAKE_NVP(linesearch_plot_every_iteration));
    a->Visit(DRAKE_NVP(print_debug_data));
    a->Visit(DRAKE_NVP(save_solver_stats_csv));
    a->Visit(DRAKE_NVP(F));
    a->Visit(DRAKE_NVP(delta));
    a->Visit(DRAKE_NVP(stiffness_exponent));
    a->Visit(DRAKE_NVP(dissipation_velocity));
    a->Visit(DRAKE_NVP(dissipation_exponent));
    a->Visit(DRAKE_NVP(stiction_velocity));
    a->Visit(DRAKE_NVP(friction_coefficient));
    a->Visit(DRAKE_NVP(save_contour_data));
    a->Visit(DRAKE_NVP(contour_q1_min));
    a->Visit(DRAKE_NVP(contour_q1_max));
    a->Visit(DRAKE_NVP(contour_q2_min));
    a->Visit(DRAKE_NVP(contour_q2_max));
    a->Visit(DRAKE_NVP(save_lineplot_data));
    a->Visit(DRAKE_NVP(lineplot_q_min));
    a->Visit(DRAKE_NVP(lineplot_q_max));
    a->Visit(DRAKE_NVP(tolerances));
  }
  // Initial state
  VectorXd q_init;
  VectorXd v_init;

  // Nominal state at each timestep is defined by linear interpolation between
  // q_nom_start and q_nom_end
  VectorXd q_nom_start;
  VectorXd q_nom_end;

  // Initial guess is defined by linear interpolation between q_init and q_guess
  VectorXd q_guess;

  // Running cost weights (diagonal matrices)
  VectorXd Qq;
  VectorXd Qv;
  VectorXd R;

  // Terminal cost weights (diagonal matrices)
  VectorXd Qfq;
  VectorXd Qfv;

  // Time step size, in seconds
  double time_step;

  // Number of time steps in the optimization problem
  int num_steps;

  // Maximum number of iterations
  int max_iters;

  // Convergence tolerances
  ConvergenceCriteriaTolerances tolerances;

  // Linesearch method, "backtracking" or "armijo"
  std::string linesearch{"armijo"};

  // Optimization method, "linesearch" or "trust_region"
  std::string method;

  // Method of computing gradients, "forward_differences",
  // "central_differences", "central_differences4" or "autodiff"
  std::string gradients_method{"forward_differences"};

  // Whether to add a proximal operator term to the cost (essentially adds to
  // the diagonal of the Hessian)
  bool proximal_operator = false;
  double rho_proximal = 1e-8;

  // Flags for playing back the target trajectory, initital guess, and optimal
  // trajectory on the visualizer
  bool play_optimal_trajectory = true;
  bool play_initial_guess = false;
  bool play_target_trajectory = false;

  // Save cost along the linesearch direction to linesearch_data.csv
  bool linesearch_plot_every_iteration = false;

  // Print additional debugging data
  bool print_debug_data = false;

  // Save convergence data to solver_stats.csv
  bool save_solver_stats_csv = true;

  // Contact model parameters
  double F = 1.0;
  double delta = 0.01;
  double stiffness_exponent = 2;
  double dissipation_velocity = 0.1;
  double dissipation_exponent = 1.0;
  double stiction_velocity = 0.05;
  double friction_coefficient = 0.0;

  // Save data for a 2d contour plot of cost/gradient/Hessian w.r.t. the first
  // two variables to contour_data.csv
  bool save_contour_data = false;
  double contour_q1_min = 0;
  double contour_q1_max = 1;
  double contour_q2_min = 0;
  double contour_q2_max = 1;

  // Save data for plotting the cost/gradient/Hessian w.r.t. the first variable
  // to lineplot_data.csv
  bool save_lineplot_data = false;
  double lineplot_q_min = 0;
  double lineplot_q_max = 1;
};

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
