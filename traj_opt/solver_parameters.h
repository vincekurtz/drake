#pragma once

namespace drake {
namespace traj_opt {

enum LinesearchMethod {
  // Simple backtracking linesearch with Armijo's condition
  kArmijo,

  // Backtracking linesearch that tries to find a local minimum
  kBacktracking
};

struct SolverParameters {
  // Which linesearch strategy to use
  LinesearchMethod linesearch_method = LinesearchMethod::kArmijo;

  // Maximum number of Gauss-Newton iterations
  int max_iterations = 100;

  // Maximum number of linesearch iterations
  int max_linesearch_iterations = 50;

  // Flag for whether to print out iteration data
  bool verbose = true;

  // Flag for whether to print (and compute) additional slow-to-compute
  // debugging info, like the condition number, at each iteration
  bool print_debug_data = false;

  // Flag for whether to record linesearch data to a file at each iteration (for
  // later plotting)
  bool linesearch_plot_every_iteration = false;
};

}  // namespace traj_opt
}  // namespace drake
