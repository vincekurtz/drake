#pragma once

namespace drake {
namespace traj_opt {

enum LinesearchMethod {
  // Simple backtracking linesearch with Armijo's condition
  kBacktrackingArmijo,

  // Backtracking linesearch that tries to find a local minimum
  kBacktracking
};

struct SolverParameters {
  // Which linesearch strategy to use
  LinesearchMethod linesearch_method = LinesearchMethod::kBacktrackingArmijo;

  // Maximum number of Gauss-Newton iterations
  int max_iterations = 100;

  // Maximum number of linesearch iterations
  int max_linesearch_iterations = 50;
};

}  // namespace traj_opt
}  // namespace drake
