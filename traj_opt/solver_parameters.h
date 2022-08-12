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

  // Augmented Lagrangian parameters
  bool augmented_lagrangian = false;
  int max_major_iterations = 10;
  double lambda = 10.0;
  double mu = 1.0;
  double omega = 1.0/mu;
  double nu = 1.0/std::pow(mu, 0.1);
};

}  // namespace traj_opt
}  // namespace drake
