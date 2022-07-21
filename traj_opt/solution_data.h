#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

using Eigen::VectorXd;

/**
 * A container for the optimal trajectory, including both state and control.
 */
struct Solution {
  // Generalized positions for each timestep along the optimal trajectory
  std::vector<VectorXd> q_star;

  // Generalized velocities for each timestep along the optimal trajectory
  std::vector<VectorXd> v_star;

  // Control inputs for each timestep along the optimal trajectory
  std::vector<VectorXd> u_star;
};

/**
 * A container for various statistics from the optimization problem, such as
 * solve time and iteration details.
 */
struct SolverStats {
  // Total time (in seconds) it took to solve the optimization problem
  double solve_time;

  // Cost of the (locally) optimal solution we found
  double optimal_cost;

  // Total number of Gauss-Newton iterations
  int num_iters;

  // Time (in seconds) for each Gauss-Newton iteration
  std::vector<double> iter_times;

  // Number of linesearch iterations required for each Gauss-Newton iteration
  std::vector<int> linesearch_iters;
};

/**
 * Output flags indicating whether our optimization routine was successful, or
 * if not, why it failed.
 */
enum SolverFlag {
    // The solver successfully found a locally optimal trajectory
    kSuccess,

    // The linesearch procedure failed
    kLinesearchFailed,

    // The maximum number of Gauss-Newton iterations was reached
    kMaxIterationsReached
};

}  // namespace traj_opt
}  // namespace drake
