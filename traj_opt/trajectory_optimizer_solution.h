#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

// Status indicator for the overall success of our trajectory optimization.
enum SolverFlag { kSuccess, kFailed };

/**
 * A container for the optimal solution, including generalized positions,
 * velocities, and forces.
 *
 * TODO(vincekurtz): consider holding control inputs u rather than generalized
 * forces tau (tau = B*u)
 */
template <typename T>
struct TrajectoryOptimizerSolution {
  // Optimal sequence of generalized positions at each timestep
  std::vector<MatrixX<T>> q;

  // Optimal sequence of generalized velocities at each timestep
  std::vector<MatrixX<T>> v;

  // Optimal sequence of generalized forces at each timestep
  std::vector<MatrixX<T>> tau;
};

}  // namespace traj_opt
}  // namespace drake
