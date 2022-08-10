#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

// Status indicator for the overall success of our trajectory optimization.
enum SolverFlag { kSuccess, kLinesearchMaxIters };

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
  std::vector<VectorX<T>> q;

  // Optimal sequence of generalized velocities at each timestep
  std::vector<VectorX<T>> v;

  // Optimal sequence of generalized forces at each timestep
  std::vector<VectorX<T>> tau;
};

/**
 * A container for data about the solve process
 */
template <typename T>
struct TrajectoryOptimizerStats {
  // Total solve time
  double solve_time;

  // Time for each iteration
  std::vector<double> iteration_times;

  // Cost at each iteration
  std::vector<T> iteration_costs;

  // Number of linesearch iterations for each outer iteration
  std::vector<int> linesearch_iterations;

  // Linsearch parameter alpha for each iteration
  std::vector<double> linesearch_alphas;

  // Norm of the gradient at each iteration
  std::vector<T> gradient_norm;

  /**
   * Save the solution to a CSV file that we cn process and make plots from
   * later.
   *
   * @param filename filename for the csv file that we'll write to
   */
  void SaveToCsv(std::string fname) {
    // Set file to write to
    std::ofstream data_file;
    data_file.open(fname);

    // Write a header
    data_file << "iter, time, cost, ls_iters, alpha, grad_norm\n";

    const int num_iters = iteration_times.size();
    for (int i = 0; i < num_iters; ++i) {
      // Write the data
      data_file << i << ", ";
      data_file << iteration_times[i] << ", ";
      data_file << iteration_costs[i] << ", ";
      data_file << linesearch_iterations[i] << ", ";
      data_file << linesearch_alphas[i] << ", ";
      data_file << gradient_norm[i] << "\n";
    }

    // Close the file
    data_file.close();
  }
};

}  // namespace traj_opt
}  // namespace drake
