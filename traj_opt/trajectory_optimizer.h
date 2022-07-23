#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/problem_data.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/solution_data.h"

namespace drake {
namespace traj_opt {

using Eigen::VectorXd;
using multibody::MultibodyPlant;
using systems::Context;

class TrajectoryOptimizer {
 public:
  /**
   * Construct a new Trajectory Optimizer object.
   *
   * @param plant A model of the system that we're trying to find an optimal
   *              trajectory for.
   * @param prob Problem definition, including cost, initial and target states,
   *             etc.
   */
  TrajectoryOptimizer(std::unique_ptr<const MultibodyPlant<double>> plant,
                      const ProblemDefinition& prob);

  /**
   * Solve the optimization problem
   *
   *    min  x_err(T)'*Qf*x_err(T) + sum{ x_err(t)'*Q*x_err(t) + u(t)'*R*u(t) }
   *    s.t. x(0) = x0
   *         multibody dynamics with contact
   *
   * using our proposed formulation, where generalized positions q are the only
   * decision variables, we use an implicit formulation of the dynamics, and we
   * exploit the sparsity of the problem as much as possible.
   *
   * @param q_guess An initial guess for the optimal trajectory. Doesn't need to
   *                be dynamically feasible.
   * @param soln Optimal solution to the problem, including state x = [q;v] and
   *             control inputs u.
   * @param stats Additional solver statistics such as timing information,
   *              optimal cost, etc.
   * @return SolverFlag indicating whether our optimization was successful, or
   *         why it failed.
   */
  SolverFlag Solve(const std::vector<VectorXd>& q_guess, Solution* soln,
                   SolverStats* stats);

  /**
   * Convienience function to get the timestep of this optimization problem.
   *
   * @return double dt, the time step for this optimization problem
   */
  double time_step() const { return plant_->time_step(); }

  /**
   * Convienience function to get the time horizon of this optimization problem.
   *
   * @return double T, the number of time steps in the optimal trajectory.
   */
  double T() const { return prob_.T; }

  /**
   * Compute a sequence of generalized velocities v from a sequence of
   * generalized positions, where
   *
   *     v_t = (q_t - q_{t-1})/dt
   *
   * and v_0 is defined by the initial state of the optimization problem.
   *
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities
   */
  void CalcV(const std::vector<VectorXd>& q, std::vector<VectorXd>* v) const;

 private:
  // A model of the system that we are trying to find an optimal trajectory for.
  std::unique_ptr<const MultibodyPlant<double>> plant_;

  // A context corresponding to plant_, to enable dynamics computations.
  std::unique_ptr<Context<double>> context_;

  // Stores the problem definition, including cost, time horizon, initial state,
  // target state, etc.
  const ProblemDefinition prob_;
};

}  // namespace traj_opt
}  // namespace drake
