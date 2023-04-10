#pragma once

#include <memory>
#include <vector>

#include <drake/common/trajectories/piecewise_polynomial.h>

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/solver_parameters.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace mpc {

using systems::BasicVector;
using systems::Context;
using systems::Diagram;
using systems::EventStatus;
using systems::InputPort;
using systems::LeafSystem;
using systems::OutputPort;
using systems::State;
using trajectories::PiecewisePolynomial;

/// A little struct for holding an optimal trajectory so that we can track it
/// between MPC solves.
struct StoredTrajectory {
  // Time (in seconds) at which this trajectory was generated
  double start_time{0.0};

  // Generalized positions
  PiecewisePolynomial<double> q;

  // Generalized velocities
  PiecewisePolynomial<double> v;

  // Control torques
  PiecewisePolynomial<double> u;
};

/// An MPC controller that recieves state estimate as input and sends control
/// torques as output. The trajectory optimization problem is re-solved at a
/// fixed frequency (params.controller_frequency), and a low-level PD controller
/// tracks the optimal trajectory between re-solves.
class ModelPredictiveController : public LeafSystem<double> {
 public:
  /**
   * Construct an MPC control system.
   *
   * @param diagram System diagram for the controller's internal model
   * @param plant MultibodyPlant model of the system, part of diagram
   * @param prob Problem definition, including cost, target state, etc
   * @param warm_start_solution Prior solution for first iteration warm-start
   * @param params Solver parameters, including frequency, PD gains, etc
   * @param Kp proportional gain of size (nu x nq)
   * @param Kd derivative gain of size (nu x nv)
   * @param replan_period time (in seconds) between optimizer solves
   */
  ModelPredictiveController(
      const Diagram<double>* diagram, const MultibodyPlant<double>* plant,
      const ProblemDefinition& prob,
      const TrajectoryOptimizerSolution<double>& warm_start_solution,
      const SolverParameters& params, const MatrixXd& Kp, const MatrixXd& Kd,
      const double replan_period);

  const InputPort<double>& get_state_input_port() const {
    return this->get_input_port(state_input_port_);
  }

  const OutputPort<double>& get_control_output_port() const {
    return this->get_output_port(control_output_port_);
  }

 private:
  /**
   * Re-solve the trajectory optimization problem and store the result in the
   * discrete state of this system.
   *
   * @param context current system context storing q and v
   * @param state abstract state used to store the optimal trajectory
   */
  EventStatus UpdateAbstractState(const Context<double>& context,
                                  State<double>* state) const;

  /**
   * Send low-level control torques to the robot according to the PD control law
   *
   *  u = u_nom(t) - Kp (q - q_nom(t)) - Kd (v - v_nom(t)),
   *
   * where u_nom, q_nom, and v_nom come from solving the trajectory optimization
   * problem.
   *
   * @param context current context for this system, storing q and q_nom
   * @param output control torques to set as output
   */
  void SendControlTorques(const Context<double>& context,
                          BasicVector<double>* output) const;

  /**
   * Set an initial guess for trajectory optimization based on the stored
   * previous solution to the trajectory optimization problem.
   *
   * @param stored_trajectory the solution from the previous optimization
   * @param current_time the current time, in seconds
   * @param q_guess the guess (sequence of positions) to set
   */
  void UpdateInitialGuess(const StoredTrajectory& stored_trajectory,
                          const double current_time,
                          std::vector<VectorXd>* q_guess) const;

  /**
   * Store the solution we get from trajectory optimization in a little struct
   * that performs interpolation between time steps.
   *
   * @param solution the trajectory optimization solution
   * @param start_time the time at which the optimization was solved
   * @param stored_trajectory a struct holding the solution
   */
  void StoreOptimizerSolution(
      const TrajectoryOptimizerSolution<double>& solution,
      const double start_time, StoredTrajectory* stored_trajectory) const;

  // Timestep size and total number of steps. We use this to interpolate the
  // solution between timesteps for the low-level PD controller.
  const double time_step_;
  const int num_steps_;

  // Number of positions, velocities, and actuators
  const int nq_;
  const int nv_;
  const int nu_;

  // Actuator selection matrix
  const MatrixXd B_;

  // Matrices of PD gains, of size (nu x nq) and (nu x nv)
  const MatrixXd Kp_;
  const MatrixXd Kd_;

  // Optimizer used to compute control inputs at each time step. Mutable because
  // the stored intitial conditions must be updated at each step.
  mutable TrajectoryOptimizer<double> optimizer_;

  // Indexes for the input and output ports
  int state_input_port_;
  int control_output_port_;

  // Index for the abstract state used to store optimal trajectories
  systems::AbstractStateIndex stored_trajectory_;
};

}  // namespace mpc
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
