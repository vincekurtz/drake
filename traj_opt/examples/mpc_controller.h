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
using systems::OutputPort;
using systems::LeafSystem;
using systems::State;
using trajectories::PiecewisePolynomial;

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
   * @param q_guess Initial guess for the first MPC iteration
   * @param params Solver parameters, including frequency, PD gains, etc
   * @param Kp proportional gain of size (nu x nq)
   * @param Kd derivative gain of size (nu x nv)
   * @param replan_period time (in seconds) between optimizer solves
   */
  ModelPredictiveController(const Diagram<double>* diagram,
                            const MultibodyPlant<double>* plant,
                            const ProblemDefinition& prob,
                            const std::vector<VectorXd>& q_guess,
                            const SolverParameters& params, const MatrixXd& Kp,
                            const MatrixXd& Kd, const double replan_period);

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

  // Number of positions, velocities, and actuators
  const int nq_;
  const int nv_;
  const int nu_;

  // Actuator selection matrix
  const MatrixXd B_;

  // Matrices of PD gains, of size (nu x nq) and (nu x nv)
  const MatrixXd Kp_;
  const MatrixXd Kd_;

  // Initial guess of sequence of generalized positions, used to warm start the
  // optimizer at each step.
  mutable std::vector<VectorXd> q_guess_;

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
