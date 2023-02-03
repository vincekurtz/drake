#pragma once

/// This file contains tools for sending and recieving state and control
/// information for trajectory optimization over LCM. Based heavily on the
/// example in drake/acrobot/acrobot_lcm.h.

#include <iostream>
#include <memory>
#include <vector>

#include "drake/lcmt_traj_opt_u.hpp"
#include "drake/lcmt_traj_opt_x.hpp"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/solver_parameters.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {

using systems::BasicVector;
using systems::Context;
using systems::Diagram;
using systems::InputPort;
using systems::LeafSystem;
using systems::InputPortIndex;
using systems::System;

/// A low-level controller that recieves nominal input and state (x_nom, u_nom)
/// from an LcmSubscriberSystem and state measurement (x_hat) from the plant.
/// Outputs control inputs to be applied directly to the plant, typically at a
/// much higher frequency than we recieve u_nom.
class LowLevelController : public LeafSystem<double> {
 public:
  /**
   * @param plant multibody plant model of the system
   * @param Kp proportional gains for all generalized positions. Gains related
   *           to unactuated DoFs will be ignored.
   * @param Kd derivative gains for all generalized velocities. Gains related
   *           to unactuated DoFs will be ignored.
   */
  LowLevelController(const MultibodyPlant<double>* plant,
                     const MultibodyPlant<double>* plant_lqr, const VectorXd Kp,
                     const VectorXd Kd, const double Vmax_, bool lqr_);

  /**
   * Returns a constant reference to the input port that takes state
   * measurements (x_hat) from the plant.
   */
  const InputPort<double>& get_state_estimate_input_port() const {
    return System<double>::get_input_port(state_estimate_port_);
  }

  /**
   * Returns a constant reference to the input port that takes the nominal
   * control input u_nom and nominal state x_nom from an LcmSubscriberSystem.
   *
   * @return const InputPort<double>&
   */
  const InputPort<double>& get_control_input_port() const {
    return System<double>::get_input_port(control_port_);
  }

 private:
  // Main function used to take inputs and compute outputs
  void OutputCommandAsVector(const Context<double>& context,
                             BasicVector<double>* output) const;

  // Number of actuators, positions, and velocities
  const int nu_;
  const int nq_;
  const int nv_;

  // Actuator selection matrix
  const MatrixXd B_;

  // Matrices of PD gains, of size (nu x nq) and (nu x nv)
  const MatrixXd Kp_;
  const MatrixXd Kd_;

  // User-specified bound on system energy
  const double Vmax_;

  // LQR-related parameters
  const bool lqr_;
  const MatrixXd Q_;
  const MatrixXd R_;

  // Internal system model
  const MultibodyPlant<double>* plant_{nullptr};
  const MultibodyPlant<double>* plant_lqr_{nullptr};
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<Context<double>> context_lqr_;

  InputPortIndex control_port_;
  InputPortIndex state_estimate_port_;
};

/// Recieves the multibody state as input and publishes that same state with
/// type lcmt_traj_opt_x. The typical use case would be to connect this output
/// to an LcmPublisherSystem to publish the states.
class StateSender : public LeafSystem<double> {
 public:
  /**
   * @param nq number of generalized positions in the state
   * @param nv number of generalized velocities in the state
   */
  StateSender(const int nq, const int nv);

 private:
  void OutputState(const Context<double>& context,
                   lcmt_traj_opt_x* output) const;
  const int nq_;
  const int nv_;
};

/// An MPC controller which recieves state information as input (type
/// lcmt_traj_opt_x) and sends control torques as output (type lcmt_traj_opt_u).
///
/// Control torques are determined by solving the trajectory optimizaiton
/// problem for a fixed number of iterations.
class TrajOptLcmController : public LeafSystem<double> {
 public:
  /**
   * Construct an MPC controller that reads state info over LCM and sends
   * control torques over LCM.
   *
   * @param diagram System diagram for the controller's internal model
   * @param plant MultibodyPlant model of the system, part of diagram
   * @param prob Problem definition, including cost, target state, etc
   * @param q_guess Initial guess for the first MPC iteration
   * @param params Solver parameters, including max iterations, etc
   */
  TrajOptLcmController(const Diagram<double>* diagram,
                       const MultibodyPlant<double>* plant,
                       const ProblemDefinition& prob,
                       const std::vector<VectorXd>& q_guess,
                       const SolverParameters& params = SolverParameters{});

 private:
  void OutputCommand(const Context<double>& context,
                     lcmt_traj_opt_u* output) const;

  // Optimizer used to compute control inputs at each time step. Mutable because
  // the stored intitial conditions must be updated at each step.
  mutable TrajectoryOptimizer<double> optimizer_;

  // Initial guess of sequence of generalized positions, used to warm start the
  // optimizer at each step.
  mutable std::vector<VectorXd> q_guess_;

  // Map from generalized forces to control torques
  const MatrixXd B_;

  // Number of positions, velocities, and actuators
  const int nq_;
  const int nv_;
  const int nu_;
};

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
