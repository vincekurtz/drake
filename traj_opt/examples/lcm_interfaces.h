#pragma once

/// This file contains tools for sending and recieving state and control
/// information for trajectory optimization over LCM. Based heavily on the
/// example in drake/acrobot/acrobot_lcm.h.

#include <iostream>
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
using systems::LeafSystem;

/// Recieves the output of an LcmSubscriberSystem that subscribes to a channel
/// with control inputs, of type lcmt_traj_opt_u, and outputs the same control
/// inputs as a BasicVector.
class CommandReciever : public LeafSystem<double> {
 public:
  /**
   * @param nu number of control torques/forces to send
   */
  explicit CommandReciever(const int nu);

 private:
  void OutputCommandAsVector(const Context<double>& context,
                             BasicVector<double>* output) const;
  const int nu_;
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
