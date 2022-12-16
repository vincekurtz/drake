#pragma once

/// This file contains tools for sending and recieving state and control
/// information for trajectory optimization over LCM. Based heavily on the
/// example in drake/acrobot/acrobot_lcm.h.

#include "drake/lcmt_traj_opt_u.hpp"
#include "drake/lcmt_traj_opt_x.hpp"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace traj_opt {
namespace examples {

using systems::BasicVector;
using systems::Context;
using systems::LeafSystem;

/// Recieves the output of an LcmSubscriberSystem that subscribes to a channel
/// with control inputs, of type lcmt_traj_opt_u, and outputs the same control
/// inputs as a BasicVector.
class CommandReciever : public LeafSystem<double> {
 public:
  /**
   * @param nu number of control torques/forces to send
   */
  CommandReciever(const int nu);

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
  TrajOptLcmController(const int nq, const int nv, const int nu);

 private:
  void OutputCommand(const Context<double>& context,
                     lcmt_traj_opt_u* output) const;
  const int nq_;
  const int nv_;
  const int nu_;
};

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
