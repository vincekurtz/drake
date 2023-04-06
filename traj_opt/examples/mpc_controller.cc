#include "drake/traj_opt/examples/mpc_controller.h"

#include <iostream>

namespace drake {
namespace traj_opt {
namespace examples {
namespace mpc {

ModelPredictiveController::ModelPredictiveController(
    const Diagram<double>* diagram, const MultibodyPlant<double>* plant,
    const ProblemDefinition& prob, const std::vector<VectorXd>& q_guess,
    const SolverParameters& params, const MatrixXd& Kp, const MatrixXd& Kd,
    const double replan_period)
    : nq_(plant->num_positions()),
      nv_(plant->num_velocities()),
      nu_(plant->num_actuators()),
      B_(plant->MakeActuationMatrix()),
      Kp_(Kp),
      Kd_(Kd),
      q_guess_(q_guess),
      optimizer_(diagram, plant, prob, params) {
  // Some size sanity checks
  DRAKE_DEMAND(Kp_.rows() == nu_);
  DRAKE_DEMAND(Kp_.cols() == nq_);
  DRAKE_DEMAND(Kd_.rows() == nu_);
  DRAKE_DEMAND(Kd_.cols() == nv_);
  DRAKE_DEMAND(static_cast<int>(q_guess.size()) == prob.num_steps + 1);
  DRAKE_DEMAND(static_cast<int>(q_guess[0].size()) == nq_);

  // Input port recieves state estimates
  state_input_port_ = this->DeclareVectorInputPort(
                              "state_estimate", BasicVector<double>(nq_ + nv_))
                          .get_index();

  // Output port sends control torques
  control_output_port_ = this->DeclareVectorOutputPort(
                                 "control_torques", BasicVector<double>(nu_),
                                 &ModelPredictiveController::SendControlTorques)
                             .get_index();

  // Discrete state stores optimal trajectories
  stored_trajectory_ =
      this->DeclareAbstractState(Value<PiecewisePolynomial<double>>());
  this->DeclarePeriodicUnrestrictedUpdateEvent(
      replan_period, 0, &ModelPredictiveController::UpdateAbstractState);
}

EventStatus ModelPredictiveController::UpdateAbstractState(
    const Context<double>& context, State<double>* state) const {
  std::cout << "Resolving at t=" << context.get_time() << std::endl;
  // Get the latest initial condition

  // Solve the trajectory optimization problem from the new initial condition

  // Store the result in the discrete state
  PiecewisePolynomial<double>& traj =
      state->get_mutable_abstract_state<PiecewisePolynomial<double>>(
          stored_trajectory_);
  (void) traj;

  return EventStatus::Succeeded();
}

void ModelPredictiveController::SendControlTorques(
    const Context<double>& context, BasicVector<double>* output) const {
  // Get the current time and state estimate
  const double t = context.get_time();
  const VectorXd& x = EvalVectorInput(context, state_input_port_)->value();
  auto q = x.topRows(nq_);
  auto v = x.bottomRows(nv_);

  // Get the nominal state and input for this timestep from the latest
  // trajectory optimization
  (void) t;
  VectorXd q_nom(1);
  VectorXd v_nom(1);
  q_nom << 3.14;
  v_nom << 0.0;
  VectorXd u_nom = VectorXd::Zero(nu_);

  // Set control torques according to a PD controller
  Eigen::VectorBlock<VectorXd> u = output->get_mutable_value();
  u = u_nom - Kp_ * (q - q_nom) - Kd_ * (v - v_nom);
}

}  // namespace mpc
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake