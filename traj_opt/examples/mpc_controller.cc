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
    : time_step_(plant->time_step()),
      num_steps_(q_guess.size()),
      nq_(plant->num_positions()),
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
  stored_trajectory_ = this->DeclareAbstractState(Value<StoredTrajectory>());
  this->DeclarePeriodicUnrestrictedUpdateEvent(
      replan_period, 0, &ModelPredictiveController::UpdateAbstractState);
}

EventStatus ModelPredictiveController::UpdateAbstractState(
    const Context<double>& context, State<double>* state) const {
  std::cout << "Resolving at t=" << context.get_time() << std::endl;
  // Get the latest initial condition
  const VectorXd& x0 = EvalVectorInput(context, state_input_port_)->value();
  const auto& q0 = x0.topRows(nq_);
  const auto& v0 = x0.bottomRows(nv_);

  // Get a reference to the previous solution stored in the discrete state
  StoredTrajectory& stored_trajectory =
      state->get_mutable_abstract_state<StoredTrajectory>(stored_trajectory_);

  // Set the initial guess from the stored solution
  if (context.get_time() > 0) {
    UpdateInitialGuess(stored_trajectory, context.get_time(), &q_guess_);
  }
  q_guess_[0] = q0;  // guess must be consistent with the initial condition

  // Solve the trajectory optimization problem from the new initial condition
  optimizer_.ResetInitialConditions(q0, v0);
  TrajectoryOptimizerStats<double> stats;
  TrajectoryOptimizerSolution<double> solution;
  optimizer_.Solve(q_guess_, &solution, &stats);

  // Store the result in the discrete state
  StoreOptimizerSolution(solution, context.get_time(), &stored_trajectory);

  return EventStatus::Succeeded();
}

void ModelPredictiveController::UpdateInitialGuess(
    const StoredTrajectory& stored_trajectory,
    const double current_time,
    std::vector<VectorXd>* q_guess) const {
  DRAKE_DEMAND(static_cast<int>(q_guess->size()) == num_steps_);

  for (int i=0; i < num_steps_; ++i) {
    const double t =
        i * time_step_ + current_time - stored_trajectory.start_time;
    q_guess->at(i) = stored_trajectory.q.value(t);
  }
}

void ModelPredictiveController::StoreOptimizerSolution(
    const TrajectoryOptimizerSolution<double>& solution,
    const double start_time, StoredTrajectory* stored_trajectory) const {
  // Set up knot points for a polynomial interpolation
  // N.B. PiecewisePolynomial requires std::vector<MatrixXd> rather than
  // std::vector<VectorXd>
  std::vector<double> time_steps;
  std::vector<MatrixXd> q_knots(num_steps_, VectorXd(nq_));
  std::vector<MatrixXd> v_knots(num_steps_, VectorXd(nq_));
  std::vector<MatrixXd> u_knots(num_steps_, VectorXd(nq_));

  for (int i = 0; i < num_steps_; ++i) {
    // Time steps
    time_steps.push_back(i * time_step_);

    // Generalized positions and velocities
    q_knots[i] = solution.q[i];
    v_knots[i] = solution.v[i];

    // Control inputs, which are undefined at the last time step
    if (i == num_steps_ - 1) {
        u_knots[i] = B_.transpose() * solution.tau[i - 1];
    } else {
        u_knots[i] = B_.transpose() * solution.tau[i];
    }
  }

  // Perform polynomial interpolation and store the result in our struct
  stored_trajectory->start_time = start_time;
  stored_trajectory->q =
      PiecewisePolynomial<double>::FirstOrderHold(time_steps, q_knots);
  stored_trajectory->v =
      PiecewisePolynomial<double>::FirstOrderHold(time_steps, v_knots);
  stored_trajectory->u =
      PiecewisePolynomial<double>::FirstOrderHold(time_steps, u_knots);
}

void ModelPredictiveController::SendControlTorques(
    const Context<double>& context, BasicVector<double>* output) const {
  // Get the current time and state estimate
  const double t = context.get_time();
  const VectorXd& x = EvalVectorInput(context, state_input_port_)->value();
  const auto& q = x.topRows(nq_);
  const auto& v = x.bottomRows(nv_);

  // TODO: handle initial step more gracefully
  if (t > 0) {
    // Get the nominal state and input for this timestep from the latest
    // trajectory optimization
    const StoredTrajectory& traj =
        context.get_abstract_state<StoredTrajectory>(stored_trajectory_);
    VectorXd q_nom = traj.q.value(t - traj.start_time);
    VectorXd v_nom = traj.v.value(t - traj.start_time);
    VectorXd u_nom = traj.u.value(t - traj.start_time);

    // Set control torques according to a PD controller
    Eigen::VectorBlock<VectorXd> u = output->get_mutable_value();
    u = u_nom - Kp_ * (q - q_nom) - Kd_ * (v - v_nom);
  } else {
    output->set_value(VectorXd::Zero(nu_));
  }
}

}  // namespace mpc
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake