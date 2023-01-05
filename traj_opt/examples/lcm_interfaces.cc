#include "drake/traj_opt/examples/lcm_interfaces.h"

namespace drake {
namespace traj_opt {
namespace examples {

StateSender::StateSender(const int nq, const int nv) : nq_(nq), nv_(nv) {
  this->DeclareInputPort("multibody_state", systems::kVectorValued, nq_ + nv_);
  this->DeclareAbstractOutputPort("lcmt_traj_opt_x", &StateSender::OutputState);
}

void StateSender::OutputState(const Context<double>& context,
                              lcmt_traj_opt_x* output) const {
  const systems::BasicVector<double>* state = this->EvalVectorInput(context, 0);

  output->utime = static_cast<int64_t>(context.get_time() * 1e6);
  output->nq = static_cast<int16_t>(nq_);
  output->nv = static_cast<int16_t>(nv_);
  output->q.resize(nq_);
  output->v.resize(nv_);

  for (int i = 0; i < nq_; ++i) {
    output->q[i] = state->value()[i];
  }
  for (int j = 0; j < nv_; ++j) {
    output->v[j] = state->value()[nq_ + j];
  }
}

CommandReciever::CommandReciever(const int nu) : nu_(nu) {
  this->DeclareAbstractInputPort("lcmt_traj_opt_u", Value<lcmt_traj_opt_u>());
  this->DeclareVectorOutputPort("control_torques", nu_,
                                &CommandReciever::OutputCommandAsVector);
}

void CommandReciever::OutputCommandAsVector(
    const Context<double>& context,
    systems::BasicVector<double>* output) const {
  const AbstractValue* abstract_command = this->EvalAbstractInput(context, 0);
  const auto& command = abstract_command->get_value<lcmt_traj_opt_u>();

  if (static_cast<int>(command.nu) == nu_) {
    // We sometimes need to wait to get the right data on the channel
    for (int i = 0; i < nu_; ++i) {
      output->SetAtIndex(i, command.u[i]);
    }
  } else {
    // If we're not getting any data, just set control torques to zero
    for (int i = 0; i < nu_; ++i) {
      output->SetAtIndex(i, 0.0);
    }
  }
}

TrajOptLcmController::TrajOptLcmController(const Diagram<double>* diagram,
                                           const MultibodyPlant<double>* plant,
                                           const ProblemDefinition& prob,
                                           const std::vector<VectorXd>& q_guess,
                                           const SolverParameters& params)
    : optimizer_(diagram, plant, prob, params),
      q_guess_(q_guess),
      B_(plant->MakeActuationMatrix()),
      nq_(plant->num_positions()),
      nv_(plant->num_velocities()),
      nu_(plant->num_actuators()) {
  DRAKE_DEMAND(static_cast<int>(q_guess.size()) == prob.num_steps + 1);
  DRAKE_DEMAND(static_cast<int>(q_guess[0].size()) == nq_);

  this->DeclareAbstractInputPort("lcmt_traj_opt_x", Value<lcmt_traj_opt_x>());
  this->DeclareAbstractOutputPort("lcmt_traj_opt_u",
                                  &TrajOptLcmController::OutputCommand);
}

void TrajOptLcmController::OutputCommand(const Context<double>& context,
                                         lcmt_traj_opt_u* output) const {
  const AbstractValue* abstract_state = this->EvalAbstractInput(context, 0);
  const auto& state = abstract_state->get_value<lcmt_traj_opt_x>();

  // Set header data
  output->utime = static_cast<int64_t>(context.get_time() * 1e6);
  output->nu = static_cast<int16_t>(nu_);
  output->nq = static_cast<int16_t>(nq_);
  output->nv = static_cast<int16_t>(nv_);
  output->u.resize(nu_);
  output->q.resize(nq_);
  output->v.resize(nv_);

  // Sometimes we need to wait a bit to get non-zero inputs
  if (state.nq > 0) {
    DRAKE_DEMAND(state.nq == nq_);
    DRAKE_DEMAND(state.nv == nv_);

    // Set initial state from the input port (state estimate)
    VectorXd q0(nq_);
    VectorXd v0(nv_);
    for (int i = 0; i < nq_; ++i) {
      q0(i) = state.q[i];
    }
    for (int i = 0; i < nv_; ++i) {
      v0(i) = state.v[i];
    }
    optimizer_.ResetInitialConditions(q0, v0);

    // Make sure the guess is consistent with the initial conditions
    q_guess_[0] = q0;

    // Solve trajectory optimization
    TrajectoryOptimizerStats<double> stats;
    TrajectoryOptimizerSolution<double> solution;
    optimizer_.Solve(q_guess_, &solution, &stats);

    // Set the control input
    const VectorXd u = B_.transpose() * solution.tau[0];
    for (int i = 0; i < nu_; ++i) {
      output->u[i] = u[i];
    }

    // Send target positions and velocities for the next time step. These can be
    // used in a PD+ controller in conjunction with u.
    for (int i = 0; i < nq_; ++i) {
      output->q[i] = solution.q[1][i];
    }
    for (int i = 0; i < nv_; ++i) {
      output->v[i] = solution.v[1][i];
    }

    // Save the solution to warm-start the next iteration
    // TODO(vincekurtz): consider shifting the guess in time
    q_guess_ = solution.q;
  }
}

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
