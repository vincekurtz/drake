#include "drake/traj_opt/examples/lcm_interfaces.h"

#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/primitives/linear_system.h"

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

LowLevelController::LowLevelController(const MultibodyPlant<double>* plant,
                                       const MultibodyPlant<double>* plant_lqr,
                                       const VectorXd Kp, const VectorXd Kd,
                                       const double Vmax, const bool lqr)
    : nu_(plant->num_actuators()),
      nq_(plant->num_positions()),
      nv_(plant->num_velocities()),
      B_(plant->MakeActuationMatrix()),
      Kp_(B_.transpose() * Kp.asDiagonal()),
      Kd_(B_.transpose() * Kd.asDiagonal()),
      Vmax_(Vmax),
      lqr_(lqr),
      Q_(1e0 * MatrixXd::Identity(nq_ + nv_, nq_ + nv_)),
      R_(1e-3 * MatrixXd::Identity(nu_, nu_)),
      plant_(plant),
      plant_lqr_(plant_lqr) {
  // Some size sanity checks
  DRAKE_DEMAND(Kp_.rows() == nu_);
  DRAKE_DEMAND(Kp_.cols() == nq_);
  DRAKE_DEMAND(Kd_.rows() == nu_);
  DRAKE_DEMAND(Kd_.cols() == nv_);

  // Define input and output ports
  control_port_ = this->DeclareAbstractInputPort("lcmt_traj_opt_u",
                                                 Value<lcmt_traj_opt_u>())
                      .get_index();
  state_estimate_port_ =
      this->DeclareVectorInputPort("state_estimate", nq_ + nv_).get_index();
  this->DeclareVectorOutputPort("control_torques", nu_,
                                &LowLevelController::OutputCommandAsVector);

  // Set up system context
  context_ = plant_->CreateDefaultContext();
  // Create context for the LQR plant
  context_lqr_ = plant_lqr_->CreateDefaultContext();
}

void LowLevelController::OutputCommandAsVector(
    const Context<double>& context,
    systems::BasicVector<double>* output) const {
  const AbstractValue* abstract_command =
      this->EvalAbstractInput(context, control_port_);
  const systems::BasicVector<double>* state_estimate =
      this->EvalVectorInput(context, state_estimate_port_);

  const auto& command = abstract_command->get_value<lcmt_traj_opt_u>();
  const VectorXd& x = state_estimate->value();
  DRAKE_DEMAND(x.size() == nq_ + nv_);

  // Define q, v, q_nom, v_nom, and u_nom
  const auto& q = x.topRows(nq_);
  const auto& v = x.bottomRows(nv_);
  Eigen::Map<const VectorXd> u_nom(command.u.data(), command.u.size());
  Eigen::Map<const VectorXd> q_nom(command.q.data(), command.q.size());
  Eigen::Map<const VectorXd> v_nom(command.v.data(), command.v.size());

  // We sometimes need to wait to get the right data on the channel
  if (static_cast<int>(command.nu) == nu_) {
    // Initialize the control input to the nominal values
    VectorXd u = u_nom;

    // Use the PD+ controller if the LQR is not enabled
    if (!lqr_) {
      // Set PD+ input to track the trajectory from the optimizer. This will be
      // applied if system energy is sufficiently low.
      u = u_nom - Kp_ * (q - q_nom) - Kd_ * (v - v_nom);

      // Compute current system energy
      plant_->SetPositionsAndVelocities(context_.get(), x);
      const double V = plant_->EvalKineticEnergy(*context_) +
                       plant_->EvalPotentialEnergy(*context_);

      // Apply a barrier function to bound the system energy
      // const double gamma = std::exp(15 * V / Vmax_ - 15);
      const double gamma = std::pow(V / Vmax_, 4);
      if ((0 <= gamma) && (gamma <= 1)) {
        u = (1 - gamma) * u - gamma * B_.transpose() * v;
      } else if (gamma > 1) {
        u = -gamma * B_.transpose() * v;
      }

      // Apply torque limits
      // TODO(vincekurtz): make this a yaml parameter
      // const double tau_max = 10;
      // u = u.cwiseMin(tau_max).cwiseMax(-tau_max);
    } else {
      // Setup the LQR plant and context
      plant_lqr_->SetPositionsAndVelocities(context_lqr_.get(), x);
      plant_lqr_->get_actuation_input_port().FixValue(context_lqr_.get(),
                                                      u_nom);
      // Linearize the forward dynamics about the current operating point
      // NOTE: The equilibrium tolerance is practically ignored using a very
      // large tolerance since we rarely operate at an equilibrium.
      auto linear_system =
          Linearize(*plant_lqr_, *context_lqr_,
                    plant_lqr_->get_actuation_input_port().get_index(),
                    systems::OutputPortSelection::kNoOutput, 1e20);
      // Solve the continuous Riccati equations to get the feedback law.
      auto lqr_result = systems::controllers::LinearQuadraticRegulator(
          linear_system->A(), linear_system->B(), Q_, R_);
      // Evaluate the LQR compensation.
      auto u_lqr = lqr_result.K.leftCols(nq_) * (q - q_nom) +
                   lqr_result.K.rightCols(nv_) * (v - v_nom);
      // Apply the LQR contribution to the control input.
      u = u_nom - u_lqr;
    }

    output->SetFromVector(u);

  } else {
    // If we're not getting any data, just set control torques to zero
    output->SetZero();
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
