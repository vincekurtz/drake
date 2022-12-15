#include "drake/traj_opt/examples/lcm_interfaces.h"

#include <iostream>

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

TrajOptLcmController::TrajOptLcmController(const int nq, const int nv, const int nu)
    : nq_(nq), nv_(nv), nu_(nu) {
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
  output->u.resize(nu_);

  // Sometimes we need to wait a bit to get non-zero inputs
  if (state.nq > 0) {
    DRAKE_DEMAND(state.nq == nq_);
    DRAKE_DEMAND(state.nv == nv_);
    
    // TODO: compute control w/ optimization

    // Set the control input
    output->u[0] = - state.v[1];

  }
}

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake