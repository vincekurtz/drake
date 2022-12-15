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

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake