#include "drake/traj_opt/examples/lcm_interfaces.h"

namespace drake {
namespace traj_opt {
namespace examples {

StateSender::StateSender(const int nq, const int nv) : nq_(nq), nv_(nv) {
  this->DeclareInputPort("multibody_state", systems::kVectorValued, nq_ + nv_);
  //this->DeclareAbstractOutputPort("lcmt_traj_opt_x", &StateSender::OutputState);
}

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake