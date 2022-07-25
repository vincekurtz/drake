#include "drake/traj_opt/trajectory_optimizer.h"

#include <iostream>

namespace drake {
namespace traj_opt {

using multibody::MultibodyPlant;
using systems::System;

TrajectoryOptimizer::TrajectoryOptimizer(
    std::unique_ptr<const MultibodyPlant<double>> plant,
    const ProblemDefinition& prob)
    : prob_(prob) {
  plant_ = std::move(plant);
  context_ = plant_->CreateDefaultContext();
}

void TrajectoryOptimizer::CalcV(const std::vector<VectorXd>& q,
                                std::vector<VectorXd>* v) const {
  // x = [x0, x1, ..., xT]
  DRAKE_DEMAND(static_cast<int>(q.size()) == T() + 1);
  DRAKE_DEMAND(static_cast<int>(v->size()) == T() + 1);

  v->at(0) = prob_.v_init;
  for (int i = 1; i <= T(); ++i) {
    v->at(i) = (q[i] - q[i - 1]) / time_step();
  }
}

}  // namespace traj_opt
}  // namespace drake
