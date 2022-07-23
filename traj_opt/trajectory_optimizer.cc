#include <iostream>

#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {

using multibody::MultibodyPlant;
using systems::System;

TrajectoryOptimizer::TrajectoryOptimizer(std::unique_ptr<const MultibodyPlant<double>> plant,
                                         const ProblemDefinition& prob) : prob_(prob) {
  plant_ = std::move(plant);
  context_ = plant_->CreateDefaultContext();

  std::cout << "hello world" << std::endl;
  std::cout << plant_->num_velocities() << std::endl;
}


}  // namespace traj_opt
}  // namespace drake