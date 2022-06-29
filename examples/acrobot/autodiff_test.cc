#include <iostream>
#include <chrono>
#include "drake/examples/acrobot/acrobot_geometry.h"
#include "drake/examples/acrobot/acrobot_plant.h"
#include "drake/examples/acrobot/gen/acrobot_state.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/common/eigen_types.h"
#include "drake/common/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace examples {
namespace acrobot {
namespace {

// Simple example of computing dynamics gradients with autodiff
int do_main() {
  // Define the simulation timestep
  const double dt = 1e-2;

  // Set up some timers
  auto st = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_normal;
  //std::chrono::duration<double> elapsed_fancy;

  // Create the autodiff plant
  AcrobotPlant<AutoDiffXd> acrobot(dt);
  auto context = acrobot.CreateDefaultContext();
  
  // Fix the input
  const AutoDiffXd tau = 0;
  acrobot.GetInputPort("elbow_torque").FixValue(context.get(), tau);

  // Set the initial state
  VectorX<double> x0_val(4);
  x0_val << 1.0, 1.0, 0.0, 0.0;
  const VectorX<AutoDiffXd> x0 = math::InitializeAutoDiff(x0_val);
  context->SetDiscreteState(x0);

  // Simulate forward one timestep
  st = std::chrono::high_resolution_clock::now();
  std::unique_ptr<systems::DiscreteValues<AutoDiffXd>> state = acrobot.AllocateDiscreteVariables();
  acrobot.CalcDiscreteVariableUpdates(*context, state.get());
  VectorX<AutoDiffXd> x = state->value();
  elapsed_normal = std::chrono::high_resolution_clock::now() - st;

  std::cout << math::ExtractValue(x) << std::endl;
  std::cout << math::ExtractGradient(x) << std::endl;
  std::cout << elapsed_normal.count() << std::endl;

  return 0;
}

}  // namespace
}  // namespace acrobot
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::acrobot::do_main();
}