#include "drake/common/autodiff.h"

#include <chrono>
#include <iostream>

#include "drake/common/eigen_types.h"
#include "drake/examples/acrobot/acrobot_geometry.h"
#include "drake/examples/acrobot/acrobot_plant.h"
#include "drake/examples/acrobot/gen/acrobot_state.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/systems/analysis/simulator.h"

namespace drake {
namespace examples {
namespace acrobot {
namespace {

/**
 * Compute a single discrete-time step for the acrobot using autodiff
 * to calculate the gradients with respect to the initial state.
 *
 * Prints the value of the next state and gradients.
 *
 * Returns the compute time (in seconds).
 */
double run_test(bool fancy_gradients) {
  // Define the simulation timestep
  const double dt = 1e-2;

  // Set up a timer
  auto st = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed;

  // Create the autodiff plant
  AcrobotPlant<AutoDiffXd> acrobot(dt, fancy_gradients);
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
  std::unique_ptr<systems::DiscreteValues<AutoDiffXd>> state =
      acrobot.AllocateDiscreteVariables();
  acrobot.CalcDiscreteVariableUpdates(*context, state.get());
  VectorX<AutoDiffXd> x = state->value();
  elapsed = std::chrono::high_resolution_clock::now() - st;

  std::cout << math::ExtractValue(x) << std::endl;
  std::cout << math::ExtractGradient(x) << std::endl;

  return elapsed.count();
}

// Simple example of computing dynamics gradients with autodiff
int do_main() { 
  std::cout << "Normal Autodiff:" << std::endl;
  double normal_autodiff_time = run_test(false);
  std::cout << "Compute time: " << normal_autodiff_time << " s" << std::endl;

  std::cout << std::endl;
  std::cout << "Fancy Autodiff:" << std::endl;
  double fancy_autodiff_time = run_test(true);
  std::cout << "Compute time: " << fancy_autodiff_time << " s" << std::endl;

  return 0; 
}

}  // namespace
}  // namespace acrobot
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::acrobot::do_main(); }