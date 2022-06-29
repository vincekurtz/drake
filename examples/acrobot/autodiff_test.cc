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
std::tuple<double, VectorX<double>, MatrixX<double>> run_test(
    const bool fancy_gradients, const int num_steps) {
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
  x0_val << 0.9, 1.1, 0.1, -0.2;
  const VectorX<AutoDiffXd> x0 = math::InitializeAutoDiff(x0_val);
  context->SetDiscreteState(x0);

  // Simulate forward a number of timesteps
  st = std::chrono::high_resolution_clock::now();
  std::unique_ptr<systems::DiscreteValues<AutoDiffXd>> state =
      acrobot.AllocateDiscreteVariables();
  VectorX<AutoDiffXd> x;

  for ( int i=0; i<num_steps; ++i ) {
    acrobot.CalcDiscreteVariableUpdates(*context, state.get());
    x = state->value();
    context->SetDiscreteState(x);
  }

  elapsed = std::chrono::high_resolution_clock::now() - st;

  return {elapsed.count(), math::ExtractValue(x), math::ExtractGradient(x)};
}

// Simple example of computing dynamics gradients with autodiff
int do_main() { 
  // Number of timesteps to simulate
  const int num_steps = 10000;

  // Compute gradients both ways
  auto [normal_time, normal_val, normal_grad] = run_test(false, num_steps);
  auto [fancy_time, fancy_val, fancy_grad] = run_test(true, num_steps);
  std::cout << "Normal time : " << normal_time << " s" << std::endl;
  std::cout << "Fancy time  : " << fancy_time << " s" << std::endl;

  // Sanity checks
  const VectorX<double> val_diff = normal_val - fancy_val;
  const MatrixX<double> grad_diff = normal_grad - fancy_grad;

  std::cout << fmt::format("Value error: {}\n", val_diff.norm());
  std::cout << fmt::format("Gradient error: {}\n", grad_diff.norm());

  return 0; 
}

}  // namespace
}  // namespace acrobot
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::acrobot::do_main(); }