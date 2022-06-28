#include <iostream>
#include "drake/examples/acrobot/acrobot_geometry.h"
#include "drake/examples/acrobot/acrobot_plant.h"
#include "drake/examples/acrobot/gen/acrobot_state.h"
#include "drake/systems/analysis/simulator.h"

namespace drake {
namespace examples {
namespace acrobot {
namespace {

// Simple example of computing dynamics gradients with autodiff

int do_main() {
  // Define the simulation timestep
  const double dt = 1e-2;

  // Create the scalar version of the plant
  AcrobotPlant<double> acrobot(dt);
  auto context = acrobot.CreateDefaultContext();

  // Set the initial state and input values
  AcrobotState<double>& x0 = acrobot.get_mutable_state(context.get());
  x0.set_theta1(1.0);
  x0.set_theta2(1.0);
  x0.set_theta1dot(0.0);
  x0.set_theta2dot(0.0);

  const double tau = 0;
  acrobot.GetInputPort("elbow_torque").FixValue(context.get(), tau);

  // Simulate forward one timestep
  //systems::DiscreteValues<double>& x = context->get_mutable_discrete_state();
  std::unique_ptr<systems::DiscreteValues<double>> x = acrobot.AllocateDiscreteVariables();
  acrobot.CalcDiscreteVariableUpdates(*context, x.get());
  std::cout << acrobot.get_state(*x) << std::endl;

  systems::Simulator<double> simulator(acrobot, std::move(context));
  simulator.Initialize();
  simulator.AdvanceTo(dt);

  const auto& new_context = simulator.get_context();
  std::cout << acrobot.get_state(new_context) << std::endl;
  
  return 0;
}

}  // namespace
}  // namespace acrobot
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::acrobot::do_main();
}