// Simple Discrete Time System Example
//
// This is meant to be a sort of "hello world" example for the
// drake::system classes.  It defines a very simple discrete time system,
// simulates it from a given initial condition, and checks the result.

#include <iostream>
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace {

// Simple Discrete Time System
//   x_{n+1} = x_n³
//         y = x
class SimpleDiscreteTimeSystem : public LeafSystem<double> {
 public:
  SimpleDiscreteTimeSystem() {
    //DeclarePeriodicDiscreteUpdateEvent(1.0, 0.0,
    //                                   &SimpleDiscreteTimeSystem::Update);
    DeclarePeriodicDiscreteUpdate(1.0, 0.0);
    auto state_index = DeclareDiscreteState(1);  // One state variable.
    DeclareStateOutputPort("y", state_index);
  }

 private:
  // x_{n+1} = x_n³
  void DoCalcDiscreteVariableUpdates(
      const Context<double>& context,
      const std::vector< const DiscreteUpdateEvent<double>*>&,
      DiscreteValues<double>* next_state) const override {
    const double x_n = context.get_discrete_state()[0];
    std::cout << "in update" << std::endl;  // debug
    (*next_state)[0] = std::pow(x_n, 3.0);
  }
};

int main() {
  // Create the simple system.
  SimpleDiscreteTimeSystem system;

  // Set the initial conditions x₀.
  std::unique_ptr<Context<double>> context = system.CreateDefaultContext();
  context->get_mutable_discrete_state()[0] = 0.99;

  // Here I want to compute the next state x_{n+1}.
  std::cout << "Initial state: " << context->get_discrete_state().get_value() << std::endl;
  std::unique_ptr<DiscreteValues<double>> next_state =
      system.AllocateDiscreteVariables();
  system.CalcDiscreteVariableUpdates(*context, next_state.get());
  std::cout << "Next state: " << next_state->get_value() << std::endl;

  return 0;
}

}  // namespace
}  // namespace systems
}  // namespace drake

int main() {
  return drake::systems::main();
}
