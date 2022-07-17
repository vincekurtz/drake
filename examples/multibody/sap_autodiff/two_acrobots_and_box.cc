#include <iostream>
#include <gflags/gflags.h>

#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/examples/multibody/sap_autodiff/test_scenario.h"

DEFINE_bool(constraint, true,
            "Whether the initial state is such that constraints are active.");
DEFINE_bool(simulate, true,
            "Whether to run a quick simulation of the scenario.");
DEFINE_double(realtime_rate, 1.0, "Realtime rate for simulation.");
DEFINE_double(simulation_time, 2.0, "The time, in seconds, to simulate for.");
DEFINE_bool(test_autodiff, true, "Whether to run some autodiff tests.");
DEFINE_string(algebra, "both",
              "Type of algebra to use for testing autodiff. Options are: "
              "'sparse', 'dense', or 'both'.");
DEFINE_int32(num_steps, 1,
             "Number of timesteps to simulate for testing autodiff.");
DEFINE_double(time_step, 1e-2, "Size of the discrete timestep, in seconds");

namespace drake {
    
using multibody::MultibodyPlant;
using multibody::Parser;

namespace examples {
namespace multibody {
namespace sap_autodiff {

class AcrobotsAndBoxScenaro final : public SapAutodiffTestScenario {
 public:
  AcrobotsAndBoxScenaro() {
    x0_ << 0.9, 1.1,         // first acrobot position
           0.9, 1.1,            // second acrobot position
           0.985, 0, 0.174, 0,  // box orientation
           -1.5, 0.25, 2.051,   // box position
           0.5, 0.5,            // first acrobot velocity
           0.5, 0.5,            // second acrobot velocity
           0.0, 0.0, 0.0,       // box angular velocity
           0.1, 0.1, 0.2;       // box linear velocity
  }

 private:
  VectorX<double> x0_ = VectorX<double>::Zero(4+4+13);

  void CreateDoublePlant(MultibodyPlant<double>* plant) const override {
    // Load the models of acrobots and box from an sdf file
    const std::string acrobot_file = FindResourceOrThrow(
        "drake/examples/multibody/sap_autodiff/two_acrobots_and_box.sdf");
    Parser(plant).AddAllModelsFromFile(acrobot_file);
    plant->Finalize();
  }

  VectorX<double> get_x0_constrained() override { return x0_; }
  VectorX<double> get_x0_unconstrained() override {
    // Move the box over so it doesn't contact the acrobot
    x0_(9) -= 1; 
    return x0_; 
  }
};

} // namespace sap_autodiff
} // namespace multibody
} // namespace examples
} // namespace drake


int main(int argc, char* argv[]) {
  using drake::examples::multibody::sap_autodiff::AcrobotsAndBoxScenaro;
  using drake::examples::multibody::sap_autodiff::SapAutodiffTestParameters;
  using drake::examples::multibody::sap_autodiff::kAlgebraType;

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  SapAutodiffTestParameters params;
  params.constraint = FLAGS_constraint;
  params.simulate = FLAGS_simulate;
  params.realtime_rate = FLAGS_realtime_rate;
  params.simulation_time = FLAGS_simulation_time;
  params.test_autodiff = FLAGS_test_autodiff;
  params.num_steps = FLAGS_num_steps;
  params.time_step = FLAGS_time_step;

  if (FLAGS_algebra == "dense") {
    params.algebra = kAlgebraType::Dense;
  } else if (FLAGS_algebra == "sparse") {
    params.algebra = kAlgebraType::Sparse;
  } else {
    params.algebra = kAlgebraType::Both;
  }

  AcrobotsAndBoxScenaro scenario;
  scenario.RunTests(params);

  return 0;
}