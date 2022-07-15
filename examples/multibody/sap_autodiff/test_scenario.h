#pragma once

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {

using multibody::MultibodyPlant;
using systems::DiagramBuilder;
using geometry::SceneGraph;

namespace examples {
namespace multibody {
namespace sap_autodiff {

/**
 * Type of algebra to use for testing autodiff. Dense is our baseline,
 * which uses autodiff all the way through. Sparse is our new fancy method.
 * Both is both, obviously.
 */
enum kAlgebraType { Dense, Sparse, Both };

/** 
 * A simple set of parameters for running autodiff tests.
 * 
 * The basic idea is that these will ultimately be set from gflags and used
 * to define what happens in the main loop of a SapAutodiffTestScenario.
 */
struct SapAutodiffTestParameters {
  // Whether to use the initial condition where constraints (could be contact or
  // joint limits) are active
  bool constraint = true;

  // Whether to run a simulation, connected to DrakeVisualizer.
  bool simulate = false;

  // Realtime rate for simulation 
  double realtime_rate = 1.0;

  // Time, in seconds, to simulate for.
  double simulation_time = 2.0;

  // Whether to run a test of the autodiff computations
  bool test_autodiff = true;

  // Which sort of algebra to use for the autodiff test
  kAlgebraType algebra = kAlgebraType::Both;

  // Number of steps to take in the autodiff test
  int num_steps = 1;

  // Discrete timestep to use for both simulation and autodiff tests.
  double time_step = 1e-2;

};

/**
 * An abstract class for testing autodiff through SAP for various systems.
 *
 * The basic idea is that we want to be able to
 *      1) Simulate the system,
 *      2) Compute gradients (w.r.t. initial conditions) using autodiff
 *         exclusively, through dense algebra,
 *      3) Compute gradients with our fancy methods, using sparse algebra,
 *      4) Compare the gradients computed with both methods,
 * all using a unified interface, regardless of the actual system under
 * consideration.
 */
class SapAutodiffTestScenario {
 public:
  SapAutodiffTestScenario() = default;
  virtual ~SapAutodiffTestScenario() = default;

  /**
   * The main thing to do - tun whatever tests are indicated by the given
   * parameters.
   *
   * @param params parameters describing what exactly to do, usually loaded from
   * command line options.
   */
  void RunTests(const SapAutodiffTestParameters& params);


 private:
  // Create a model of the system in question.
  virtual void CreateDoublePlant(MultibodyPlant<double>* plant) const = 0;
  
  // Initial condition where constraints, either contact or joint limit, are
  // active
  virtual VectorX<double> get_x0_constrained() const = 0;

  // Initial condition where constraints, either contact or joint limit, are
  // notactive
  virtual VectorX<double> get_x0_unconstrained() const = 0;

  /**
   * Simulate several steps and use autodiff to compute gradients with respect
   * to the initial state. Return the final state and gradient matrix along
   * with the computation time.
   *
   * @param x0            Initial state of the system
   * @param num_steps     Number of timesteps to simulate
   * @param dense_algebra Whether to use dense algebra for sap
   * @return std::tuple<double, VectorX<double>, MatrixX<double>> tuple of
   * runtime in seconds, x, dx_dx0.
   */
  std::tuple<double, VectorX<double>, MatrixX<double>> TakeAutodiffSteps(
      const VectorX<double>& x0, const int num_steps,
      const bool dense_algebra) const;

  /**
   * Run a quick simulation of the system with zero input, connected to the
   * Drake visualizer. This is just to get a general sense of what is going on
   * in the example.
   *
   * @param x0            The initial state of the system
   */
  void SimulateWithVisualizer(const VectorX<double>& x0) const;

  // Set the SAP solver options to use dense algebra or not for the given plant.
  void SetSapSolverOptions(MultibodyPlant<double>* plant,
                           const bool dense_algebra) const;

  SapAutodiffTestParameters params_;
};

}  // namespace sap_autodiff
}  // namespace multibody
}  // namespace examples
}  // namespace drake
