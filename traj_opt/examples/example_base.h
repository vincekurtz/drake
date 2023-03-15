#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/examples/yaml_config.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {

using geometry::DrakeVisualizerd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;

/**
 * Abstract base class for trajectory optimization examples.
 */
class TrajOptExample {
 public:
  virtual ~TrajOptExample() = default;

  /**
   * Run the example, either solving a single trajectory optimization problem or
   * running MPC, as specified by the options in the YAML file.
   *
   * @param options_file YAML file containing cost function definition, solver
   * parameters, etc., with fields as defined in yaml_config.h.
   */
  void RunExample(const std::string options_file) const;

  /**
   * Solve the optimization problem, as defined by the parameters in the given
   * YAML file.
   *
   * @param options YAML options, incluidng cost function definition, solver
   * parameters, etc.
   * @return TrajectoryOptimizerSolution<double> the optimal trajectory
   */
  TrajectoryOptimizerSolution<double> SolveTrajectoryOptimization(
      const TrajOptExampleParams& options) const;

  /**
   * Use the optimizer as an MPC controller in simulation. The simulator reads
   * control inputs sent over LCM, while in another thread, the controller reads
   * system state from the controller and computes optimal control inputs,
   * terminating the optimization early after a fixed number of iterations.
   *
   * @param options YAML options, incluidng cost function definition, solver
   * parameters, etc.
   */
  void RunModelPredictiveControl(const TrajOptExampleParams& options) const;

 private:
  /**
   * Create a MultibodyPlant model of the system that we're optimizing over.
   * This is the only method that needs to be overwritten to specialize to
   * different systems.
   *
   * @param plant the MultibodyPlant that we'll add the system to.
   */
  virtual void CreatePlantModel(MultibodyPlant<double>*) const {}

  /**
   * Play back the given trajectory on the Drake visualizer
   *
   * @param q sequence of generalized positions defining the trajectory
   * @param options parameters loaded from yaml
   */
  void PlayBackTrajectory(const std::vector<VectorXd>& q,
                          const TrajOptExampleParams& options) const;

  /**
   * Set an optimization problem from example options which were loaded from
   * YAML.
   *
   * @param options parameters loaded from yaml
   * @param opt_prob the problem definition (cost, initital state, etc)
   */
  void SetProblemDefinition(const TrajOptExampleParams& options,
                            ProblemDefinition* opt_prob) const;

  /**
   * Set solver parameters (used to pass options to the optimizer)
   * from example options (loaded from a YAML file).
   *
   * @param options parameters loaded from yaml
   * @param solver_params parameters for the optimizer that we'll set
   */
  void SetSolverParameters(const TrajOptExampleParams& options,
                           SolverParameters* solver_params) const;

  /**
   * Return a vector that interpolates linearly between q_start and q_end.
   * Useful for setting initial guesses and target trajectories.
   *
   * @param start initial vector
   * @param end final vector
   * @param N number of elements in the linear interpolation
   * @return std::vector<VectorXd> vector that interpolates between start and
   * end.
   */
  std::vector<VectorXd> MakeLinearInterpolation(const VectorXd& start,
                                                const VectorXd& end,
                                                int N) const {
    std::vector<VectorXd> result;
    double lambda = 0;
    for (int i = 0; i < N; ++i) {
      lambda = i / (N - 1.0);
      result.push_back((1 - lambda) * start + lambda * end);
    }
    return result;
  }

  /**
   * Simulate the system from the given initial condition, reading control
   * inputs and publishing (ground truth) state information over LCM.
   *
   * @param options parameters, read from a YAML file, defining the initial
   * condition, simulation time, realtime rate, etc.
   */
  void SimulateWithControlFromLcm(const TrajOptExampleParams& options) const;

  /**
   * Use MPC to control the system, reading state measurements from LCM and
   * sending control inputs back over LCM.
   *
   * @param options parameters, read from a YAML file, defining the cost
   * function, MPC parameters, etc.
   * @param q_guess initial guess to warm-start the first MPC iteration
   */
  void ControlWithStateFromLcm(const TrajOptExampleParams& options,
                               const std::vector<VectorXd>& q_guess) const;
};

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
