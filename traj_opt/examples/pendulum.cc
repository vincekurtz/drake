#include <iostream>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/solution_data.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace pendulum {

using geometry::DrakeVisualizerd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;
using systems::Simulator;

/**
 * Just run a simple passive simulation of the pendulum, connected to the Drake
 * visualizer.
 *
 * @param time_step Time step for discretization (seconds)
 * @param sim_time How long to simulate for (seconds)
 */
void run_passive_simulation(double time_step, double sim_time) {
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = time_step;
  config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const double u = 0;
  VectorX<double> x0(2);
  x0 << 0.5, 0.1;
  plant.get_actuation_input_port().FixValue(&plant_context, u);
  plant.SetPositionsAndVelocities(&plant_context, x0);

  Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(sim_time);
}

/**
 * Test our computation of generalized velocities
 *
 *   v_t = (q_t - q_{t-1})/dt
 *
 */
void test_v_from_q() {
  // Set up the system model
  // TODO(vincekurtz): set SAP as discrete contact solver
  auto plant = std::make_unique<MultibodyPlant<double>>(1e-2);
  const std::string urdf_file =
      FindResourceOrThrow("drake/traj_opt/examples/pendulum.urdf");
  Parser(plant.get()).AddAllModelsFromFile(urdf_file);
  plant->Finalize();
  auto plant_context = plant->CreateDefaultContext();

  // Simulate forward, recording the state
  int num_steps = 5;
  MatrixXd x(plant->num_multibody_states(), num_steps + 1);
  auto q = x.topRows<1>();
  auto v = x.bottomRows<1>();

  auto x0 = x.leftCols<1>();  // set initial state
  x0 << 1.3, 0.4;
  plant->SetPositionsAndVelocities(plant_context.get(), x.leftCols<1>());

  auto state = plant->AllocateDiscreteVariables();
  for (int i = 0; i < num_steps; ++i) {
    // Use some non-zero actuations to make things more interesting
    plant->get_actuation_input_port().FixValue(plant_context.get(),
                                               sin(plant->time_step() * i));

    plant->CalcDiscreteVariableUpdates(*plant_context, state.get());
    plant_context->SetDiscreteState(state->get_value());
    x.col(i + 1) = state->get_value();
  }

  // Create a trajectory optimizer
  ProblemDefinition opt_prob;
  opt_prob.q0 = q.col(0);
  opt_prob.v0 = v.col(0);
  opt_prob.T = num_steps;
  TrajectoryOptimizer optimizer(std::move(plant), opt_prob);

  // Trajectory optimizer computes v from q
  std::vector<VectorXd> q_vector;
  for (int i = 0; i <= num_steps; ++i) {
    q_vector.emplace_back(q.col(i));
  }

  std::vector<VectorXd> v_vector(num_steps + 1);
  optimizer.CalcV(q_vector, &v_vector);

  // Check that our computed v matches the recorded v
  MatrixXd v_computed(v.rows(), v.cols());
  for (int i = 0; i <= num_steps; ++i) {
    v_computed.col(i) = v_vector[i];
  }

  std::cout << v << std::endl;
  std::cout << v_computed << std::endl;
  std::cout << v - v_computed << std::endl;
}

int do_main() {
  // For now we'll just run a simple passive simulation of the pendulum
  // run_passive_simulation(1e-2, 2.0);

  // Test computation of v from q
  test_v_from_q();

  return 0;
}

}  // namespace pendulum
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() { return drake::traj_opt::examples::pendulum::do_main(); }
