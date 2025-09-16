import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Simulator,
    SimulatorConfig,
    ApplySimulatorConfig,
    Parser,
)

##
#
# Undamped double pendulum to test energy conservation.
#
##

double_pendulum_xml = """
<?xml version="1.0"?>
<mujoco model="robot">
  <worldbody>
    <body>
      <joint type="hinge" axis="0 1 0" pos="0 0 0.1" damping="0.0"/>
      <geom type="capsule" size="0.01 0.1"/>
      <body>
        <joint type="hinge" axis="0 1 0" pos="0 0 -0.1" damping="0.0"/>
        <geom type="capsule" size="0.01 0.1" pos="0 0 -0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def create_double_pendulum_sim(
    integration_scheme, error_estimation_strategy, time_step
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelsFromString(double_pendulum_xml, "xml")
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, [3.0, 0.1])

    simulator = Simulator(diagram, context)
    config = SimulatorConfig()
    config.integration_scheme = integration_scheme
    config.max_step_size = time_step
    config.use_error_control = False
    config.accuracy = 1e-3
    ApplySimulatorConfig(config, simulator)

    if config.integration_scheme == "convex":
        ci = simulator.get_mutable_integrator()
        ci.set_plant(plant)
        ci_params = ci.get_solver_parameters()
        ci_params.error_estimation_strategy = error_estimation_strategy
        ci.set_solver_parameters(ci_params)

    return simulator, plant


def run_simulation(integration_scheme, error_estimation_strategy, time_step):
    simulator, plant = create_double_pendulum_sim(
        integration_scheme, error_estimation_strategy, time_step
    )
    simulator.Initialize()

    # Get the initial system energy
    context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(context)
    initial_energy = plant.CalcPotentialEnergy(
        plant_context
    ) + plant.CalcKineticEnergy(plant_context)

    # Simulate with the given integration scheme
    simulator.AdvanceTo(2.0)

    # Get the final system energy
    final_energy = plant.CalcPotentialEnergy(
        plant_context
    ) + plant.CalcKineticEnergy(plant_context)

    print("  Initial Energy: ", initial_energy)
    print("  Final Energy:   ", final_energy)

    return abs(initial_energy - final_energy)


if __name__ == "__main__":
    # Run a bunch of simulations at different time steps, estimate integration
    # order.
    time_steps = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    errors = []
    for time_step in time_steps:
        print("Time step:", time_step)
        err = run_simulation("convex", "half_stepping", time_step)
        errors.append(err)

    plt.plot(time_steps, errors, "o-")
    plt.plot(time_steps, 1e2 * np.array(time_steps), "k--", label="O(h)")
    plt.plot(time_steps, 1e3 * np.array(time_steps) ** 2, "k:", label="O(h²)")
    plt.legend()

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time step [s]")
    plt.ylabel("Energy error [J]")
    plt.grid(True)
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.show()
