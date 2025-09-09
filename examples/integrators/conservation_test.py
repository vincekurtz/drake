import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    MultibodyPlant,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Simulator,
    PrintSimulatorStatistics,
    SimulatorConfig,
    ApplySimulatorConfig,
    Parser,
    AddDefaultVisualization,
    SceneGraphConfig,
    StartMeshcat,
)

undamped_double_pendulum_xml = """
<?xml version="1.0"?>
<mujoco model="double_pendulum">
<worldbody>
  <body>
  <joint type="free"/>
  <geom type="capsule" size="0.01 0.1"/>
  <body>
    <joint type="hinge" axis="0 1 0" pos="0 0 -0.1" damping="0.0"/>
    <geom type="capsule" size="0.01 0.1" pos="0 0 -0.2"/>
  </body>
  </body>
</worldbody>
</mujoco> 
"""

sphere_xml = """
<?xml version="1.0"?>
<mujoco model="sphere">
<worldbody>
  <body>
    <joint type="free"/>
    <geom type="sphere" size="0.03"/>
  </body>
</worldbody>
</mujoco>
"""

def run_conservation_test(time_step, integration_scheme, convex_scheme, visualize=False):
    """Simulate a double pendulum floating in space, hit by a sphere.

    Record energy and (linear, angular) momentum at the beginning and end of the
    simulation, and report conservation errors. This will help us to measure the
    order of various integration schemes. 
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    # Double pendulum model
    Parser(plant).AddModelsFromString(undamped_double_pendulum_xml, "xml")

    # Sphere model
    Parser(plant).AddModelsFromString(sphere_xml, "xml")

    # Turn off gravity
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant.Finalize()

    # Elastic collisions (zero dissipation)
    sg_config = SceneGraphConfig()
    sg_config.default_proximity_properties.hunt_crossley_dissipation = 0.0
    sg_config.default_proximity_properties.static_friction = 0.0
    sg_config.default_proximity_properties.dynamic_friction = 0.0
    sg_config.default_proximity_properties.point_stiffness = 1e3
    scene_graph.set_config(sg_config)

    # Optional visualization
    if visualize:
        meshcat = StartMeshcat()
        AddDefaultVisualization(builder=builder, meshcat=meshcat)

    diagram = builder.Build()

    # Set up the simulator
    simulator = Simulator(diagram)
    config = SimulatorConfig()
    config.integration_scheme = integration_scheme
    config.max_step_size = time_step
    config.use_error_control = False
    config.publish_every_time_step = True
    config.target_realtime_rate = 0.0
    ApplySimulatorConfig(config, simulator)

    if config.integration_scheme == "convex":
        ci = simulator.get_mutable_integrator()
        ci.set_plant(plant)
        ci_params = ci.get_solver_parameters()
        ci_params.error_estimation_strategy = convex_scheme
        ci.set_solver_parameters(ci_params)

    # Set initial conditions
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    q0 = plant.GetPositions(plant_context)
    q0[-3:] = [0.5, 0.0, 0.1]  # sphere position
    v0 = plant.GetVelocities(plant_context)
    v0[-3:] = [-1.0, 0.0, 0.0]  # sphere velocity
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
    simulator.Initialize()

    # Compute initial conserved quantities
    initial_energy = plant.CalcKineticEnergy(plant_context)
    p_CoM = plant.CalcCenterOfMassPositionInWorld(plant_context)
    initial_momentum = plant.CalcSpatialMomentumInWorldAboutPoint(
        plant_context, p_CoM
    )

    if visualize:
        input("waiting for meshcat, [ENTER] to continue...")
        meshcat.StartRecording()

    simulator.AdvanceTo(5.0)
    if visualize:
        meshcat.StopRecording()
        meshcat.PublishRecording()
        PrintSimulatorStatistics(simulator)

    final_energy = plant.CalcKineticEnergy(plant_context)
    final_momentum = plant.CalcSpatialMomentumInWorldAboutPoint(
        plant_context, p_CoM
    )

    energy_error = np.abs(final_energy - initial_energy)
    linear_momentum_error = np.linalg.norm(
        final_momentum.translational() - initial_momentum.translational()
    )
    angular_momentum_error = np.linalg.norm(
        final_momentum.rotational() - initial_momentum.rotational()
    )

    name = f"{integration_scheme}"
    if integration_scheme == "convex":
        name += f" ({convex_scheme})"

    print(f"Integration scheme: {name}")
    print(f"Time step: {time_step}")
    print(f"Energy error: {energy_error}")
    print(f"Linear momentum error: {linear_momentum_error}")
    print(f"Angular momentum error: {angular_momentum_error}")
    print("")

    return energy_error, linear_momentum_error, angular_momentum_error

if __name__ == "__main__":
    # Set options for what data to collect
    integration_scheme = "convex"
    convex_scheme = "half_stepping"
    time_steps = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

    # Collect data
    energy_errors = []
    linear_momentum_errors = []
    angular_momentum_errors = []

    for time_step in time_steps:
        e_err, p_err, L_err = run_conservation_test(
            time_step, integration_scheme, convex_scheme, visualize=False
        )
        energy_errors.append(e_err)
        linear_momentum_errors.append(p_err)
        angular_momentum_errors.append(L_err)

    # Make plots
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    ax[0].plot(time_steps, energy_errors, "o-", label="Energy error")
    ax[0].plot(time_steps, np.array(time_steps) ** 2, "k--", label="O(dt²)")
    ax[0].plot(time_steps, np.array(time_steps), "k:", label="O(dt)")
    ax[0].set_ylabel("Kinetic Energy Error")

    ax[1].plot(time_steps, linear_momentum_errors, "o-", label="Linear momentum error")
    ax[1].plot(time_steps, np.array(time_steps) ** 2, "k--", label="O(dt²)")
    ax[1].plot(time_steps, np.array(time_steps), "k:", label="O(dt)")
    ax[1].set_ylabel("Linear Momentum Error")

    ax[2].plot(time_steps, angular_momentum_errors, "o-", label="Angular momentum error")
    ax[2].plot(time_steps, np.array(time_steps) ** 2, "k--", label="O(dt²)")
    ax[2].plot(time_steps, np.array(time_steps), "k:", label="O(dt)")
    ax[2].set_ylabel("Angular Momentum Error")

    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.grid(True)
        a.legend()

    ax[2].invert_xaxis()
    ax[2].set_xlabel("Time step (s)")

    name = f"{integration_scheme}"
    if integration_scheme == "convex":
        name += f" ({convex_scheme})"
    ax[0].set_title(f"Integration Scheme: {name}")

    plt.tight_layout()
    plt.show()
