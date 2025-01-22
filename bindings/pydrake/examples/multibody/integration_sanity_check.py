import matplotlib.pyplot as plt
import numpy as np

import time
from pydrake.geometry import SceneGraphConfig, StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator, SimulatorConfig, ApplySimulatorConfig, PrintSimulatorStatistics
from pydrake.systems.primitives import LogVectorOutput
from pydrake.systems.framework import DiagramBuilder, TriggerType
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

##
#
# Run some simple sanity checks on the possibility of error-controlled
# integration with kLagged
#
##

meshcat = StartMeshcat()

xml = """
<?xml version="1.0"?>
<mujoco model="robot">
  <worldbody>
    <geom name="table_top" type="box" pos="0.0 0.0 0.0" size="0.55 1.1 0.05" rgba="0.9 0.8 0.7 1"/>
    <body>
        <joint type="free"/>
        <geom name="object" type="sphere" pos="0.0 0.0 0.5" euler="80 0 0" size="0.1" rgba="1.0 1.0 1.0 1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

def create_scene(sim_time_step, visualize=False):
    # Clean up the Meshcat instance.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Load the table top and the cylinder we created.
    parser.AddModelsFromString(xml, "xml")

    # Finalize the plant after loading the scene.
    plant.Finalize()
    
    # Add visualization to see the geometries.
    # NOTE: using the visualization messes with the published step sizes
    if visualize:
        AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a logger (mainly to keep track of timesteps)
    logger = LogVectorOutput(plant.get_state_output_port(), builder, publish_triggers={TriggerType.kForced}, publish_period = 0)
    logger.set_name("logger")

    diagram = builder.Build()
    return diagram, logger

def run_simulation():
    # Set MbP timestep (> 0 uses discrete solver)
    time_step = 0.0
    visualize = True
    
    # Set integrator parameters
    config = SimulatorConfig()
    config.integration_scheme = "implicit_euler"
    config.max_step_size=0.1
    config.accuracy=0.1
    config.target_realtime_rate = 0.0
    config.use_error_control = True
    config.publish_every_time_step = True

    # Use hydroelastic contact
    diagram, logger = create_scene(time_step, visualize)
    # scene_graph = diagram.GetSubsystemByName("scene_graph")
    # sg_config = SceneGraphConfig()
    # sg_config.default_proximity_properties.compliance_type = "compliant"
    # scene_graph.set_config(sg_config)

    # Check which contact solver and model we're using
    plant = diagram.GetSubsystemByName("plant")
    approximation = plant.get_discrete_contact_approximation().name
    print("")
    print(f"Discrete contact approximation: {approximation}")
    print(f"Discrete contact solver: {plant.get_discrete_contact_solver()}")
    approx = plant.get_discrete_contact_approximation()

    # Set up the simulation
    simulator = Simulator(diagram)
    ApplySimulatorConfig(config, simulator)
    simulator.Initialize()

    # Run the sim
    meshcat.StartRecording()
    st = time.time()
    simulator.AdvanceTo(1.0)
    wall_time = time.time() - st
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print(f"\nWall clock time: {wall_time:.4f} seconds\n")
    PrintSimulatorStatistics(simulator)

    # Try to (roughly) plot the integration time steps
    log = logger.FindLog(simulator.get_context())
    timesteps = log.sample_times()
    dts = timesteps[1:] - timesteps[0:-1]
    plt.plot(timesteps[0:-1], dts, "o")
    plt.yscale("log")
    plt.ylim((1e-8, 1e-0))
    if time_step == 0:
        plt.title(f"{config.integration_scheme}, accuracy = {config.accuracy}")
    else:
        plt.title(f"{approximation}, dt={time_step}")
    plt.show()

run_simulation()
