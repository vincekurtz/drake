import matplotlib.pyplot as plt
import numpy as np

from pydrake.geometry import SceneGraphConfig, StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator, SimulatorConfig, ApplySimulatorConfig
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

# Model is defined in-line here
cylinder_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="cylinder">
    <pose>0 0 0 0 0 0</pose>
    <link name="cylinder_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.005833</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.005833</iyy>
          <iyz>0.0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.2</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.2</length>
          </cylinder>
        </geometry>
        <material>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

table_top_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table_top">
    <link name="table_top_link">
      <visual name="visual">
        <pose>0 0 0.445 0 0 0</pose>
        <geometry>
          <box>
            <size>0.55 1.1 0.05</size>
          </box>
        </geometry>
        <material>
         <diffuse>0.9 0.8 0.7 1.0</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <pose>0 0 0.445  0 0 0</pose>
        <geometry>
          <box>
            <size>0.55 1.1 0.05</size>
          </box>
        </geometry>
      </collision>
    </link>
    <frame name="table_top_center">
      <pose relative_to="table_top_link">0 0 0.47 0 0 0</pose>
    </frame>
  </model>
</sdf>
"""

def create_scene(sim_time_step):
    # Clean up the Meshcat instance.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Load the table top and the cylinder we created.
    parser.AddModelsFromString(cylinder_sdf, "sdf")
    parser.AddModelsFromString(table_top_sdf, "sdf")

    # Weld the table to the world so that it's fixed during the simulation.
    table_frame = plant.GetFrameByName("table_top_center")
    plant.WeldFrames(plant.world_frame(), table_frame)

    # Finalize the plant after loading the scene.
    plant.Finalize()
    # We use the default context to calculate the transformation of the table
    # in world frame but this is NOT the context the Diagram consumes.
    plant_context = plant.CreateDefaultContext()

    # Set the initial pose for the free bodies, i.e., the custom cylinder,
    # the cracker box, and the sugar box.
    cylinder = plant.GetBodyByName("cylinder_link")
    X_WorldTable = table_frame.CalcPoseInWorld(plant_context)
    X_TableCylinder = RigidTransform(
        RollPitchYaw(np.asarray([80, 0, 0]) * np.pi / 180), p=[0,0,0.5])
    X_WorldCylinder = X_WorldTable.multiply(X_TableCylinder)
    plant.SetDefaultFreeBodyPose(cylinder, X_WorldCylinder)
    
    # Add visualization to see the geometries.
    # NOTE: using the visualization messes with the published step sizes
    # AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a logger (mainly to keep track of timesteps)
    logger = LogVectorOutput(plant.get_state_output_port(), builder, publish_triggers={TriggerType.kForced}, publish_period = 0)
    logger.set_name("logger")

    diagram = builder.Build()
    return diagram, logger

def run_simulation():
    # Set MbP timestep (> 0 uses discrete solver)
    time_step = 0.01
    
    # Set integrator parameters
    config = SimulatorConfig()
    config.integration_scheme = "runge_kutta3"
    config.max_step_size=0.1
    config.accuracy=0.1
    config.target_realtime_rate = 1.0
    config.use_error_control = True
    config.publish_every_time_step = True

    # Set up the simulation
    diagram, logger = create_scene(time_step)
    simulator = Simulator(diagram)
    ApplySimulatorConfig(config, simulator)
    simulator.Initialize()

    # Run the sim
    meshcat.StartRecording()
    simulator.AdvanceTo(1.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # Try to (roughly) plot the integration time steps
    log = logger.FindLog(simulator.get_context())
    timesteps = log.sample_times()
    dts = timesteps[1:] - timesteps[0:-1]
    plt.plot(timesteps[0:-1], dts, "o")
    plt.yscale("log")
    plt.ylim((1e-8, 1e-1))
    plt.show()


run_simulation()
