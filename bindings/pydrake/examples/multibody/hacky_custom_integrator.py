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
# Super hacked-together "integrator" that takes discrete MbP steps of varying
# sizes.
#
##

meshcat = StartMeshcat()

# Define a test model inline here
xml = """
<?xml version="1.0"?>
<mujoco model="robot">
  <worldbody>
    <geom name="table_top" type="box" pos="0.0 0.0 0.0" size="0.55 1.1 0.05" rgba="0.9 0.8 0.7 1"/>
    <body>
        <joint type="free"/>
        <geom name="cylinder" type="cylinder" pos="0.0 0.0 3.5" euler="80 0 0" size="0.1 0.1" rgba="1.0 1.0 1.0 1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

def create_scene(sim_time_step, visualize=False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=sim_time_step)

    parser = Parser(plant)
    parser.AddModelsFromString(xml, "xml")
    plant.Finalize()

    # Use hydroelastic contact 
    sg_config = SceneGraphConfig()
    sg_config.default_proximity_properties.compliance_type = "compliant"
    scene_graph.set_config(sg_config)

    if visualize:
        # Connect to meshcat
        AddDefaultVisualization(builder=builder, meshcat=meshcat)
    
    diagram = builder.Build()
    return diagram, plant

def step(state, time_step):
    # Set up the (new) model
    diagram, plant = create_scene(time_step)

    # Set the initial state
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    plant.SetPositionsAndVelocities(plant_context, state)

    # Do a single timestep of integration with kLagged 
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(time_step)

    return plant.GetPositionsAndVelocities(plant_context)

def simulate():
    # Set initial conditions
    x = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -3.])
    t = 0.0

    # Set some sim parameters
    dt = 0.1
    sim_time = 1.0
    accuracy = 0.001
    max_dt = 0.1
    min_dt = 0.001

    # Constants from IntegratorBase::CalcAdjustedStepSice    
    kSafety = 0.9
    kMinShrink = 0.1
    kMaxGrow = 5.0
    kHysteresisLow = 0.9
    kHysteresisHigh = 1.2

    # Create a model for visualization
    diagram, plant = create_scene(0.0, visualize=True)
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)

    meshcat.StartRecording()
    timesteps = []
    while t < sim_time:
        # Record the current timestep
        timesteps.append(dt)
        
        # Step the simulation two ways: one with a full step, and one with
        # two half steps.
        x_full = step(x, dt)
        x_half = step(x, dt / 2)
        x_half = step(x_half, dt / 2)

        # We'll use the smaller steps to advance the sim
        # TODO: add conditions for rejecting the step
        x = x_half
        t += dt

        # Error is the difference between the one-step and two-step estimates
        # TODO: use something more like IntegratorBase::CalcStateChangeNorm
        # to compute the error norm
        err = x_full - x_half
        err = np.linalg.norm(err)
        
        # Set the next timestep based on this error. Logic roughly follows
        # IntegratorBase::CalcAdjustedStepSize
        if err == 0.0:
            new_dt = kMaxGrow * dt
        else:
            # N.B. assuming second order error estimate
            new_dt = kSafety * dt * (accuracy / err) ** (1.0 / 2.0)

        if (new_dt > dt) and (new_dt < kHysteresisHigh * dt):
            new_dt = dt

        if new_dt < dt:
            if err < accuracy:
                new_dt = dt
            else:
                new_dt = np.minimum(new_dt, kHysteresisLow * dt)

        max_grow_dt = kMaxGrow * dt
        min_shrink_dt = kMinShrink * dt
        new_dt = np.minimum(new_dt, max_grow_dt)
        new_dt = np.maximum(new_dt, min_shrink_dt)

        new_dt = np.minimum(new_dt, max_dt)
        new_dt = np.maximum(new_dt, min_dt)

        print(t)
        dt = new_dt

        # Update the visualizer
        context.SetTime(t)
        plant.SetPositionsAndVelocities(plant_context, x)
        diagram.ForcedPublish(context)

    meshcat.PublishRecording()

    return np.array(timesteps)

timesteps = simulate()

t = np.cumsum(timesteps)
plt.plot(t, timesteps, "o")
plt.xlabel("Sim time (s)")
plt.ylabel("dt (s)")

plt.yscale("log")
plt.ylim(1e-8, 1e0)

plt.title("Hacky custom integrator with kLagged")

plt.show()
