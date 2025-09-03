import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Simulator,
    PrintSimulatorStatistics,
    SimulatorConfig,
    ApplySimulatorConfig,
    StartMeshcat,
    Parser,
    ApplyVisualizationConfig,
    VisualizationConfig,
)

##
#
# Newton's cradle demo to test energy conservation.
#
##

# TODO: parse command line flags

# Set up the simulation system diagram
meshcat = StartMeshcat()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
Parser(plant).AddModels(url="package://drake/examples/integrators/newtons_cradle.xml")
plant.Finalize()
vis_config = VisualizationConfig()
vis_config.publish_period = np.inf
ApplyVisualizationConfig(vis_config, builder=builder, meshcat=meshcat)
diagram = builder.Build()

# Set up the simulator
simulator = Simulator(diagram)
config = SimulatorConfig()
config.integration_scheme = "convex"
config.max_step_size = 1e-3
config.use_error_control = False
config.publish_every_time_step = True
ApplySimulatorConfig(config, simulator)

if config.integration_scheme == "convex":
    ci = simulator.get_mutable_integrator()
    ci.set_plant(plant)
    ci_params = ci.get_solver_parameters()
    ci_params.error_estimation_strategy = "half_stepping"
    ci.set_solver_parameters(ci_params)

# Set initial conditions
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)
q0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
plant.SetPositions(plant_context, q0)
simulator.Initialize()

input("Waiting for meshcat, press [ENTER] to continue...")
meshcat.StartRecording()
simulator.AdvanceTo(10.0)
meshcat.StopRecording()
meshcat.PublishRecording()

PrintSimulatorStatistics(simulator)

input("Waiting for meshcat, press [ENTER] to quit.")