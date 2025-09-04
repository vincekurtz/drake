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
    SceneGraphConfig,
)

##
#
# Newton's cradle demo to test energy conservation.
#
##

# TODO: parse command line flags

# Set up the simulation system diagram
meshcat = StartMeshcat()
# meshcat.Set2dRenderMode()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
sg_config = SceneGraphConfig()
sg_config.default_proximity_properties.hunt_crossley_dissipation = 0.0
# sg_config.default_proximity_properties.compliance_type = "compliant"
sg_config.default_proximity_properties.point_stiffness = 1e3
sg_config.default_proximity_properties.hydroelastic_modulus = 1e5
sg_config.default_proximity_properties.static_friction = 0.0
sg_config.default_proximity_properties.dynamic_friction = 0.0
scene_graph.set_config(sg_config)
Parser(plant).AddModels(url="package://drake/examples/integrators/newtons_cradle.xml")
plant.Finalize()
vis_config = VisualizationConfig()
vis_config.publish_period = np.inf
vis_config.publish_proximity = False
vis_config.publish_contacts = False
ApplyVisualizationConfig(vis_config, builder=builder, meshcat=meshcat)
diagram = builder.Build()

# Set up the simulator
simulator = Simulator(diagram)
config = SimulatorConfig()
config.integration_scheme = "convex"
config.max_step_size = 1e-4
config.accuracy = 1e-5
config.use_error_control = False
config.publish_every_time_step = True
ApplySimulatorConfig(config, simulator)

if config.integration_scheme == "convex":
    ci = simulator.get_mutable_integrator()
    ci.set_plant(plant)
    ci_params = ci.get_solver_parameters()
    ci_params.error_estimation_strategy = "midpoint"
    ci.set_solver_parameters(ci_params)

# Set initial conditions
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)
q0 = np.array([0.5, -0.5])
plant.SetPositions(plant_context, q0)
simulator.Initialize()

e0 = plant.CalcPotentialEnergy(plant_context) + plant.CalcKineticEnergy(plant_context)

input("Waiting for meshcat, press [ENTER] to continue...")
meshcat.StartRecording()
simulator.AdvanceTo(5.0)
meshcat.StopRecording()
meshcat.PublishRecording()

PrintSimulatorStatistics(simulator)

print("")
ef = plant.CalcPotentialEnergy(plant_context) + plant.CalcKineticEnergy(plant_context)
print(f"Initial Energy: {e0}")
print(f"Final Energy: {ef}")
print(f"Energy Delta: {ef - e0} ({abs(ef - e0) / e0 * 100:.2f}%)")
print("")

input("Waiting for meshcat, press [ENTER] to quit.")