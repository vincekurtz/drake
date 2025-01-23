from pydrake.all import *

import matplotlib.pyplot as plt
import numpy as np

##
#
# Compare different integration schemes, including a hacked-together version 
# of our convex error-controlled integrator.
#
##

def ball_on_table():
    """XML description of a simple sphere on a tabletop."""
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
    return xml

def create_scene(
    xml: str, 
    time_step: float, 
    hydroelastic: bool = False,
    meshcat: Meshcat = None,
):
    """
    Set up a drake system dyagram

    Args:
        xml: mjcf robot description
        time_step: dt for MultibodyPlant
        hydroelastic: whether to use hydroelastic contact
        meshcat: meshcat instance for visualization. Defaults to no visualization.

    Returns:
        The system diagram, the MbP within that diagram, and the logger instance
        used to keep track of time steps
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step)

    parser = Parser(plant)
    parser.AddModelsFromString(xml, "xml")
    if time_step > 0:
        plant.set_discrete_contact_approximation(
            DiscreteContactApproximation.kLagged)
    plant.Finalize()

    if hydroelastic:
        sg_config = SceneGraphConfig()
        sg_config.default_proximity_properties.compliance_type = "compliant"
        scene_graph.set_config(sg_config)

    if meshcat is not None:
        AddDefaultVisualization(builder=builder, meshcat=meshcat)

    logger = LogVectorOutput(
        plant.get_state_output_port(), 
        builder, 
        publish_triggers={TriggerType.kForced}, 
        publish_period = 0)
    logger.set_name("logger")
    
    diagram = builder.Build()
    return diagram, plant, logger


def run_simulation(
    xml: str,
    use_hydroelastic: bool,
    initial_state: np.array,
    sim_time: float,
    integrator: str,
    accuracy: float,
    max_step_size: float,
    visualize: bool = False
):
    """
    Run a short simulation, and report the time-steps used throughout.

    Args:
        xml: string defining the model in mjcf format.
        use_hydroelastic: whether the model should use hydroelastic contact.
        initial_state: the state [q, v] to start the simulation from
        sim_time: the total simulation time (in seconds)
        integrator: which integration strategy to use ("implicit_euler", "runge_kutta3", "convex", "discrete")
        accuracy: the desired accuracy (ignored for "discrete")
        max_step_size: the maximum (and initial) timestep dt
        visualize: flag for playing the sim in meshcat. Note that this breaks
                   timestep visualizations

    Returns:
        Timesteps (dt) throughout the simulation.
    """
    if visualize:
        meshcat = StartMeshcat()
    else:
        meshcat = None

    if integrator == "convex":
        # Use our hacky custom integrator to step through the sim
        def step(state, time_step):
            """Take a single step with kLagged at a given timestep."""
            # Create a fresh system model
            diagram, plant, _ = create_scene(xml, time_step, use_hydroelastic)
            
            # Set the initial state
            context = diagram.CreateDefaultContext()
            plant_context = diagram.GetMutableSubsystemContext(plant, context)
            plant.SetPositionsAndVelocities(plant_context, state)

            # Do a single timestep of integration with kLagged 
            simulator = Simulator(diagram, context)
            simulator.AdvanceTo(time_step)

            return plant.GetPositionsAndVelocities(plant_context)

        # Hacky version of error-controlled convex integrator
        x = initial_state
        t = 0.0
        dt = max_step_size
        max_dt = max_step_size
        min_dt = 1e-10

        # Constants from IntegratorBase::CalcAdjustedStepSize    
        kSafety = 0.9
        kMinShrink = 0.1
        kMaxGrow = 5.0
        kHysteresisLow = 0.9
        kHysteresisHigh = 1.2

        # Create a model for visualization
        if meshcat is not None:
            diagram, plant, _ = create_scene(xml, 1.0, use_hydroelastic, meshcat)
            context = diagram.CreateDefaultContext()
            plant_context = diagram.GetMutableSubsystemContext(plant, context)
            diagram.ForcedPublish(context)
            input("Waiting for meshcat... [ENTER] to continue")
            meshcat.StartRecording()

        timesteps = []
        while t < sim_time:
            # Step the simulation two ways: one with a full step, and one with
            # two half steps.
            x_full = step(x, dt)
            x_half = step(x, dt / 2)
            x_half = step(x_half, dt / 2)

            # Error is the difference between the one-step and two-step
            # estimates.
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

            if err <= accuracy:
                # We'll use the smaller steps to advance the sim, and only
                # advance if it respects the desired accuracy
                x = x_half
                t += dt
            
                # Record the timestep only if we actually take the step
                timesteps.append(dt)

            max_grow_dt = kMaxGrow * dt
            min_shrink_dt = kMinShrink * dt
            new_dt = np.minimum(new_dt, max_grow_dt)
            new_dt = np.maximum(new_dt, min_shrink_dt)
            
            new_dt = np.minimum(new_dt, max_dt)
            new_dt = np.maximum(new_dt, min_dt)

            print(f"Time: {t:.4f} / {sim_time:.4f}")
            dt = new_dt

            # Update the visualizer
            if meshcat is not None:
                context.SetTime(t)
                plant.SetPositionsAndVelocities(plant_context, x)
                diagram.ForcedPublish(context)

        if meshcat is not None:
            meshcat.PublishRecording()

        return np.array(timesteps)

    else:
        # We can use a more standard simulation setup and rely on a logger to
        # tell use the time step information. Note that in this case enabling
        # visualization messes with the time step report though. 

        if integrator == "discrete":
            pass 
        else:
            # Configure Drake's built-in error controlled integration
            config = SimulatorConfig()
            config.integration_scheme = integrator
            config.max_step_size = max_step_size
            config.accuracy = accuracy
            config.target_realtime_rate = 1.0
            config.use_error_control = True
            config.publish_every_time_step = True

            # Set up the system diagram and initial condition
            diagram, plant, logger = create_scene(
                xml, 0.0, use_hydroelastic, meshcat)
            context = diagram.CreateDefaultContext()
            plant_context = diagram.GetMutableSubsystemContext(plant, context)
            plant.SetPositionsAndVelocities(plant_context, initial_state)

            simulator = Simulator(diagram, context)
            ApplySimulatorConfig(config, simulator)
            simulator.Initialize()
            input("Waiting for meshcat... [ENTER] to continue")

            # Simulate
            if meshcat is not None:
                meshcat.StartRecording()
            simulator.AdvanceTo(sim_time)
            if meshcat is not None:
                meshcat.PublishRecording()

            PrintSimulatorStatistics(simulator)

            # Get timesteps from the logger
            log = logger.FindLog(context)
            times = log.sample_times()
            timesteps = times[1:] - times[0:-1]
            return np.asarray(timesteps)


if __name__=="__main__":
    time_steps = run_simulation(
        xml = ball_on_table(),
        use_hydroelastic = False,
        initial_state = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
        sim_time = 1.0,
        integrator = "convex",
        accuracy = 0.1,
        max_step_size = 0.01,
        visualize = True
    )

    # Plot stuff
    times = np.cumsum(time_steps)
    plt.plot(times, time_steps, "o")
    plt.ylim(1e-10, 1e0)
    plt.yscale("log")
    plt.xlabel("time (s)")
    plt.ylabel("step size (s)")
    plt.show()