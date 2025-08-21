#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <gflags/gflags.h>

#include "drake/common/cpu_timing_logger.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/problem_size_logger.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/convex_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/visualization/visualization_config.h"
#include "drake/visualization/visualization_config_functions.h"

// Parameters for squeezing the spatula.
DEFINE_double(gripper_force, 1,
              "The baseline force to be applied by the gripper. [N].");
DEFINE_double(amplitude, 5,
              "The amplitude of the oscillations "
              "carried out by the gripper. [N].");
DEFINE_double(duty_cycle, 0.5, "Duty cycle of the control signal.");
DEFINE_double(period, 3, "Period of the control signal. [s].");

// Scaling parameter for multiple grippers/spatulas.
DEFINE_int32(num_instances, 1,
             "Number of gripper-spatula instances to simulate. "
             "Instances will be arranged in a grid pattern.");

// MultibodyPlant settings.
DEFINE_double(stiction_tolerance, 1e-4, "Default stiction tolerance. [m/s].");
DEFINE_double(mbp_discrete_update_period, 4.0e-2,
              "If zero, the plant is modeled as a continuous system. "
              "If positive, the period (in seconds) of the discrete updates "
              "for the plant modeled as a discrete system."
              "This parameter must be non-negative.");
DEFINE_string(contact_model, "hydroelastic_with_fallback",
              "Contact model. Options are: 'point', 'hydroelastic', "
              "'hydroelastic_with_fallback'.");
DEFINE_string(contact_surface_representation, "polygon",
              "Contact-surface representation for hydroelastics. "
              "Options are: 'triangle' or 'polygon'.");
DEFINE_string(contact_approximation, "sap",
              "Discrete contact approximation. Options are: 'tamsi', "
              "'sap', 'similar', 'lagged'");
DEFINE_bool(
    use_sycl, false,
    "Use SYCL for hydroelastic contact. This flag is only used when "
    "the contact model is 'hydroelastic' or 'hydroelastic_with_fallback'.");

// Simulator settings.
DEFINE_double(realtime_rate, 0,
              "Desired rate of the simulation compared to realtime."
              "A value of 1 indicates real time.");
DEFINE_double(simulation_sec, 30, "Number of seconds to simulate. [s].");
// The following flags are only in effect in continuous mode.
DEFINE_double(accuracy, 1.0e-3, "The integration accuracy.");
DEFINE_double(max_time_step, 1.0e-2,
              "The maximum time step the integrator is allowed to take, [s].");
DEFINE_string(integration_scheme, "convex",
              "Integration scheme to be used. Available options are: "
              "'semi_explicit_euler','runge_kutta2','runge_kutta3', 'convex'"
              "'implicit_euler'");

DEFINE_int32(mesh_res, 5, "Mesh resolution hint for the spatula. In (mm)");

DEFINE_bool(print_perf, true, "Print performance statistics");
DEFINE_bool(visualize, false, "Visualize the simulation");
DEFINE_bool(use_error_control, false, "Use error control for the simulation.");

namespace drake {

using drake::systems::ConvexIntegrator;
using drake::systems::ConvexIntegratorSolverParameters;
using geometry::SceneGraph;
using math::RigidTransform;
using math::RollPitchYaw;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::PrismaticJoint;
using systems::ApplySimulatorConfig;
using systems::BasicVector;
using systems::Context;
using systems::SimulatorConfig;
namespace examples {
namespace spatula_slip_control {
namespace {
std::string GetBaseOutDir() {
  if (const char* ws = std::getenv("BUILD_WORKSPACE_DIRECTORY")) {
    return std::string(ws);
  }
  if (const char* wd = std::getenv("BUILD_WORKING_DIRECTORY")) {
    return std::string(wd);
  }
  return std::filesystem::current_path().string();
}
// We create a simple leaf system that outputs a square wave signal for our
// open loop controller. The Square system here supports an arbitrarily
// dimensional signal, but we will use a 2-dimensional signal for our gripper.
class Square final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Square);

  // Constructs a %Square system where different amplitudes, duty cycles,
  // periods, and phases can be applied to each square wave.
  //
  // @param[in] amplitudes the square wave amplitudes. (unitless)
  // @param[in] duty_cycles the square wave duty cycles.
  //                        (ratio of pulse duration to period of the waveform)
  // @param[in] periods the square wave periods. (seconds)
  // @param[in] phases the square wave phases. (radians)
  Square(const Eigen::VectorXd& amplitudes, const Eigen::VectorXd& duty_cycles,
         const Eigen::VectorXd& periods, const Eigen::VectorXd& phases)
      : amplitude_(amplitudes),
        duty_cycle_(duty_cycles),
        period_(periods),
        phase_(phases) {
    // Ensure the incoming vectors are all the same size.
    DRAKE_THROW_UNLESS(duty_cycles.size() == amplitudes.size());
    DRAKE_THROW_UNLESS(duty_cycles.size() == periods.size());
    DRAKE_THROW_UNLESS(duty_cycles.size() == phases.size());

    this->DeclareVectorOutputPort("Square Wave Output", duty_cycles.size(),
                                  &Square::CalcValueOutput);
  }

 private:
  void CalcValueOutput(const Context<double>& context,
                       BasicVector<double>* output) const {
    Eigen::VectorBlock<VectorX<double>> output_block =
        output->get_mutable_value();

    const double time = context.get_time();

    for (int i = 0; i < duty_cycle_.size(); ++i) {
      // Add phase offset.
      double t = time + (period_[i] * phase_[i] / (2 * M_PI));

      output_block[i] =
          amplitude_[i] *
          (t - floor(t / period_[i]) * period_[i] < duty_cycle_[i] * period_[i]
               ? 1
               : 0);
    }
  }

  const Eigen::VectorXd amplitude_;
  const Eigen::VectorXd duty_cycle_;
  const Eigen::VectorXd period_;
  const Eigen::VectorXd phase_;
};

void PrintPerformanceStats(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::geometry::SceneGraph<double>& scene_graph,
    const drake::systems::Context<double>& scene_graph_context, bool sycl_used,
    int mesh_res, int num_instances, double sim_time, std::string out_dir) {
  std::string demo_name = "spatula_slip_control_" + std::to_string(mesh_res) +
                          "_" + std::to_string(num_instances);
  std::string runtime_device;
  const char* env_var = std::getenv("ONEAPI_DEVICE_SELECTOR");
  if (env_var != nullptr) {
    runtime_device = env_var;
  }

  // Create the output directory if it doesn't exist
  if (!std::filesystem::exists(out_dir)) {
    std::filesystem::create_directories(out_dir);
  }
  std::string run_type;
  if (runtime_device.empty()) {
    run_type = sycl_used ? "sycl-gpu" : "drake-cpu";
  } else if (runtime_device == "cuda:*" || runtime_device == "cuda:gpu") {
    run_type = sycl_used ? "sycl-gpu" : "drake-cpu";
  } else {
    run_type = "sycl-cpu";
  }

  std::string json_path =
      out_dir + "/" + demo_name + "_" + run_type + "_problem_size.json";

  // Ensure output directory exists
  if (!std::filesystem::exists(out_dir)) {
    std::cerr << "Performance output directory does not exist: " << out_dir
              << std::endl;
    return;
  }

  fmt::print("Problem Size Stats:\n");
  const auto& inspector = scene_graph.model_inspector();
  int hydro_bodies = 0;
  std::ostringstream hydro_json;
  hydro_json << "\"hydroelastic_bodies\": [";
  bool first = true;
  for (int i = 0; i < plant.num_bodies(); ++i) {
    const auto& body = plant.get_body(drake::multibody::BodyIndex(i));
    bool has_hydro = false;
    int tet_count = 0;
    for (const auto& gid : plant.GetCollisionGeometriesForBody(body)) {
      const auto* props = inspector.GetProximityProperties(gid);
      if (props &&
          props->HasProperty(drake::geometry::internal::kHydroGroup,
                             drake::geometry::internal::kComplianceType)) {
        has_hydro = true;
      }
      auto mesh_variant = inspector.maybe_get_hydroelastic_mesh(gid);
      if (std::holds_alternative<const drake::geometry::VolumeMesh<double>*>(
              mesh_variant)) {
        const auto* mesh =
            std::get<const drake::geometry::VolumeMesh<double>*>(mesh_variant);
        if (mesh) tet_count += mesh->num_elements();
      }
    }
    if (has_hydro) ++hydro_bodies;
    if (tet_count > 0) {
      if (!first) hydro_json << ",";
      first = false;
      hydro_json << "{ \"body\": \"" << body.name()
                 << "\", \"tetrahedra\": " << tet_count << "}";
    }
  }
  hydro_json << "]";
  fmt::print("Number of bodies with hydroelastic contact: {}\n", hydro_bodies);
  for (int i = 0; i < plant.num_bodies(); ++i) {
    const auto& body = plant.get_body(drake::multibody::BodyIndex(i));
    int tet_count = 0;
    for (const auto& gid : plant.GetCollisionGeometriesForBody(body)) {
      auto mesh_variant = inspector.maybe_get_hydroelastic_mesh(gid);
      if (std::holds_alternative<const drake::geometry::VolumeMesh<double>*>(
              mesh_variant)) {
        const auto* mesh =
            std::get<const drake::geometry::VolumeMesh<double>*>(mesh_variant);
        if (mesh) tet_count += mesh->num_elements();
      }
    }
    if (tet_count > 0) {
      fmt::print("Body '{}' has {} tetrahedra in its hydroelastic mesh.\n",
                 body.name(), tet_count);
    }
  }
  drake::common::ProblemSizeLogger::GetInstance().PrintStats();
  drake::common::ProblemSizeLogger::GetInstance().PrintStatsJson(
      json_path, hydro_json.str());

  fmt::print("Timing Stats:\n");
  json_path =
      out_dir + "/" + demo_name + "_" + run_type + "_timing_overall.json";

  drake::common::CpuTimingLogger::GetInstance().PrintStats();
  drake::common::CpuTimingLogger::GetInstance().PrintStatsJson(json_path);
  json_path = out_dir + "/" + demo_name + "_" + run_type + "_timing.json";
  const auto& query_object =
      scene_graph.get_query_output_port().Eval<geometry::QueryObject<double>>(
          scene_graph_context);
  query_object.PrintSyclTimingStats();
  query_object.PrintSyclTimingStatsJson(json_path);

  fmt::print("AdvanceTo() time [sec]: {}\n", sim_time);
  json_path =
      out_dir + "/" + demo_name + "_" + run_type + "_timing_advance_to.json";
  // Write to simple json file
  std::ofstream ofs(json_path);
  if (ofs.is_open()) {
    ofs << "{\"advance_to_time\": " << sim_time << "}";
    ofs.close();
  } else {
    std::cerr << "Failed to open file for writing: " << json_path << std::endl;
  }
}

int DoMain() {
  std::string base_out_dir = GetBaseOutDir();
  // Add trailing slash if not present
  if (base_out_dir.back() != '/') {
    base_out_dir += '/';
  }

  if (FLAGS_print_perf || FLAGS_visualize) {
    fmt::print("Base output directory: {}\n", base_out_dir);
  }
  std::string timing_out_dir;
  std::string html_out_dir;
  if (FLAGS_integration_scheme == "convex") {
    if (FLAGS_use_error_control) {
      timing_out_dir = base_out_dir +
                       "performance_jsons_spatula_slip_control_scale_convex_"
                       "error_control";
      html_out_dir = base_out_dir + "spatula_convex_error_control_html/";
    } else {
      timing_out_dir =
          base_out_dir + "performance_jsons_spatula_slip_control_scale_convex/";
      html_out_dir = base_out_dir + "spatula_convex_html/";
    }
  } else {
    timing_out_dir =
        base_out_dir + "performance_jsons_spatula_slip_control_scale/";
    html_out_dir = base_out_dir + "spatula_html/";
  }
  if (FLAGS_print_perf) {
    fmt::print("Timing out dir: {}\n", timing_out_dir);
  }

  if (FLAGS_visualize) {
    fmt::print("HTML out dir: {}\n", html_out_dir);
  }

  // Construct a MultibodyPlant and a SceneGraph.
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  if (FLAGS_integration_scheme == "convex") {
    fmt::print("Setting time step to 0 for convex integrator\n");
    plant_config.time_step = 0;
  } else {
    plant_config.time_step = FLAGS_mbp_discrete_update_period;
  }
  plant_config.stiction_tolerance = FLAGS_stiction_tolerance;
  plant_config.contact_model = FLAGS_contact_model;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;
  plant_config.contact_surface_representation =
      FLAGS_contact_surface_representation;

  DRAKE_DEMAND(FLAGS_mbp_discrete_update_period >= 0);
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(plant_config, &builder);

  // Calculate grid dimensions for arranging multiple instances
  const int num_instances = FLAGS_num_instances;
  const int grid_size = static_cast<int>(std::ceil(std::sqrt(num_instances)));
  const double spacing = 0.5;  // Distance between instances in meters

  std::cout << "Creating " << num_instances
            << " gripper-spatula instances in a " << grid_size << "x"
            << grid_size << " grid" << std::endl;

  // Parse multiple gripper and spatula models
  multibody::Parser parser(&builder);

  // Enable auto-renaming to handle multiple instances of the same model
  parser.SetAutoRenaming(true);

  for (int i = 0; i < num_instances; ++i) {
    // Add gripper model
    parser.AddModelsFromUrl(
        "package://drake_models/wsg_50_description/sdf/"
        "schunk_wsg_50_hydro_bubble.sdf");

    // Add spatula model
    parser.AddModelsFromUrl(
        "package://drake/examples/hydroelastic/spatula_slip_control/"
        "spatula" +
        std::to_string(FLAGS_mesh_res) + ".sdf");
  }

  // Get all model instances to position them correctly
  std::vector<multibody::ModelInstanceIndex> gripper_instances;
  std::vector<multibody::ModelInstanceIndex> spatula_instances;

  // Find the model instances for grippers and spatulas
  for (int i = 0; i < plant.num_model_instances(); ++i) {
    const auto model_instance = multibody::ModelInstanceIndex(i);
    const std::string& model_name = plant.GetModelInstanceName(model_instance);

    if (model_name.find("Schunk_Gripper") == 0) {
      gripper_instances.push_back(model_instance);
    } else if (model_name.find("spatula") == 0) {
      spatula_instances.push_back(model_instance);
    }
  }

  // Sort them to ensure consistent ordering
  std::sort(gripper_instances.begin(), gripper_instances.end());
  std::sort(spatula_instances.begin(), spatula_instances.end());

  // Position all grippers in a grid pattern BEFORE finalizing
  for (int i = 0; i < num_instances; ++i) {
    const int row = i / grid_size;
    const int col = i % grid_size;

    // Calculate position for this instance
    const double x_offset = col * spacing;
    const double y_offset = row * spacing;

    // Position gripper
    const math::RigidTransform<double> X_WGripper =
        math::RigidTransform<double>(math::RollPitchYaw(0.0, -1.57, 0.0),
                                     Eigen::Vector3d(x_offset, y_offset, 0.25));
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("gripper", gripper_instances[i]),
                     X_WGripper);
  }

  plant.Finalize();

  // Create a single controller that outputs forces for all instances
  // Each gripper needs 2 actuation inputs (left and right finger)
  const int total_actuation_size = num_instances * 2;

  const double f0 = FLAGS_gripper_force;
  // Create amplitudes with alternating signs for left/right fingers
  Eigen::VectorXd amplitudes =
      Eigen::VectorXd::Constant(total_actuation_size, FLAGS_amplitude);
  for (int i = 0; i < num_instances; ++i) {
    amplitudes[2 * i + 1] =
        -FLAGS_amplitude;  // Right finger gets negative amplitude
  }

  const Eigen::VectorXd duty_cycles =
      Eigen::VectorXd::Constant(total_actuation_size, FLAGS_duty_cycle);
  const Eigen::VectorXd periods =
      Eigen::VectorXd::Constant(total_actuation_size, FLAGS_period);
  const Eigen::VectorXd phases = Eigen::VectorXd::Zero(total_actuation_size);

  const auto& square_force =
      *builder.AddSystem<Square>(amplitudes, duty_cycles, periods, phases);
  // Create constant forces with alternating signs for left/right fingers
  Eigen::VectorXd constant_forces =
      Eigen::VectorXd::Constant(total_actuation_size, f0);
  for (int i = 0; i < num_instances; ++i) {
    constant_forces[2 * i + 1] = -f0;  // Right finger gets negative force
  }
  const auto& constant_force_signed =
      *builder.AddSystem<systems::ConstantVectorSource<double>>(
          constant_forces);

  const auto& adder =
      *builder.AddSystem<systems::Adder<double>>(2, total_actuation_size);
  builder.Connect(square_force.get_output_port(), adder.get_input_port(0));
  builder.Connect(constant_force_signed.get_output_port(),
                  adder.get_input_port(1));

  // Connect the output of the adder to the plant's actuation input.
  builder.Connect(adder.get_output_port(0), plant.get_actuation_input_port());

  std::shared_ptr<geometry::Meshcat> meshcat;
  if (FLAGS_visualize) {
    meshcat = std::make_shared<geometry::Meshcat>();
    visualization::ApplyVisualizationConfig(
        visualization::VisualizationConfig{
            .default_proximity_color = geometry::Rgba{1, 0, 0, 0.25},
            .enable_alpha_sliders = true,
        },
        &builder, nullptr, nullptr, nullptr, meshcat);
  }

  // Construct a simulator.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();

  if (FLAGS_integration_scheme != "convex") {
    FLAGS_use_error_control = false;
  }

  SimulatorConfig sim_config;
  sim_config.integration_scheme = FLAGS_integration_scheme;
  if (FLAGS_integration_scheme == "convex") {
    if (!FLAGS_use_error_control) {
      fmt::print("Using fixed step size for convex integrator\n");
      sim_config.max_step_size = FLAGS_mbp_discrete_update_period;
    } else {
      fmt::print("Using error control for convex integrator\n");
      sim_config.max_step_size = FLAGS_max_time_step;
    }
  } else {
    sim_config.max_step_size = FLAGS_max_time_step;
  }
  sim_config.accuracy = FLAGS_accuracy;
  sim_config.target_realtime_rate = 0;
  sim_config.publish_every_time_step = false;
  if (FLAGS_use_error_control) {
    sim_config.use_error_control = true;
  } else {
    fmt::print("Setting use_error_control to false\n");
    sim_config.use_error_control = false;
  }

  systems::Simulator<double> simulator(*diagram);
  ApplySimulatorConfig(sim_config, &simulator);

  drake::systems::IntegratorBase<double>& integrator =
      simulator.get_mutable_integrator();
  if (FLAGS_integration_scheme == "convex") {
    auto& ci = dynamic_cast<ConvexIntegrator<double>&>(integrator);
    ci.set_plant(&plant);

    ConvexIntegratorSolverParameters ci_params;
    ci_params.enable_hessian_reuse = false;
    ci_params.max_iterations_for_hessian_reuse = 10;
    ci_params.kappa = 1e-3;
    ci_params.alpha_max = 1.0;
    ci_params.ls_tolerance = 1e-8;
    ci_params.tolerance = 1e-8;
    ci_params.log_solver_stats = false;
    ci_params.print_solver_stats = false;
    ci_params.use_dense_algebra = false;
    ci_params.error_estimation_strategy = "half_stepping";
    ci.set_solver_parameters(ci_params);
  }

  // Set the initial conditions for all spatula poses and gripper finger
  // positions
  Context<double>& mutable_root_context = simulator.get_mutable_context();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, &mutable_root_context);

  Context<double>& scene_graph_context =
      diagram->GetMutableSubsystemContext(scene_graph, &mutable_root_context);

  // Position spatulas and set initial joint positions
  for (int i = 0; i < num_instances; ++i) {
    const int row = i / grid_size;
    const int col = i % grid_size;

    // Calculate position for this instance
    const double x_offset = col * spacing;
    const double y_offset = row * spacing;

    // Position spatula relative to gripper
    const math::RigidTransform<double> X_WSpatula =
        math::RigidTransform<double>(
            math::RollPitchYaw(-0.4, 0.0, 1.57),
            Eigen::Vector3d(x_offset + 0.35, y_offset, 0.25));
    const auto& spatula_body =
        plant.GetBodyByName("spatula", spatula_instances[i]);
    plant.SetFreeBodyPose(&plant_context, spatula_body, X_WSpatula);
  }

  if (FLAGS_use_sycl) {
    if (FLAGS_contact_model == "hydroelastic" ||
        FLAGS_contact_model == "hydroelastic_with_fallback") {
      plant.set_sycl_for_hydroelastic_contact(true);
    } else {
      fmt::print(stderr,
                 "SYCL is not used for hydroelastic contact because "
                 "the contact model is not 'hydroelastic' or "
                 "'hydroelastic_with_fallback'.\n");
    }
  } else {
    plant.set_sycl_for_hydroelastic_contact(false);
  }

  // Set finger joint positions for all grippers
  for (int i = 0; i < num_instances; ++i) {
    const PrismaticJoint<double>& left_joint =
        plant.GetJointByName<PrismaticJoint>("left_finger_sliding_joint",
                                             gripper_instances[i]);
    left_joint.set_translation(&plant_context, -0.01);
    const PrismaticJoint<double>& right_joint =
        plant.GetJointByName<PrismaticJoint>("right_finger_sliding_joint",
                                             gripper_instances[i]);
    right_joint.set_translation(&plant_context, 0.01);
  }

  // Simulate.
  simulator.Initialize();
  if (FLAGS_visualize) {
    meshcat->StartRecording();
  }
  std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();
  simulator.AdvanceTo(FLAGS_simulation_sec);
  std::chrono::steady_clock::time_point end_time =
      std::chrono::steady_clock::now();
  double total_advance_to_time =
      std::chrono::duration<double>(end_time - start_time).count();
  if (FLAGS_visualize) {
    meshcat->StopRecording();
  }

  // TODO(#19142) According to issue 19142, we can playback contact forces and
  //  torques; however, contact surfaces are not recorded properly.
  //  For now, we delete contact surfaces to prevent confusion in the playback.
  //  Remove deletion when 19142 is resovled.
  if (FLAGS_visualize) {
    // Delete contact surfaces for all instances
    for (int i = 0; i < num_instances; ++i) {
      std::string spatula_name = "spatula";
      if (i > 0) {
        spatula_name += "_" + std::to_string(i);
      }

      meshcat->Delete(
          "/drake/contact_forces/hydroelastic/"
          "left_finger_bubble+" +
          spatula_name +
          "/"
          "contact_surface");
      meshcat->Delete(
          "/drake/contact_forces/hydroelastic/"
          "right_finger_bubble+" +
          spatula_name +
          "/"
          "contact_surface");
    }
    meshcat->PublishRecording();

    if (!std::filesystem::exists(html_out_dir)) {
      std::filesystem::create_directories(html_out_dir);
    }

    std::string device_type = FLAGS_use_sycl ? "sycl-gpu" : "drake-cpu";
    std::string demo_name = "spatula_slip_control_" +
                            std::to_string(FLAGS_mesh_res) + "_" +
                            std::to_string(FLAGS_num_instances);

    std::ofstream f(html_out_dir + demo_name + "_" + device_type + ".html");
    fmt::print("Writing meshcat html to {}\n",
               html_out_dir + demo_name + "_" + device_type + ".html");
    f << meshcat->StaticHtml();
    f.close();
  }
  PrintSimulatorStatistics(simulator);
  if (FLAGS_print_perf) {
    if (FLAGS_use_sycl) {
      PrintPerformanceStats(plant, scene_graph, scene_graph_context,
                            /*sycl_used=*/true, FLAGS_mesh_res,
                            FLAGS_num_instances, total_advance_to_time,
                            timing_out_dir);
    } else {
      PrintPerformanceStats(plant, scene_graph, scene_graph_context,
                            /*sycl_used=*/false, FLAGS_mesh_res,
                            FLAGS_num_instances, total_advance_to_time,
                            timing_out_dir);
    }
  }
  return 0;
}

}  // namespace
}  // namespace spatula_slip_control
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is an example of using the hydroelastic contact model with\n"
      "robot grippers with compliant bubble fingers and compliant spatulas.\n"
      "The example poses spatulas in the closed grip of grippers and\n"
      "uses an open loop square wave controller to perform controlled\n"
      "rotational slip of the spatulas while maintaining them in\n"
      "the grippers' grasp. Multiple instances can be simulated by\n"
      "setting --num_instances. Use the MeshCat URL from the console log\n"
      "messages for visualization. See the README.md file for more\n"
      "information.\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::spatula_slip_control::DoMain();
}
