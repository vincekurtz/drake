#include <filesystem>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/cpu_timing_logger.h"
#include "drake/common/eigen_types.h"
#include "drake/common/problem_size_logger.h"
#include "drake/examples/hydroelastic/objects_scaling/make_objects_plant.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/visualization/visualization_config_functions.h"

DEFINE_double(simulation_time, 1.0,
              "Desired duration of the simulation in seconds.");
// See MultibodyPlantConfig for the valid strings of contact_model.
DEFINE_string(contact_model, "hydroelastic_with_fallback",
              "Contact model. Options are: 'point', 'hydroelastic', "
              "'hydroelastic_with_fallback'.");
// See MultibodyPlantConfig for the valid strings of contact surface
// representation.
DEFINE_string(contact_surface_representation, "polygon",
              "Contact-surface representation for hydroelastics. "
              "Options are: 'triangle' or 'polygon'. Default is 'polygon'.");

DEFINE_double(mbp_dt, 0.001,
              "The fixed time step period (in seconds) of discrete updates "
              "for the multibody plant modeled as a discrete system. "
              "Strictly positive.");

DEFINE_bool(
    use_sycl, false,
    "Use SYCL for hydroelastic contact. This flag is only used when "
    "the contact model is 'hydroelastic' or 'hydroelastic_with_fallback'.");

DEFINE_bool(
    use_legacy_single_ball, false,
    "Use the legacy single ball setup instead of the comprehensive benchmark.");

DEFINE_string(
    config_name, "",
    "Descriptive name for this benchmark configuration (e.g., 'legacy', "
    "'resolution_study', 'scaling_study'). If empty, will generate a name "
    "based on object counts.");

DEFINE_bool(print_perf, true, "Print performance statistics");

// === OBJECT COUNT PARAMETERS ===
DEFINE_int32(num_spheres, 0, "Number of sphere objects to create.");
DEFINE_int32(num_cylinders, 0, "Number of cylinder objects to create.");
DEFINE_int32(num_boxes, 0, "Number of box objects to create.");
DEFINE_int32(num_grippers, 0, "Number of gripper objects to create.");
DEFINE_int32(num_peppers, 0, "Number of pepper objects to create.");

// === PRIMITIVE OBJECT PARAMETERS ===
// Sphere parameters
DEFINE_double(sphere_radius, 0.05, "Radius of sphere objects [m].");
DEFINE_double(sphere_mass, 0.1, "Mass of sphere objects [kg].");
DEFINE_double(sphere_hydroelastic_modulus, 3.0e4,
              "Hydroelastic modulus of spheres [Pa].");
DEFINE_double(sphere_dissipation, 3.0,
              "Hunt & Crossley dissipation of spheres [s/m].");
DEFINE_double(sphere_friction, 0.3, "Friction coefficient of spheres.");
DEFINE_double(sphere_resolution_min, 0.5,
              "Minimum resolution factor for spheres.");
DEFINE_double(sphere_resolution_max, 0.5,
              "Maximum resolution factor for spheres.");

// Cylinder parameters
DEFINE_double(cylinder_radius, 0.05, "Radius of cylinder objects [m].");
DEFINE_double(cylinder_height, 0.1, "Height of cylinder objects [m].");
DEFINE_double(cylinder_mass, 0.1, "Mass of cylinder objects [kg].");
DEFINE_double(cylinder_hydroelastic_modulus, 3.0e4,
              "Hydroelastic modulus of cylinders [Pa].");
DEFINE_double(cylinder_dissipation, 3.0,
              "Hunt & Crossley dissipation of cylinders [s/m].");
DEFINE_double(cylinder_friction, 0.3, "Friction coefficient of cylinders.");
DEFINE_double(cylinder_resolution_min, 0.5,
              "Minimum resolution factor for cylinders.");
DEFINE_double(cylinder_resolution_max, 0.5,
              "Maximum resolution factor for cylinders.");

// Box parameters
DEFINE_double(box_width, 0.1, "Width of box objects [m].");
DEFINE_double(box_depth, 0.1, "Depth of box objects [m].");
DEFINE_double(box_height, 0.1, "Height of box objects [m].");
DEFINE_double(box_mass, 0.1, "Mass of box objects [kg].");
DEFINE_double(box_hydroelastic_modulus, 3.0e4,
              "Hydroelastic modulus of boxes [Pa].");
DEFINE_double(box_dissipation, 3.0,
              "Hunt & Crossley dissipation of boxes [s/m].");
DEFINE_double(box_friction, 0.3, "Friction coefficient of boxes.");
DEFINE_double(box_resolution_min, 0.5, "Minimum resolution factor for boxes.");
DEFINE_double(box_resolution_max, 0.5, "Maximum resolution factor for boxes.");

// === OBJECT PLACEMENT PARAMETERS ===
DEFINE_double(object_spacing, 0.10, "Spacing between object centers [m].");
DEFINE_double(layer_height, 0.3, "Height between object layers [m].");
DEFINE_double(initial_height, 0.30,
              "Initial height of first layer above floor [m].");
DEFINE_int32(random_seed, 42, "Random seed for resolution sampling.");

// === LEGACY PARAMETERS (for single ball mode) ===
DEFINE_double(hydroelastic_modulus, 3.0e4,
              "Hydroelastic modulus of the Primitives, [Pa].");
DEFINE_double(
    resolution_hint_factor, 0.3,
    "This scaling factor, [unitless], multiplied by the radius of "
    "the Primitives gives the target edge length of the mesh of the Primitives "
    "on the surface of its hydroelastic representation. The smaller "
    "number gives a finer mesh with more tetrahedral elements.");
DEFINE_double(dissipation, 3.0,
              "Hunt & Crossley dissipation, [s/m], for the ball");
DEFINE_double(friction_coefficient, 0.3,
              "coefficient for both static and dynamic friction, [unitless], "
              "of the ball.");

// Ball's initial spatial velocity.
DEFINE_double(vx, 0,
              "Ball's initial translational velocity in the x-axis in m/s.");
DEFINE_double(vy, 0.0,
              "Ball's initial translational velocity in the y-axis in m/s.");
DEFINE_double(vz, -7.0,
              "Ball's initial translational velocity in the z-axis in m/s.");
DEFINE_double(wx, 0.0,
              "Ball's initial angular velocity in the x-axis in degrees/s.");
DEFINE_double(wy, -10.0,
              "Ball's initial angular velocity in the y-axis in degrees/s.");
DEFINE_double(wz, 0.0,
              "Ball's initial angular velocity in the z-axis in degrees/s.");

// Ball's initial pose.
DEFINE_double(z0, 0.15, "Ball's initial position in the z-axis.");
DEFINE_double(x0, 0.10, "Ball's initial position in the x-axis.");

namespace drake {
using systems::BasicVector;
using systems::Context;
using systems::SimulatorConfig;
namespace examples {
namespace objects_scaling {
namespace {

using drake::geometry::internal::kComplianceType;
using drake::geometry::internal::kHydroGroup;
using drake::math::RigidTransformd;
using drake::multibody::CoulombFriction;
using drake::multibody::SpatialVelocity;
using Eigen::Vector3d;

BenchmarkConfig CreateBenchmarkConfig() {
  BenchmarkConfig config;

  // Object counts
  config.num_spheres = FLAGS_num_spheres;
  config.num_cylinders = FLAGS_num_cylinders;
  config.num_boxes = FLAGS_num_boxes;
  config.num_grippers = FLAGS_num_grippers;
  config.num_peppers = FLAGS_num_peppers;

  // Sphere parameters
  config.sphere_params.radius = FLAGS_sphere_radius;
  config.sphere_params.mass = FLAGS_sphere_mass;
  config.sphere_params.hydroelastic_modulus = FLAGS_sphere_hydroelastic_modulus;
  config.sphere_params.dissipation = FLAGS_sphere_dissipation;
  config.sphere_params.friction_coefficient = FLAGS_sphere_friction;
  config.sphere_params.resolution_factor_min = FLAGS_sphere_resolution_min;
  config.sphere_params.resolution_factor_max = FLAGS_sphere_resolution_max;

  // Cylinder parameters
  config.cylinder_params.radius = FLAGS_cylinder_radius;
  config.cylinder_params.height = FLAGS_cylinder_height;
  config.cylinder_params.mass = FLAGS_cylinder_mass;
  config.cylinder_params.hydroelastic_modulus =
      FLAGS_cylinder_hydroelastic_modulus;
  config.cylinder_params.dissipation = FLAGS_cylinder_dissipation;
  config.cylinder_params.friction_coefficient = FLAGS_cylinder_friction;
  config.cylinder_params.resolution_factor_min = FLAGS_cylinder_resolution_min;
  config.cylinder_params.resolution_factor_max = FLAGS_cylinder_resolution_max;

  // Box parameters
  config.box_params.width = FLAGS_box_width;
  config.box_params.depth = FLAGS_box_depth;
  config.box_params.height = FLAGS_box_height;
  config.box_params.mass = FLAGS_box_mass;
  config.box_params.hydroelastic_modulus = FLAGS_box_hydroelastic_modulus;
  config.box_params.dissipation = FLAGS_box_dissipation;
  config.box_params.friction_coefficient = FLAGS_box_friction;
  config.box_params.resolution_factor_min = FLAGS_box_resolution_min;
  config.box_params.resolution_factor_max = FLAGS_box_resolution_max;

  // Placement parameters
  config.object_spacing = FLAGS_object_spacing;
  config.layer_height = FLAGS_layer_height;
  config.initial_height = FLAGS_initial_height;
  config.random_seed = FLAGS_random_seed;
  config.floor_width = 3.0;
  config.floor_depth = 3.0;

  return config;
}

void PrintPerformanceStats(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::geometry::SceneGraph<double>& scene_graph,
    const drake::systems::Context<double>& scene_graph_context, bool sycl_used,
    const BenchmarkConfig& config) {
  // Create a descriptive name for output files
  std::string demo_name = "objects_scaling";

  if (!FLAGS_config_name.empty()) {
    // Use the provided config name
    demo_name += "_" + FLAGS_config_name;
  } else if (FLAGS_use_legacy_single_ball) {
    // Legacy mode naming
    demo_name += "_legacy";
  } else {
    // Generate name based on object counts
    demo_name += "_s" + std::to_string(config.num_spheres);
    demo_name += "_c" + std::to_string(config.num_cylinders);
    demo_name += "_b" + std::to_string(config.num_boxes);
    demo_name += "_g" + std::to_string(config.num_grippers);
    demo_name += "_p" + std::to_string(config.num_peppers);
  }

  std::string runtime_device;
  const char* env_var = std::getenv("ONEAPI_DEVICE_SELECTOR");
  if (env_var != nullptr) {
    runtime_device = env_var;
  }
  std::string out_dir =
      "/home/huzaifaunjhawala/drake_vince/performance_jsons_bvh_opt2/";
  // Create output directory if it doesn't exist
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
      if (props && props->HasProperty(kHydroGroup, kComplianceType)) {
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
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  multibody::MultibodyPlantConfig config;
  // We allow only discrete systems.
  DRAKE_DEMAND(FLAGS_mbp_dt > 0.0);
  config.time_step = FLAGS_mbp_dt;
  config.penetration_allowance = 0.001;
  config.contact_model = FLAGS_contact_model;
  config.contact_surface_representation = FLAGS_contact_surface_representation;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  if (FLAGS_use_legacy_single_ball) {
    // Legacy single ball setup
    const double radius = 0.05;  // m
    const double mass = 0.1;     // kg
    AddBallPlateBodies(
        radius, mass, FLAGS_hydroelastic_modulus, FLAGS_dissipation,
        CoulombFriction<double>{// static friction (unused in discrete systems)
                                FLAGS_friction_coefficient,
                                // dynamic friction
                                FLAGS_friction_coefficient},
        FLAGS_resolution_hint_factor, &plant);
  } else {
    // New comprehensive benchmark setup
    BenchmarkConfig benchmark_config = CreateBenchmarkConfig();
    AddBenchmarkObjects(benchmark_config, &plant);
  }

  plant.Finalize();

  auto meshcat = std::make_shared<geometry::Meshcat>();
  visualization::ApplyVisualizationConfig(
      visualization::VisualizationConfig{
          .default_proximity_color = geometry::Rgba{1, 0, 0, 0.25},
          .enable_alpha_sliders = true,
      },
      &builder, nullptr, nullptr, nullptr, meshcat);

  auto diagram = builder.Build();
  auto simulator = MakeSimulatorFromGflags(*diagram);

  if (FLAGS_use_legacy_single_ball) {
    // Set the ball's initial pose and velocity (legacy mode)
    systems::Context<double>& plant_context =
        plant.GetMyMutableContextFromRoot(&simulator->get_mutable_context());
    plant.SetFreeBodyPose(
        &plant_context, plant.GetBodyByName("Ball"),
        math::RigidTransformd{Vector3d(FLAGS_x0, 0.0, FLAGS_z0)});
    plant.SetFreeBodySpatialVelocity(
        &plant_context, plant.GetBodyByName("Ball"),
        SpatialVelocity<double>{
            M_PI / 180.0 * Vector3d(FLAGS_wx, FLAGS_wy, FLAGS_wz),
            Vector3d(FLAGS_vx, FLAGS_vy, FLAGS_vz)});
  }

  // Use SYCL if required
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
  }
  meshcat->StartRecording();
  simulator->AdvanceTo(FLAGS_simulation_time);
  meshcat->StopRecording();
  meshcat->PublishRecording();

  Context<double>& mutable_root_context = simulator->get_mutable_context();
  Context<double>& scene_graph_context =
      diagram->GetMutableSubsystemContext(scene_graph, &mutable_root_context);

  BenchmarkConfig benchmark_config = CreateBenchmarkConfig();
  if (FLAGS_print_perf) {
    if (FLAGS_use_sycl) {
      PrintPerformanceStats(plant, scene_graph, scene_graph_context,
                            /*sycl_used=*/true, benchmark_config);
    } else {
      PrintPerformanceStats(plant, scene_graph, scene_graph_context,
                            /*sycl_used=*/false, benchmark_config);
    }
  }
  return 0;
}

}  // namespace
}  // namespace objects_scaling
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(R"""(
This is a comprehensive benchmark for testing hydroelastic contact with 
multiple object types including primitives (spheres, cylinders, boxes) and 
complex objects (grippers, peppers). Objects are automatically arranged in 
layers with configurable mesh resolutions and material properties.

Examples:
  # Single ball (legacy mode)
  --use_legacy_single_ball=true
  
  # 10 spheres with resolution factors between 0.1 and 2.0
  --num_spheres=10 --sphere_resolution_min=0.1 --sphere_resolution_max=2.0
  
  # Mixed objects: 5 spheres, 3 cylinders, 2 boxes, 1 gripper
  --num_spheres=5 --num_cylinders=3 --num_boxes=2 --num_grippers=1
  
  # High resolution test
  --num_spheres=20 --sphere_resolution_min=2.0 --sphere_resolution_max=5.0

See the README.md file for more information.)""");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::objects_scaling::do_main();
}
