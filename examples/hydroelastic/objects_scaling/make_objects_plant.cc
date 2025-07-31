#include "drake/examples/hydroelastic/objects_scaling/make_objects_plant.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>

#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/shape_specification.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/multibody/tree/uniform_gravity_field_element.h"

namespace drake {
namespace examples {
namespace objects_scaling {

using Eigen::Vector3d;
using geometry::AddCompliantHydroelasticProperties;
using geometry::AddContactMaterial;
using geometry::Box;
using geometry::Cylinder;
using geometry::ProximityProperties;
using geometry::Sphere;
using math::RigidTransformd;
using math::RotationMatrixd;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::RigidBody;
using multibody::SpatialInertia;

// Helper class to manage object placement in layers
class ObjectPlacer {
 public:
  ObjectPlacer(double floor_width, double floor_depth, double object_spacing,
               double layer_height, double initial_height)
      : floor_width_(floor_width),
        floor_depth_(floor_depth),
        object_spacing_(object_spacing),
        layer_height_(layer_height),
        current_z_(initial_height),
        current_x_(-floor_width / 2.0 +
                   0.15),  // Hard code for sufficient distance of edge
        current_y_(-floor_depth / 2.0 + 0.15) {}

  Vector3d GetNextPosition() {
    Vector3d position(current_x_, current_y_, current_z_);

    // Move to next position in current layer
    current_x_ += object_spacing_;
    if (current_x_ > floor_width_ / 2.0 - 0.15 - object_spacing_ / 2.0) {
      // Move to next row
      current_x_ = (-floor_width_ / 2.0 + 0.15);
      current_y_ += object_spacing_;

      if (current_y_ > floor_depth_ / 2.0 - 0.15 - object_spacing_ / 2.0) {
        // Move to next layer
        current_y_ = (-floor_depth_ / 2.0 + 0.15);
        current_z_ += layer_height_;
      }
    }

    return position;
  }

 private:
  double floor_width_;
  double floor_depth_;
  double object_spacing_;
  double layer_height_;
  double current_z_;
  double current_x_;
  double current_y_;
};

// Helper function to create a random resolution factor within range
double SampleResolutionFactor(std::mt19937& gen, double min_factor,
                              double max_factor) {
  std::uniform_real_distribution<double> dist(min_factor, max_factor);
  return dist(gen);
}

// Helper function to add a sphere primitive
void AddSphere(const std::string& name, const Vector3d& position,
               const PrimitiveParams& params, double resolution_factor,
               MultibodyPlant<double>* plant) {
  const double radius = params.radius;
  const double mass = params.mass;

  // Add the sphere body
  const RigidBody<double>& sphere = plant->AddRigidBody(
      name, SpatialInertia<double>::SolidSphereWithMass(mass, radius));

  // Set up mechanical properties
  ProximityProperties sphere_props;
  CoulombFriction<double> friction(params.friction_coefficient,
                                   params.friction_coefficient);
  AddContactMaterial(params.dissipation, {} /* point stiffness */, friction,
                     &sphere_props);
  AddCompliantHydroelasticProperties(
      radius * resolution_factor, params.hydroelastic_modulus, &sphere_props);

  // Register collision geometry
  plant->RegisterCollisionGeometry(sphere, RigidTransformd(position),
                                   Sphere(radius), name + "_collision",
                                   std::move(sphere_props));

  // Register visual geometry
  const Vector4<double> orange(1.0, 0.55, 0.0, 0.8);
  plant->RegisterVisualGeometry(sphere, RigidTransformd(position),
                                Sphere(radius), name + "_visual", orange);
}

// Helper function to add a cylinder primitive
void AddCylinder(const std::string& name, const Vector3d& position,
                 const PrimitiveParams& params, double resolution_factor,
                 MultibodyPlant<double>* plant) {
  const double radius = params.radius;
  const double height = params.height;
  const double mass = params.mass;

  // Add the cylinder body
  const RigidBody<double>& cylinder =
      plant->AddRigidBody(name, SpatialInertia<double>::SolidCylinderWithMass(
                                    mass, radius, height, Vector3d::UnitZ()));

  // Set up mechanical properties
  ProximityProperties cylinder_props;
  CoulombFriction<double> friction(params.friction_coefficient,
                                   params.friction_coefficient);
  AddContactMaterial(params.dissipation, {} /* point stiffness */, friction,
                     &cylinder_props);
  AddCompliantHydroelasticProperties(
      radius * resolution_factor, params.hydroelastic_modulus, &cylinder_props);

  // Register collision geometry
  plant->RegisterCollisionGeometry(
      cylinder, RigidTransformd(position), Cylinder(radius, height),
      name + "_collision", std::move(cylinder_props));

  // Register visual geometry
  const Vector4<double> blue(0.0, 0.5, 1.0, 0.8);
  plant->RegisterVisualGeometry(cylinder, RigidTransformd(position),
                                Cylinder(radius, height), name + "_visual",
                                blue);
}

// Helper function to add a box primitive
void AddBox(const std::string& name, const Vector3d& position,
            const PrimitiveParams& params, double resolution_factor,
            MultibodyPlant<double>* plant) {
  const double width = params.width;
  const double depth = params.depth;
  const double height = params.height;
  const double mass = params.mass;

  // Add the box body
  const RigidBody<double>& box = plant->AddRigidBody(
      name,
      SpatialInertia<double>::SolidBoxWithMass(mass, width, depth, height));

  // Set up mechanical properties
  ProximityProperties box_props;
  CoulombFriction<double> friction(params.friction_coefficient,
                                   params.friction_coefficient);
  AddContactMaterial(params.dissipation, {} /* point stiffness */, friction,
                     &box_props);
  // Use the smallest dimension for resolution hint
  double min_dimension = std::min({width, depth, height});
  AddCompliantHydroelasticProperties(min_dimension * resolution_factor,
                                     params.hydroelastic_modulus, &box_props);

  // Register collision geometry
  plant->RegisterCollisionGeometry(box, RigidTransformd(position),
                                   Box(width, depth, height),
                                   name + "_collision", std::move(box_props));

  // Register visual geometry
  const Vector4<double> green(0.0, 1.0, 0.5, 0.8);
  plant->RegisterVisualGeometry(box, RigidTransformd(position),
                                Box(width, depth, height), name + "_visual",
                                green);
}

// Helper function to add complex objects from SDF files
void AddComplexObject(const std::string& sdf_path,
                      const std::string& frame_name,
                      const std::string& instance_name,
                      const Vector3d& position, MultibodyPlant<double>* plant) {
  // Create a parser with a unique model name prefix to avoid naming conflicts
  drake::multibody::Parser parser(plant, instance_name);
  auto model_instance = parser.AddModelsFromUrl(sdf_path);

  if (!model_instance.empty()) {
    // Get the frame for the model
    auto& frame = plant->GetFrameByName(frame_name, model_instance[0]);
    // Set the default pose instead of welding, so the object can move under
    // gravity
    plant->SetDefaultFreeBodyPose(frame.body(), RigidTransformd(position));
  }
}

void AddBenchmarkObjects(const BenchmarkConfig& config,
                         MultibodyPlant<double>* plant) {
  DRAKE_DEMAND(plant != nullptr);

  // Initialize random number generator for resolution sampling
  std::mt19937 gen(config.random_seed);

  // Initialize object placer
  ObjectPlacer placer(config.floor_width, config.floor_depth,
                      config.object_spacing, config.layer_height,
                      config.initial_height);

  // Add spheres
  for (int i = 0; i < config.num_spheres; ++i) {
    std::string name = "Sphere_" + std::to_string(i);
    Vector3d position = placer.GetNextPosition();
    double resolution_factor =
        SampleResolutionFactor(gen, config.sphere_params.resolution_factor_min,
                               config.sphere_params.resolution_factor_max);
    AddSphere(name, position, config.sphere_params, resolution_factor, plant);
  }

  // Add cylinders
  for (int i = 0; i < config.num_cylinders; ++i) {
    std::string name = "Cylinder_" + std::to_string(i);
    Vector3d position = placer.GetNextPosition();
    double resolution_factor = SampleResolutionFactor(
        gen, config.cylinder_params.resolution_factor_min,
        config.cylinder_params.resolution_factor_max);
    AddCylinder(name, position, config.cylinder_params, resolution_factor,
                plant);
  }

  // Add boxes
  for (int i = 0; i < config.num_boxes; ++i) {
    std::string name = "Box_" + std::to_string(i);
    Vector3d position = placer.GetNextPosition();
    double resolution_factor =
        SampleResolutionFactor(gen, config.box_params.resolution_factor_min,
                               config.box_params.resolution_factor_max);
    AddBox(name, position, config.box_params, resolution_factor, plant);
  }

  // Add grippers
  for (int i = 0; i < config.num_grippers; ++i) {
    std::string instance_name = "Gripper_" + std::to_string(i);
    Vector3d position = placer.GetNextPosition();
    AddComplexObject(
        "package://drake_models/wsg_50_description/sdf/"
        "schunk_wsg_50_hydro_bubble.sdf",
        "body_frame", instance_name, position, plant);
  }

  // Add peppers
  for (int i = 0; i < config.num_peppers; ++i) {
    std::string instance_name = "Pepper_" + std::to_string(i);
    Vector3d position = placer.GetNextPosition();
    AddComplexObject(
        "package://drake_models/veggies/"
        "yellow_bell_pepper_no_stem_low.sdf",
        "yellow_bell_pepper_no_stem", instance_name, position, plant);
  }

  // Add the floor
  drake::multibody::Parser parser(plant);
  auto floor_models = parser.AddModelsFromUrl(
      "package://drake/examples/hydroelastic/objects_scaling/floor.sdf");
  if (!floor_models.empty()) {
    plant->WeldFrames(plant->world_frame(),
                      plant->GetFrameByName("Floor", floor_models[0]),
                      RigidTransformd::Identity());
  }

  // Set gravity
  plant->mutable_gravity_field().set_gravity_vector(Vector3d{0, 0, -9.81});
}

// Legacy function for backward compatibility
void AddBallPlateBodies(double radius, double mass, double hydroelastic_modulus,
                        double dissipation,
                        const CoulombFriction<double>& surface_friction,
                        double resolution_hint_factor,
                        MultibodyPlant<double>* plant) {
  DRAKE_DEMAND(plant != nullptr);

  // Add the ball. Let B be the ball's frame (at its center). The ball's
  // center of mass Bcm is coincident with Bo.
  const RigidBody<double>& ball = plant->AddRigidBody(
      "Ball", SpatialInertia<double>::SolidSphereWithMass(mass, radius));

  // Set up mechanical properties of the ball.
  ProximityProperties ball_props;
  AddContactMaterial(dissipation, {} /* point stiffness */, surface_friction,
                     &ball_props);
  AddCompliantHydroelasticProperties(radius * resolution_hint_factor,
                                     hydroelastic_modulus, &ball_props);
  plant->RegisterCollisionGeometry(ball, RigidTransformd::Identity(),
                                   Sphere(radius), "collision",
                                   std::move(ball_props));
  const Vector4<double> orange(1.0, 0.55, 0.0, 0.2);
  plant->RegisterVisualGeometry(ball, RigidTransformd::Identity(),
                                Sphere(radius), "visual", orange);

  drake::multibody::Parser parser(plant);
  // Add the floor. Assume the frame named "Floor" is in the SDFormat file.
  parser.AddModelsFromUrl(
      "package://drake/examples/hydroelastic/objects_scaling/floor.sdf");
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("Floor"),
                    RigidTransformd::Identity());

  // Gravity acting in the -z direction.
  plant->mutable_gravity_field().set_gravity_vector(Vector3d{0, 0, -9.81});
}

}  // namespace objects_scaling
}  // namespace examples
}  // namespace drake
