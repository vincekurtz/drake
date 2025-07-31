#pragma once

#include <memory>
#include <random>
#include <vector>

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace examples {
namespace objects_scaling {

// Structure to hold parameters for primitive objects
struct PrimitiveParams {
  // Object properties
  double radius = 0.05;  // For sphere and cylinder
  double height = 0.1;   // For cylinder
  double width = 0.1;    // For box
  double depth = 0.1;    // For box
  double mass = 0.1;

  // Hydroelastic properties
  double hydroelastic_modulus = 3.0e4;
  double dissipation = 3.0;
  double friction_coefficient = 0.3;

  // Mesh resolution range for sampling
  double resolution_factor_min = 1.0;
  double resolution_factor_max = 1.0;
};

// Structure to hold benchmark configuration
struct BenchmarkConfig {
  // Primitive object counts
  int num_spheres = 1;
  int num_cylinders = 0;
  int num_boxes = 0;

  // Complex object counts
  int num_grippers = 0;
  int num_peppers = 0;

  // Primitive parameters
  PrimitiveParams sphere_params;
  PrimitiveParams cylinder_params;
  PrimitiveParams box_params;

  // Object placement parameters
  double floor_width = 3.0;      // meters
  double floor_depth = 3.0;      // meters
  double object_spacing = 0.10;  // meters between object centers
  double layer_height = 0.3;     // meters between z layers
  double initial_height = 0.30;  // meters above floor for first layer

  // Random seed for reproducible results
  int random_seed = 42;
};

/** This function modifies a MultibodyPlant by adding a comprehensive set of
 objects for benchmarking the proximity engine. It can add multiple types of
 primitive objects (spheres, cylinders, boxes) with varying mesh resolutions,
 as well as complex objects (grippers, peppers) arranged in layers.

 @param[in] config
   The benchmark configuration specifying object counts, parameters, and layout.
 @param[in,out] plant
   The bodies will be added here.

 @pre `plant` is not null.

 See also
 https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
*/
void AddBenchmarkObjects(const BenchmarkConfig& config,
                         multibody::MultibodyPlant<double>* plant);

/** Legacy function for backward compatibility - adds a single ball and floor.
 This function modifies a MultibodyPlant by adding a ball falling on a dinner
 plate. The plate and floor are read from sdf files but the ball is
 constructed programmatically.

 @param[in] radius
   The radius (meters) of the ball.
 @param[in] mass
   The mass (kg) of the ball.
 @param[in] hydroelastic_modulus
   The hydroelastic modulus (Pa) of the ball.
 @param[in] dissipation
   The Hunt & Crossley dissipation constant (s/m) for the ball.
 @param[in] surface_friction
   The Coulomb's law coefficients (unitless) of friction of the ball.
 @param[in] resolution_hint_factor
   This scaling factor (unitless) multiplied by the radius of the ball
   gives the target edge length of the mesh on the surface of the ball.
   The smaller number gives a finer mesh with more tetrahedral elements.
 @param[in,out] plant
   The bodies will be added here.

 @pre `plant` is not null.

 See also
 https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
*/
void AddBallPlateBodies(
    double radius, double mass, double hydroelastic_modulus, double dissipation,
    const multibody::CoulombFriction<double>& surface_friction,
    double resolution_hint_factor, multibody::MultibodyPlant<double>* plant);

}  // namespace objects_scaling
}  // namespace examples
}  // namespace drake
