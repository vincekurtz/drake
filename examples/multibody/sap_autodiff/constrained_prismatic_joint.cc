#include <iostream>
#include <gflags/gflags.h>

#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/examples/multibody/sap_autodiff/test_scenario.h"

DEFINE_bool(constraint, true,
            "Whether the initial state is such that constraints are active.");
DEFINE_bool(simulate, true,
            "Whether to run a quick simulation of the scenario.");
DEFINE_double(realtime_rate, 1.0, "Realtime rate for simulation.");
DEFINE_double(simulation_time, 2.0, "The time, in seconds, to simulate for.");
DEFINE_bool(test_autodiff, true, "Whether to run some autodiff tests.");
DEFINE_string(algebra, "both",
              "Type of algebra to use for testing autodiff. Options are: "
              "'sparse', 'dense', or 'both'.");
DEFINE_int32(num_steps, 1,
             "Number of timesteps to simulate for testing autodiff.");
DEFINE_double(time_step, 1e-2, "Size of the discrete timestep, in seconds");

namespace drake {
    
using multibody::PrismaticJoint;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;
using multibody::MultibodyPlant;

namespace examples {
namespace multibody {
namespace sap_autodiff {

class ConstrainedPrismaticJointScenaro final : public SapAutodiffTestScenario {
  void CreateDoublePlant(MultibodyPlant<double>* plant) const override {
    // Some parameters
    const double radius = 0.1;
    const double mass = 0.8;
    const double lower_limit = radius;  // this makes it look like the ground
                                        // is a constraint

    // Create the plant
    UnitInertia<double> G_Bo = UnitInertia<double>::SolidSphere(radius);
    SpatialInertia<double> M_Bo(mass, Vector3<double>::Zero(), G_Bo);
    const RigidBody<double>& body = plant->AddRigidBody("body", M_Bo);
    const math::RigidTransform<double> X;
    plant->RegisterVisualGeometry(body, X, geometry::Sphere(radius), "body");
    plant->RegisterVisualGeometry(plant->world_body(), X,
                                  geometry::Cylinder(0.01, 10), "vertical_rod");
    plant->AddJoint<PrismaticJoint>("joint", plant->world_body(), std::nullopt,
                                    body, std::nullopt,
                                    Vector3<double>::UnitZ(), lower_limit);
    plant->Finalize();
  }

  VectorX<double> get_x0_constrained() const override {
    return Vector2<double>(0.1, 0.0);
  }
  VectorX<double> get_x0_unconstrained() const override {
    return Vector2<double>(0.5, 0.0);
  }
};

} // namespace sap_autodiff
} // namespace multibody
} // namespace examples
} // namespace drake


int main(int argc, char* argv[]) {
  using drake::examples::multibody::sap_autodiff::ConstrainedPrismaticJointScenaro;
  using drake::examples::multibody::sap_autodiff::SapAutodiffTestParameters;
  using drake::examples::multibody::sap_autodiff::kAlgebraType;

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  SapAutodiffTestParameters params;
  params.constraint = FLAGS_constraint;
  params.simulate = FLAGS_simulate;
  params.realtime_rate = FLAGS_realtime_rate;
  params.simulation_time = FLAGS_simulation_time;
  params.test_autodiff = FLAGS_test_autodiff;
  params.num_steps = FLAGS_num_steps;
  params.time_step = FLAGS_time_step;

  if (FLAGS_algebra == "dense") {
    params.algebra = kAlgebraType::Dense;
  } else if (FLAGS_algebra == "sparse") {
    params.algebra = kAlgebraType::Sparse;
  } else {
    params.algebra = kAlgebraType::Both;
  }

  ConstrainedPrismaticJointScenaro scenario;
  scenario.RunTests(params);

  return 0;
}