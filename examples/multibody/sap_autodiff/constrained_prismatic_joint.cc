#include <iostream>
#include <gflags/gflags.h>

#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/examples/multibody/sap_autodiff/test_scenario.h"

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

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  SapAutodiffTestParameters params;
  params.simulate = true;

  ConstrainedPrismaticJointScenaro scenario;
  scenario.RunTests(params);

  return 0;
}