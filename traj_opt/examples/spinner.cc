#include "drake/common/find_resource.h"
#include "drake/common/profiler.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace spinner {

using Eigen::Matrix4d;
using math::RigidTransformd;
using multibody::MultibodyPlant;
using multibody::Parser;

class SpinnerExample : public TrajOptExample {
 public:
  SpinnerExample() {
    Matrix4d viewer_matrix;
    viewer_matrix << 0.5542041605188553, 1.6653345369377343e-16,
        -0.8323807713201875, 0, -0.5394122147293351, 0.7616102066689284,
        -0.3591439205924354, 0, 0.6339496912724103, 0.6480354103733158,
        0.4220875452295455, 0, 2.7077050875903486, 2.385351603880167,
        1.5497360941544034, 1;
    const RigidTransformd X(viewer_matrix.transpose());
    meshcat_->SetTransform("/Cameras/default/rotated/<object>", X);
  }

 private:
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    // N.B. geometry of the spinner is chosen via gflags rather than yaml so
    // that we can use the same yaml format for all of the examples, without
    // cluttering it with spinner-specific options.
    std::string urdf_file = FindResourceOrThrow(
        "drake/traj_opt/examples/models/spinner_friction.urdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);
  }
};

}  // namespace spinner
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::spinner::SpinnerExample spinner_example;
  spinner_example.RunExample("drake/traj_opt/examples/spinner.yaml");
  return 0;
}
