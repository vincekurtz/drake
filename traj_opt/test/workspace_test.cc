#include <iostream>

#include <gtest/gtest.h>

#include "drake/traj_opt/trajectory_optimizer_workspace.h"
#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"

#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace drake {
namespace traj_opt {
namespace internal {

using Eigen::VectorXd;
using Eigen::MatrixXd;

using multibody::MultibodyPlant;
using multibody::Parser;

GTEST_TEST(WorkspaceTest, SimpleWorkspaceDouble) {
  SimpleWorkspace<double> workspace(3, 0.0);

  double& a = workspace.get();
  EXPECT_TRUE(a == 0.0);
  a = 1.1;

  double& b = workspace.get();
  b = 2.2;
  EXPECT_FALSE(&a == &b);

  workspace.release(a);
  
  double& c = workspace.get();
  EXPECT_TRUE(&c == &a);
  EXPECT_TRUE(c == 1.1);
  c = 3.3;
  EXPECT_TRUE(a == 3.3);

  double& d = workspace.get();
  d = 4.4;

  EXPECT_THROW(workspace.get(), std::runtime_error);
  EXPECT_THROW(workspace.release(4.4), std::runtime_error);
}

GTEST_TEST(WorkspaceTest, Spinner) {
  const int num_steps = 40;
  MultibodyPlant<double> plant(1e-3);
  const std::string urdf_file = FindResourceOrThrow(
      "drake/traj_opt/examples/models/spinner_friction.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  const int num_eq_constraints = (num_steps + 1) * 1;
  const int num_vars = (num_steps + 1) * plant.num_positions();

  TrajectoryOptimizerWorkspace<double> workspace(num_steps, num_eq_constraints,
                                                 plant);

  VectorXd& a = workspace.get_q_size_tmp();
  VectorXd& b = workspace.get_q_size_tmp();
  workspace.release_q_size_tmp(a);
  VectorXd& c = workspace.get_q_size_tmp();
  EXPECT_FALSE(&a == &b);
  EXPECT_TRUE(&a == &c);
  EXPECT_TRUE(a.size() == plant.num_positions());

  VectorXd& v = workspace.get_v_size_tmp();
  EXPECT_TRUE(v.size() == plant.num_velocities());
  
  VectorXd& num_vars_tmp = workspace.get_num_vars_size_tmp();
  EXPECT_TRUE(num_vars_tmp.size() == num_vars);

  MatrixXd& J_tmp = workspace.get_num_vars_times_num_eq_size_tmp();
  EXPECT_TRUE(J_tmp.rows() == num_vars);
  EXPECT_TRUE(J_tmp.cols() == num_eq_constraints);

  std::vector<VectorXd>& q_tmp = workspace.get_q_sequence_tmp();
  EXPECT_TRUE(q_tmp.size() == num_steps);
  EXPECT_TRUE(q_tmp[0].size() == plant.num_positions());

  MultibodyForces<double>& f_ext = workspace.get_multibody_forces_tmp();
  EXPECT_TRUE(f_ext.CheckHasRightSizeForModel(plant));

}

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
