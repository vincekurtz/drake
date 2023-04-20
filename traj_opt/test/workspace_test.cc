#include <iostream>

#include <gtest/gtest.h>

#include "drake/traj_opt/trajectory_optimizer_workspace.h"

#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace drake {
namespace traj_opt {
namespace internal {

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

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
