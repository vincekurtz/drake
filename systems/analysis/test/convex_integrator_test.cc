#include "drake/systems/analysis/convex_integrator.h"

#include <iostream>

#include <gtest/gtest.h>

namespace drake {
namespace systems {
namespace analysis_test {

// Simulate a simple double pendulum system with the convex integrator.
GTEST_TEST(ConvexIntegratorTest, DoublePendulum) {
  std::cout << "hello world!" << std::endl;
}

}  // namespace analysis_test
}  // namespace systems
}  // namespace drake
