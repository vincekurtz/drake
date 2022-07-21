#include <iostream>

#include "drake/traj_opt/solution_data.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace pendulum {

int do_main() {
  std::cout << "hello world" << std::endl;

  Solution soln;
  SolverStats stats;
  SolverFlag flag;

  std::cout << (flag == SolverFlag::kSuccess) << std::endl;

  (void) soln;
  (void) stats;

  return 0;
}

}  // namespace pendulum
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() { return drake::traj_opt::examples::pendulum::do_main(); }