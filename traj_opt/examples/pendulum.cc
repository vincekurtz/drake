#include <iostream>

#include "drake/traj_opt/solution_data.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/problem_data.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace pendulum {

int do_main() {
  std::cout << "hello world" << std::endl;

  ProblemDefinition optimization_problem;
  (void) optimization_problem;
  
  ProblemData prob_data;
  (void) prob_data;

  return 0;
}

}  // namespace pendulum
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() { return drake::traj_opt::examples::pendulum::do_main(); }