#include "drake/bindings/pydrake/traj_opt/traj_opt_py.h"

namespace drake {
namespace pydrake {

PYBIND11_MODULE(traj_opt, m) {
  PYDRAKE_PREVENT_PYTHON3_MODULE_REIMPORT(m);

  m.doc() = R"""(
A collection of trajectory optimization methods.
)""";

  // The order of these calls matters. Some modules rely on prior definitions.
  internal::DefineTrajOptConvergenceCriteriaTolerances(m);
}

}  // namespace pydrake
}  // namespace drake
