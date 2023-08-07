#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/serialize_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/traj_opt/convergence_criteria_tolerances.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefineTrajOptConvergenceCriteriaTolerances(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::traj_opt;
  constexpr auto& doc = pydrake_doc.drake.traj_opt;

  {
    using Class = ConvergenceCriteriaTolerances;
    constexpr auto& cls_doc = doc.ConvergenceCriteriaTolerances;
    py::class_<Class> cls(m, "ConvergenceCriteriaTolerances", cls_doc.doc);
    cls  // BR
        .def(ParamInit<Class>(), cls_doc.doc);
    DefAttributesUsingSerialize(&cls, cls_doc);
    DefReprUsingSerialize(&cls);
    DefCopyAndDeepCopy(&cls);
  }
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
