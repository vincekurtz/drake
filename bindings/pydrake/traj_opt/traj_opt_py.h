#pragma once

/* This file declares the functions that bind the drake::traj_opt namespace.
These functions form a complete partition of the drake::traj_opt bindings.

The implementations of these functions are parceled out into various *.cc
files as indicated in each function's documentation. */

#include "drake/bindings/pydrake/pydrake_pybind.h"

namespace drake {
namespace pydrake {
namespace internal {

// For simplicity, these declarations are listed in alphabetical order.

/* Defines bindings per traj_opt_py_covergence_criteria_tol.cc. */
void DefineTrajOptConvergenceCriteriaTolerances(py::module m);

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
