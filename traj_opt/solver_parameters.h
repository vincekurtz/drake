#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/multibody/fem/petsc_symmetric_block_sparse_matrix.h"
#include "drake/traj_opt/convergence_criteria_tolerances.h"

namespace drake {
namespace traj_opt {

enum LinesearchMethod {
  // Simple backtracking linesearch with Armijo's condition
  kArmijo,

  // Backtracking linesearch that tries to find a local minimum
  kBacktracking
};

enum SolverMethod { kLinesearch, kTrustRegion };

enum GradientsMethod {
  // First order forward differences.
  kForwardDifferences,
  // Second order central differences.
  kCentralDifferences,
  // Fourth order central differences.
  kCentralDifferences4,
  // Automatic differentiation.
  kAutoDiff,
  // The optimizer will not be used for the computation of gradients. If
  // requested, an exception will be thrown.
  kNoGradients
};

struct SolverParameters {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SolverParameters);

  enum LinearSolverType {
    // Dense Eigen::LDLT solver.
    kDenseLdlt,
    // Pentadiagonal LU solver.
    kPentaDiagonalLu,
    // PETSc solver.
    kPetsc,
  };

  struct PetscSolverPatameters {
    using SolverType = drake::multibody::fem::internal::
        PetscSymmetricBlockSparseMatrix::SolverType;
    using PreconditionerType = drake::multibody::fem::internal::
        PetscSymmetricBlockSparseMatrix::PreconditionerType;
    double relative_tolerance{1.0e-12};
    SolverType solver_type{SolverType::kConjugateGradient};
    PreconditionerType preconditioner_type{
        PreconditionerType::kIncompleteCholesky};
  };

  ConvergenceCriteriaTolerances convergence_tolerances;

  SolverParameters() = default;

  // Which overall optimization strategy to use - linesearch or trust region
  // TODO(vincekurtz): better name for this?
  SolverMethod method = SolverMethod::kTrustRegion;

  // Which linesearch strategy to use
  LinesearchMethod linesearch_method = LinesearchMethod::kArmijo;

  // Maximum number of Gauss-Newton iterations
  int max_iterations = 100;

  // Maximum number of linesearch iterations
  int max_linesearch_iterations = 50;

  GradientsMethod gradients_method{kForwardDifferences};

  // Select the linear solver to be used in the Gauss-Newton step computation.
  LinearSolverType linear_solver{LinearSolverType::kPentaDiagonalLu};

  // Parameters for the PETSc solver. Ignored if linear_solver != kPetsc.
  PetscSolverPatameters petsc_parameters{};

  // Flag for whether to print out iteration data
  bool verbose = true;

  // Flag for whether to print (and compute) additional slow-to-compute
  // debugging info, like the condition number, at each iteration
  bool print_debug_data = false;

  // Only for debugging. When `true`, the computation with sparse algebra is
  // checked against a dense LDLT computation. This is an expensive check and
  // must be avoided unless we are trying to debug loss of precision due to
  // round-off errors or similar problems.
  bool debug_compare_against_dense{false};

  // Flag for whether to record linesearch data to a file at each iteration (for
  // later plotting). This saves a file called "linesearch_data_[k].csv" for
  // each iteration, where k is the iteration number. This file can then be
  // found somewhere in drake/bazel-out/.
  bool linesearch_plot_every_iteration = false;

  // Flag for whether to add a proximal operator term,
  //
  //      1/2 * rho * (q_k - q_{k-1})' * diag(H) * (q_k - q_{k-1})
  //
  // to the cost, where q_{k-1} are the decision variables at iteration {k-1}
  // and H_{k-1} is the Hessian at iteration k-1.
  bool proximal_operator = false;

  // Scale factor for the proximal operator cost
  double rho_proximal = 1e-8;

  // Contact model parameters
  // TODO(vincekurtz): this is definitely the wrong place to specify the contact
  // model - figure out the right place and put these parameters there
  double F = 1.0;       // force (in Newtons) at delta meters of penetration
  double delta = 0.01;  // penetration distance at which we apply F newtons
  double dissipation_velocity{0.1};  // Hunt-Crossley velocity, in m/s.
  double stiction_velocity{1.0e-2};  // Regularization of stiction, in m/s.
  double friction_coefficient{1.0};  // Coefficient of friction.

  bool force_at_a_distance{false};  // whether to allow force at a distance
  double smoothing_factor{0.01};    // force at a distance smoothing

  // Flags for making a contour plot with the first two decision variables.
  bool save_contour_data = false;
  double contour_q1_min = 0.0;
  double contour_q1_max = 1.0;
  double contour_q2_min = 0.0;
  double contour_q2_max = 1.0;

  // Flags for making line plots with the first decision variable
  bool save_lineplot_data = false;
  double lineplot_q_min = 0.0;
  double lineplot_q_max = 1.0;
};

}  // namespace traj_opt
}  // namespace drake
