#pragma once

#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/systems/analysis/implicit_integrator.h"

namespace drake {
namespace systems {

/**
 * The simplest exponential integrator,
 * https://en.wikipedia.org/wiki/Exponential_integrator#Second-order_method
 */
template <class T>
class ExponentialRosenbrockIntegrator final : public ImplicitIntegrator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExponentialRosenbrockIntegrator);

  ~ExponentialRosenbrockIntegrator() override;

  explicit ExponentialRosenbrockIntegrator(const System<T>& system,
                                           const T& max_step_size,
                                           Context<T>* context = nullptr)
      : ImplicitIntegrator<T>(system, context) {
    this->set_maximum_step_size(max_step_size);
  }

  /**
   * Returns false, because this integrator does not support error estimation.
   */
  bool supports_error_estimation() const final { return false; }

  /**
   * The error estimate order is 0, because this integrator does not provide
   * any error estimation.
   */
  int get_error_estimate_order() const final { return 0; }

 private:
  // Integrator parameters adopted from MATLODE, and simplified where possible.
  static struct Rosenbrock2Parameters {
    static constexpr double gamma = 0.29289321881345254;  // 1 - 1/√2
    static constexpr double m1 = 3.0 / (2.0 * gamma);
    static constexpr double m2 = 1.0 / (2.0 * gamma);
    static constexpr double c = -2.0 / gamma;
    static constexpr double a = 1.0 / gamma;
    static constexpr double m_hat1 = 1.0 / gamma;
  } params_;

  // Compute and factor the iteration matrix G = [I/(h*γ) - J]. The same
  // iteration matrix is used in both stages.
  static void ComputeAndFactorIterationMatrix(
      const MatrixX<T>& J, const T& h,
      typename ImplicitIntegrator<T>::IterationMatrix* iteration_matrix);

  // Rosenbrock integrators take one NR iteration for each stage, at each step.
  int64_t do_get_num_newton_raphson_iterations() const final {
    return 2 * this->get_num_steps_taken();
  }

  // The embedded error estimate means that no additional derivative
  // evaluations, NR iterations, matrix factorizations, etc are needed for the
  // error estimate.
  int64_t do_get_num_error_estimator_derivative_evaluations() const final {
    return 0;
  }
  int64_t do_get_num_error_estimator_derivative_evaluations_for_jacobian()
      const final {
    return 0;
  }
  int64_t do_get_num_error_estimator_newton_raphson_iterations() const final {
    return 0;
  }
  int64_t do_get_num_error_estimator_jacobian_evaluations() const final {
    return 0;
  }
  int64_t do_get_num_error_estimator_iteration_matrix_factorizations()
      const final {
    return 0;
  }

  void DoResetImplicitIntegratorStatistics() final {};

  void DoResetCachedJacobianRelatedMatrices() final {};

  void DoInitialize() final;

  std::unique_ptr<ImplicitIntegrator<T>> DoImplicitIntegratorClone()
      const final;

  // Takes a given step of the requested size, if possible.
  // @returns `true` if successful; on `true`, the time and continuous state
  //          will be advanced in the context (e.g., from t0 to t0 + h). On a
  //          `false` return, the time and continuous state in the context will
  //          be restored to its original value (at t0).
  bool DoImplicitIntegratorStep(const T& h) final;

  // For very small h, we'll fall back to an explicit Euler scheme.
  bool ExplicitEulerFallbackStep(const T& h);

  // The iteration matrix G = [I/(h*γ) - J] and its factorization.
  typename ImplicitIntegrator<T>::IterationMatrix iteration_matrix_;

  // Vector used for error estimation
  VectorX<T> error_est_vec_;

  // Intermediate variables to avoid heap allocations
  VectorX<T> x0_, x_, xdot_, k1_, k2_;

  // Track how often it's been since we re-computed the Jacobian
  int num_steps_since_last_jacobian_update_{0};
};

}  // namespace systems
}  // namespace drake
