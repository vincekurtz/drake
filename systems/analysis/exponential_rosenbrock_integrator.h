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
  // This integrator is templated on ImplicitIntegrator to gain access to the
  // Jacobian computation abilities, but does not take any Newton iterations.
  int64_t do_get_num_newton_raphson_iterations() const final { return 0; }

  // Error estimation is not supported.
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
  bool DoImplicitIntegratorStep(const T& h) final;

  // Intermediate variables to avoid heap allocations
  VectorX<T> x0_, x_, xdot_;

  // Helper variable for computing the φ₁ function via matrix exponentiation.
  MatrixX<T> exponential_matrix_;

};

}  // namespace systems
}  // namespace drake
