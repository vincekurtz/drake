#include "drake/systems/analysis/exponential_rosenbrock_integrator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <unsupported/Eigen/MatrixFunctions>

#include "drake/common/drake_assert.h"
#include "drake/common/fmt_eigen.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace systems {

template <class T>
ExponentialRosenbrockIntegrator<T>::~ExponentialRosenbrockIntegrator() =
    default;

template <class T>
std::unique_ptr<ImplicitIntegrator<T>>
ExponentialRosenbrockIntegrator<T>::DoImplicitIntegratorClone() const {
  return std::make_unique<ExponentialRosenbrockIntegrator>(
      this->get_system(), this->get_maximum_step_size());
}

template <class T>
void ExponentialRosenbrockIntegrator<T>::DoInitialize() {
  using std::isnan;

  if (isnan(this->get_maximum_step_size()))
    throw std::logic_error("Maximum step size has not been set! ");

  // Allocate intermediate variables
  const int nx = this->get_system().num_continuous_states();
  x0_.resize(nx);
  x_.resize(nx);
  xdot_.resize(nx);
  exponential_matrix_.resize(nx + 1, nx + 1);
}

template <class T>
bool ExponentialRosenbrockIntegrator<T>::DoImplicitIntegratorStep(const T& h) {
  using std::sqrt;

  // Store the current time and state
  Context<T>* context = this->get_mutable_context();
  const T t0 = context->get_time();
  x0_ = context->get_continuous_state().CopyToVector();

  // Make sure everything is the correct size
  const int n = x0_.size();
  x_.resize(n);
  xdot_.resize(n);
  exponential_matrix_.resize(n + 1, n + 1);

  // Compute the Jacobian J = ∂/∂x f(x₀)
  const MatrixX<T>& J = this->CalcJacobian(t0, x0_);

  // Compute xdot = f(x₀)
  xdot_ = this->EvalTimeDerivatives(*context).CopyToVector();

  // Compute δt φ₁(δt A) f(x), where φ₁(z) = ∫₀¹ exp((1−θ)z)dθ, following
  // https://math.stackexchange.com/questions/4170258/how-to-apply-phi-functions-to-matrices
  exponential_matrix_.setZero();
  exponential_matrix_.block(0, 0, n, n) = J;      // top-left block is J
  exponential_matrix_.block(0, n, n, 1) = xdot_;  // last column is f(x₀)
  exponential_matrix_ *= h;
  MatrixX<T> expX = exponential_matrix_.exp();
  x_ = expX.block(0, n, n, 1);  // last column holds δt φ₁(δt A) f(x₀)

  // Advance the time and state:
  //    t = t₀ + h
  //    x = x₀ + δt φ₁(δt A) f(x)
  context->SetTime(t0 + h);
  x_ += x0_;
  context->get_mutable_continuous_state().SetFromVector(x_);

  return true;  // step was successful
}

}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class drake::systems::ExponentialRosenbrockIntegrator);
