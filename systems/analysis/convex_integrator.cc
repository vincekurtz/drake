#include "drake/systems/analysis/convex_integrator.h"

namespace drake {
namespace systems {

/**
 * Integrates the system forward in time by h, starting at the current time t₀.
 * This value of h is determined by IntegratorBase::Step().
 */
template <class T>
bool ConvexIntegrator<T>::DoStep(const T& h) {
  Context<T>& context = *this->get_mutable_context();

  // CAUTION: This is performance-sensitive inner loop code that uses dangerous
  // long-lived references into state and cache to avoid unnecessary copying and
  // cache invalidation. Be careful not to insert calls to methods that could
  // invalidate any of these references before they are used.

  // Evaluate derivative xcdot₀ ← xcdot(t₀, x(t₀), u(t₀)).
  const ContinuousState<T>& xc_deriv = this->EvalTimeDerivatives(context);
  const VectorBase<T>& xcdot0 = xc_deriv.get_vector();

  // Cache: xcdot0 references the live derivative cache value, currently
  // up to date but about to be marked out of date. We do not want to make
  // an unnecessary copy of this data.

  // Update continuous state and time. This call marks t- and xc-dependent
  // cache entries out of date, including xcdot0.
  VectorBase<T>& xc = context.SetTimeAndGetMutableContinuousStateVector(
      context.get_time() + h);  // t ← t₀ + h

  // Cache: xcdot0 still references the derivative cache value, which is
  // unchanged, although it is marked out of date.

  xc.PlusEqScaled(h, xcdot0);  // xc(t₀ + h) ← xc(t₀) + h * xcdot₀

  // This integrator always succeeds at taking the step.
  return true;
}

}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::systems::ConvexIntegrator);
