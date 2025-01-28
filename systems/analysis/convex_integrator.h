#pragma once

#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/integrator_base.h"

namespace drake {
namespace systems {

using multibody::MultibodyPlant;

/**
 * An experimental implicit integrator that solves a convex SAP problem to
 * advance the state, rather than relying on non-convex Newton-Raphson.
 *
 * @tparam_default_scalar
 * @ingroup integrators
 */
template <class T>
class ConvexIntegrator final : public IntegratorBase<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConvexIntegrator);

  ~ConvexIntegrator() override = default;

  /**
   * Constructs the experimental convex integrator. For now this only supports
   * a simple MultibodyPlant, unconnected to anything else.
   *
   * @param system A reference to the system to be simulated
   * @param max_step_size The maximum (fixed) step size; the integrator will
   *                      not take larger step sizes than this.
   * @param context Pointer to the context (nullptr is ok, but the caller
   *                must set a non-null context before Initialize()-ing the
   *                integrator).
   * @sa Initialize()
   */
  ConvexIntegrator(const System<T>& system, const T& max_step_size,
                   Context<T>* context = nullptr)
      : IntegratorBase<T>(system, context) {
    IntegratorBase<T>::set_maximum_step_size(max_step_size);

    // Check that the system we're simulating is a diagram with a plant in it
    const Diagram<T>* diagram = dynamic_cast<const Diagram<T>*>(&system);
    DRAKE_DEMAND(diagram != nullptr);

    plant_ = dynamic_cast<const MultibodyPlant<T>*>(
        &diagram->GetSubsystemByName("plant"));
    DRAKE_DEMAND(plant_ != nullptr);

    // TODO(vincekurtz): consider additional checks to ensure that there are no
    // significant systems other than the plant in the diagram.
  }

  // TODO(vincekurtz): add error estimation
  bool supports_error_estimation() const override { return false; }

  // TODO(vincekurtz): add error estimation
  int get_error_estimate_order() const override { return 0; }

  const MultibodyPlant<T>& plant() const {return *plant_; }

 private:
  // The primary integration step, sets x_{t+h}
  bool DoStep(const T& h) override;

  // Plant model, since convex integration is specific to MbP
  const MultibodyPlant<T>* plant_;
};

}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::systems::ConvexIntegrator);
