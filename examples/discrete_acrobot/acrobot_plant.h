#pragma once

#include <memory>

#include "drake/examples/discrete_acrobot/gen/acrobot_input.h"
#include "drake/examples/discrete_acrobot/gen/acrobot_params.h"
#include "drake/examples/discrete_acrobot/gen/acrobot_state.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/affine_system.h"

namespace drake {
namespace examples {
namespace acrobot {

/// @defgroup acrobot_systems Acrobot
/// @{
/// @brief Systems related to the Acrobot example.
/// @ingroup example_systems
/// @}

/// The Acrobot - a canonical underactuated system as described in <a
/// href="http://underactuated.mit.edu/underactuated.html?chapter=3">Chapter 3
/// of Underactuated Robotics</a>.
///
/// @system
/// name: AcrobotPlant
/// input_ports:
/// - elbow_torque (optional)
/// output_ports:
/// - acrobot_state
/// @endsystem
///
/// Note: If the elbow_torque input port is not connected, then the torque is
/// taken to be zero.
///
/// @tparam_default_scalar
/// @ingroup acrobot_systems
template <typename T>
class AcrobotPlant : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(AcrobotPlant)

  /// Constructs the plant.  The parameters of the system are stored as
  /// Parameters in the Context (see acrobot_params_named_vector.yaml).
  AcrobotPlant(double dt);

  /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit AcrobotPlant(const AcrobotPlant<U>&);

  /// Sets the parameters to describe MIT Robot Locomotion Group's hardware
  /// acrobot.
  void SetMitAcrobotParameters(AcrobotParams<T>* parameters) const;

  ///@{
  /// Manipulator equation of Acrobot: M(q)q̈ + bias(q,q̇) = B*u.
  ///
  /// - M[2x2] is the mass matrix.
  /// - bias[2x1] includes the Coriolis term, gravity term and the damping term,
  ///   i.e. bias[2x1] = C(q,v)*v - τ_g(q) + [b1*q̇₁;b2*q̇₂].
  // TODO(russt): Update this to the newest conventions.
  Vector2<T> DynamicsBiasTerm(const systems::Context<T> &context) const;
  Matrix2<T> MassMatrix(const systems::Context<T> &context) const;
  ///@}

  /// Evaluates the input port and returns the scalar value of the commanded
  /// torque.  If the input port is not connected, then the torque is taken to
  /// be zero.
  const T get_tau(const systems::Context<T>& context) const {
    const systems::BasicVector<T>* u_vec = this->EvalVectorInput(context, 0);
    return u_vec ? u_vec->GetAtIndex(0) : 0.0;
  }

  static const AcrobotState<T>& get_state(
      const systems::DiscreteValues<T>& dstate) {
    return dynamic_cast<const AcrobotState<T>&>(dstate.get_vector());
  }

  static const AcrobotState<T>& get_state(const systems::Context<T>& context) {
    return get_state(context.get_discrete_state());
  }

  static AcrobotState<T>& get_mutable_state(
      systems::DiscreteValues<T>* dstate) {
    return dynamic_cast<AcrobotState<T>&>(dstate->get_mutable_vector());
  }

  static AcrobotState<T>& get_mutable_state(systems::Context<T>* context) {
    return get_mutable_state(&context->get_mutable_discrete_state());
  }

  const AcrobotParams<T>& get_parameters(
      const systems::Context<T>& context) const {
    return this->template GetNumericParameter<AcrobotParams>(context, 0);
  }

  AcrobotParams<T>& get_mutable_parameters(systems::Context<T>* context) const {
    return this->template GetMutableNumericParameter<AcrobotParams>(context, 0);
  }

  double time_step() const {
    return time_step_;
  }

 private:
  double time_step_ = 0.1;

  void DiscreteUpdate(
      const systems::Context<T>& context,
      systems::DiscreteValues<T>* next_state) const;

};

}  // namespace acrobot
}  // namespace examples
}  // namespace drake
