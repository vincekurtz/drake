#include "drake/examples/discrete_acrobot/acrobot_plant.h"

#include <cmath>
#include <vector>
#include <iostream>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_throw.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/sensors/rotary_encoders.h"

using std::sin;
using std::cos;

namespace drake {
namespace examples {
namespace acrobot {

template <typename T>
AcrobotPlant<T>::AcrobotPlant(double dt)
    : systems::LeafSystem<T>(systems::SystemTypeTag<AcrobotPlant>{}) {
  time_step_ = dt; 
  this->DeclareNumericParameter(AcrobotParams<T>());
  this->DeclareVectorInputPort("elbow_torque", AcrobotInput<T>());
  this->DeclarePeriodicDiscreteUpdateEvent(time_step_, 0, &AcrobotPlant::DiscreteUpdate);
  auto state_index = this->DeclareDiscreteState(AcrobotState<T>());
  this->DeclareStateOutputPort("acrobot_state", state_index);
}

template <typename T>
template <typename U>
AcrobotPlant<T>::AcrobotPlant(const AcrobotPlant<U>& other) : AcrobotPlant<T>(other.time_step()) {}

template <typename T>
void AcrobotPlant<T>::SetMitAcrobotParameters(
    AcrobotParams<T>* parameters) const {
  DRAKE_DEMAND(parameters != nullptr);
  parameters->set_m1(2.4367);
  parameters->set_m2(0.6178);
  parameters->set_l1(0.2563);
  parameters->set_lc1(1.6738);
  parameters->set_lc2(1.5651);
  parameters->set_Ic1(
      -4.7443);  // Notes: Yikes!  Negative inertias (from sysid).
  parameters->set_Ic2(-1.0068);
  parameters->set_b1(0.0320);
  parameters->set_b2(0.0413);
  // Note: parameters are identified in a way that torque has the unit of
  // current (Amps), in order to simplify the implementation of torque
  // constraint on motors. Therefore, some of the numbers here have incorrect
  // units.
}

template <typename T>
Matrix2<T> AcrobotPlant<T>::MassMatrix(
    const systems::Context<T> &context) const {
  const AcrobotState<T>& state = get_state(context);
  const AcrobotParams<T>& p = get_parameters(context);
  const T c2 = cos(state.theta2());
  const T I1 = p.Ic1() + p.m1() * p.lc1() * p.lc1();
  const T I2 = p.Ic2() + p.m2() * p.lc2() * p.lc2();
  const T m2l1lc2 = p.m2() * p.l1() * p.lc2();

  const T m12 = I2 + m2l1lc2 * c2;
  Matrix2<T> M;
  M << I1 + I2 + p.m2() * p.l1() * p.l1() + 2 * m2l1lc2 * c2, m12, m12,
      I2;
  return M;
}

template <typename T>
Vector2<T> AcrobotPlant<T>::DynamicsBiasTerm(const systems::Context<T> &
context)
const {
  const AcrobotState<T>& state = get_state(context);
  const AcrobotParams<T>& p = get_parameters(context);

  const T s1 = sin(state.theta1()), s2 = sin(state.theta2());
  const T s12 = sin(state.theta1() + state.theta2());
  const T m2l1lc2 = p.m2() * p.l1() * p.lc2();

  Vector2<T> bias;
  // C(q,v)*v terms.
  bias << -2 * m2l1lc2 * s2 * state.theta2dot() * state.theta1dot() +
           -m2l1lc2 * s2 * state.theta2dot() * state.theta2dot(),
      m2l1lc2 * s2 * state.theta1dot() * state.theta1dot();

  // -τ_g(q) terms.
  bias(0) += p.gravity() * p.m1() * p.lc1() * s1 +
          p.gravity() * p.m2() * (p.l1() * s1 + p.lc2() * s12);
  bias(1) += p.gravity() * p.m2() * p.lc2() * s12;

  // Damping terms.
  bias(0) += p.b1() * state.theta1dot();
  bias(1) += p.b2() * state.theta2dot();

  return bias;
}

// Compute the actual physics.
template <typename T>
void AcrobotPlant<T>::DiscreteUpdate(
    const systems::Context<T>& context,
    systems::DiscreteValues<T>* new_state) const {

  // Compute current state
  const Vector4<T> x0 = context.get_discrete_state_vector().value();
  const Vector2<T> q0 = x0.template segment<2>(0);
  const Vector2<T> v0 = x0.template segment<2>(2);

  // Compute manipulator dynamics terms, M*v + bias = B*tau
  const T& tau = get_tau(context);
  const Matrix2<T> M = MassMatrix(context);
  const Vector2<T> bias = DynamicsBiasTerm(context);
  const Vector2<T> B(0, 1);  // input matrix

  // Compute next state using symplectic Euler
  // TODO: use factorization instead of inverse
  auto x = new_state->get_mutable_value();
  auto q = x.template segment<2>(0);
  auto v = x.template segment<2>(2);
  v = M.inverse() * ( M * v0 + time_step() * (B * tau - bias) );
  q = q0 + time_step() * v;
}

}  // namespace acrobot
}  // namespace examples
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::examples::acrobot::AcrobotPlant)
