#include "drake/examples/acrobot/acrobot_plant.h"

#include <cmath>
#include <vector>
#include <iostream> // DEBUG

#include "drake/common/default_scalars.h"
#include "drake/common/drake_throw.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/sensors/rotary_encoders.h"
#include "drake/common/autodiff.h"
#include "drake/math/autodiff_gradient.h"

using std::sin;
using std::cos;

namespace drake {
namespace examples {
namespace acrobot {

template <typename T>
AcrobotPlant<T>::AcrobotPlant(double dt, bool fancy_gradients)
    : systems::LeafSystem<T>(systems::SystemTypeTag<AcrobotPlant>{}),
     time_step_(dt), fancy_gradients_(fancy_gradients) {
  DRAKE_DEMAND(dt >= 0);

  if (time_step_ == 0) {
    // Continuous time system
    auto state_index = this->DeclareContinuousState(
        AcrobotState<T>(), 2 /* num_q */, 2 /* num_v */, 0 /* num_z */);
    this->DeclareStateOutputPort("acrobot_state", state_index);
  } else {
    // Discrete time system
    auto state_index = this->DeclareDiscreteState(AcrobotState<T>());
    this->DeclareStateOutputPort("acrobot_state", state_index);
    this->DeclarePeriodicDiscreteUpdate(time_step_, 0);
  }
  this->DeclareNumericParameter(AcrobotParams<T>());
  this->DeclareVectorInputPort("elbow_torque", AcrobotInput<T>());
}

template <typename T>
template <typename U>
AcrobotPlant<T>::AcrobotPlant(const AcrobotPlant<U>& other) :
  AcrobotPlant<T>(other.time_step()) {}

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
void AcrobotPlant<T>::DoCalcTimeDerivatives(
    const systems::Context<T>& context,
    systems::ContinuousState<T>* derivatives) const {
  const AcrobotState<T>& state = get_state(context);
  const T& tau = get_tau(context);

  const Matrix2<T> M = MassMatrix(context);
  const Vector2<T> bias = DynamicsBiasTerm(context);
  const Vector2<T> B(0, 1);  // input matrix

  Vector4<T> xdot;
  xdot << state.theta1dot(), state.theta2dot(),
          M.inverse() * (B * tau - bias);
  derivatives->SetFromVector(xdot);
}

template <typename T>
void AcrobotPlant<T>::DoCalcImplicitTimeDerivativesResidual(
    const systems::Context<T>& context,
    const systems::ContinuousState<T>& proposed_derivatives,
    EigenPtr<VectorX<T>> residual) const {
  DRAKE_DEMAND(residual != nullptr);
  const AcrobotState<T>& state = get_state(context);
  const T& tau = get_tau(context);

  const Matrix2<T> M = MassMatrix(context);
  const Vector2<T> bias = DynamicsBiasTerm(context);
  const Vector2<T> B(0, 1);  // input matrix

  const auto& proposed_qdot = proposed_derivatives.get_generalized_position();
  const auto proposed_vdot =
      proposed_derivatives.get_generalized_velocity().CopyToVector();

  *residual << proposed_qdot[0] - state.theta1dot(),
               proposed_qdot[1] - state.theta2dot(),
               M * proposed_vdot - (B * tau - bias);
}


template <typename T>
void AcrobotPlant<T>::DiscreteUpdate(
    const systems::Context<T>& context,
    systems::DiscreteValues<T>* new_state) const {

  // Compute current state
  const Vector4<T>& x0 = context.get_discrete_state_vector().value();
  const auto q0 = x0.template segment<2>(0);
  const auto v0 = x0.template segment<2>(2);

  // Compute manipulator dynamics terms, M*v = bias
  const T& tau = get_tau(context);
  const Matrix2<T> M = MassMatrix(context);
  const Vector2<T> B(0, 1);  // input matrix
  const Vector2<T> bias = B * tau - DynamicsBiasTerm(context);

  // Compute next state using symplectic Euler
  Eigen::VectorBlock<VectorX<T>> x = new_state->get_mutable_value();
  auto q = x.template segment<2>(0);
  auto v = x.template segment<2>(2);

  DiscreteAcrobotSolver<T> solver;
  solver.SolveForwardDynamics(M, bias, v0, time_step_, &v);
  q = q0 + time_step() * v;
}

template <typename T>
void AcrobotPlant<T>::CalcResidual(
    const Matrix2<T>& M,
    const Vector2<T>& bias,
    const Vector2<T>& v,
    const Vector2<T>& v0,
    EigenPtr<Vector2<T>> r
) const {
  *r = M * (v - v0) - time_step() * bias;
}

template <typename T>
void AcrobotPlant<T>::DoCalcDiscreteVariableUpdates(
    const systems::Context<T>& context,
    const std::vector< const systems::DiscreteUpdateEvent<T>*>&,
    systems::DiscreteValues<T>* new_state) const {
  DiscreteUpdate(context, new_state);
}

// Discrete update override for T=AutoDiffXd
template <>
void AcrobotPlant<AutoDiffXd>::DoCalcDiscreteVariableUpdates(
    const systems::Context<AutoDiffXd>& context,
    const std::vector< const systems::DiscreteUpdateEvent<AutoDiffXd>*>&,
    systems::DiscreteValues<AutoDiffXd>* new_state) const {

  if ( fancy_gradients_ ) {
    // Compute current state
    const Vector4<AutoDiffXd>& x0 = context.get_discrete_state_vector().value();
    const auto q0 = x0.template segment<2>(0);
    const auto v0 = x0.template segment<2>(2);

    // Compute dynamics terms with autodiff
    const Matrix2<AutoDiffXd> M = MassMatrix(context);
    const AutoDiffXd& tau = get_tau(context);
    const Vector2<AutoDiffXd> B(0, 1);  // input matrix
    const Vector2<AutoDiffXd> bias = B * tau - DynamicsBiasTerm(context);
    
    // Compute forward dynamics and factorization with double
    const Matrix2<double> M_double = math::ExtractValue(M);
    const Vector2<double> bias_double = math::ExtractValue(bias);
    const Vector2<double> v0_double = math::ExtractValue(v0);
    const Vector2<double> q0_double = math::ExtractValue(q0);

    VectorX<double> x_double(4);
    auto q_double = x_double.template segment<2>(0);
    auto v_double = x_double.template segment<2>(2);
    DiscreteAcrobotSolver<double> solver;
    solver.SolveForwardDynamics(M_double, bias_double, v0_double, time_step_, &v_double);
    q_double = q0_double + time_step() * v_double;

    // Compute the gradient of the residual via autodiff
    Vector2<AutoDiffXd> r;
    Vector2<AutoDiffXd> v(v_double);
    CalcResidual(M, bias, v, v0, &r);
    MatrixX<double> dr_dtheta = math::ExtractGradient(r);

    // Use implicit function theorem to compute gradients
    MatrixX<double> dx_dtheta(4, dr_dtheta.cols());
    auto dq_dtheta = dx_dtheta.template topRows<2>();
    auto dv_dtheta = dx_dtheta.template bottomRows<2>();
    solver.PropagateDerivatives(dr_dtheta, &dv_dtheta);
    dq_dtheta = math::ExtractGradient(q0) + time_step() * dv_dtheta;

    // Load gradients and values back into the result
    auto new_x = new_state->get_mutable_value();
    math::InitializeAutoDiff(x_double, dx_dtheta, &new_x);

  } else {
    // Just do things normally with AutoDiffXd
    DiscreteUpdate(context, new_state);
  }
}


template <typename T>
T AcrobotPlant<T>::DoCalcKineticEnergy(
    const systems::Context<T>& context) const {
  const AcrobotState<T>& state = dynamic_cast<const AcrobotState<T>&>(
      context.get_continuous_state_vector());

  Matrix2<T> M = MassMatrix(context);
  Vector2<T> qdot(state.theta1dot(), state.theta2dot());

  return 0.5 * qdot.transpose() * M * qdot;
}

template <typename T>
T AcrobotPlant<T>::DoCalcPotentialEnergy(
    const systems::Context<T>& context) const {
  const AcrobotState<T>& state = get_state(context);
  const AcrobotParams<T>& p = get_parameters(context);

  using std::cos;
  const T c1 = cos(state.theta1());
  const T c12 = cos(state.theta1() + state.theta2());

  return -p.m1() * p.gravity() * p.lc1() * c1 -
         p.m2() * p.gravity() * (p.l1() * c1 + p.lc2() * c12);
}

template <typename T>
AcrobotWEncoder<T>::AcrobotWEncoder(bool acrobot_state_as_second_output) {
  systems::DiagramBuilder<T> builder;

  acrobot_plant_ = builder.template AddSystem<AcrobotPlant<T>>();
  acrobot_plant_->set_name("acrobot_plant");
  auto encoder =
      builder.template AddSystem<systems::sensors::RotaryEncoders<T>>(
          4, std::vector<int>{0, 1});
  encoder->set_name("encoder");
  builder.Cascade(*acrobot_plant_, *encoder);
  builder.ExportInput(acrobot_plant_->get_input_port(0), "elbow_torque");
  builder.ExportOutput(encoder->get_output_port(), "measured_joint_positions");
  if (acrobot_state_as_second_output)
    builder.ExportOutput(acrobot_plant_->get_output_port(0), "acrobot_state");

  builder.BuildInto(this);
}

template <typename T>
AcrobotState<T>& AcrobotWEncoder<T>::get_mutable_acrobot_state(
    systems::Context<T>* context) const {
  AcrobotState<T>* x = dynamic_cast<AcrobotState<T>*>(
      &this->GetMutableSubsystemContext(*acrobot_plant_, context)
           .get_mutable_continuous_state_vector());
  DRAKE_DEMAND(x != nullptr);
  return *x;
}

std::unique_ptr<systems::AffineSystem<double>> BalancingLQRController(
    const AcrobotPlant<double>& acrobot) {
  auto context = acrobot.CreateDefaultContext();

  // Set nominal torque to zero.
  acrobot.GetInputPort("elbow_torque").FixValue(context.get(), 0.0);

  // Set nominal state to the upright fixed point.
  AcrobotState<double>* x = dynamic_cast<AcrobotState<double>*>(
      &context->get_mutable_continuous_state_vector());
  DRAKE_ASSERT(x != nullptr);
  x->set_theta1(M_PI);
  x->set_theta2(0.0);
  x->set_theta1dot(0.0);
  x->set_theta2dot(0.0);

  // Setup LQR Cost matrices (penalize position error 10x more than velocity
  // to roughly address difference in units, using sqrt(g/l) as the time
  // constant.
  Eigen::Matrix4d Q = Eigen::Matrix4d::Identity();
  Q(0, 0) = 10;
  Q(1, 1) = 10;
  Vector1d R = Vector1d::Constant(1);

  return systems::controllers::LinearQuadraticRegulator(acrobot, *context, Q,
                                                        R);
}

}  // namespace acrobot
}  // namespace examples
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::examples::acrobot::AcrobotPlant)

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::examples::acrobot::AcrobotWEncoder)
