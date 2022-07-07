#pragma once

#include <memory>
#include <vector>

#include "drake/examples/acrobot/gen/acrobot_input.h"
#include "drake/examples/acrobot/gen/acrobot_params.h"
#include "drake/examples/acrobot/gen/acrobot_state.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/affine_system.h"

namespace drake {
namespace examples {
namespace acrobot {

/// A simple solver for the discrete-time acrobot. Solves
///
///  M * (v - v0) = dt * bias
///
/// for v, and stores the factorization of M.
///
/// M is the mass matrix, v0 is the velocity at
/// the previous timestep, dt is the step size, and bias collects
/// all nonlinear terms including external torques.
///
/// This is meant to be a pale imitation of SapSolver.
template <typename T>
class DiscreteAcrobotSolver {
 public:
  // Solve M * (v - v0) = dt * bias for v
  void SolveForwardDynamics(const MatrixX<T>& M, const VectorX<T>& bias,
                            const VectorX<T>& v0, const double dt,
                            EigenPtr<VectorX<T>> v) {
    M_ldlt.compute(M);
    *v = M_ldlt.solve(M * v0 + dt * bias);
  }

  // Solve M * dv_dtheta = - dr_dtheta for dv_dtheta using the factorization
  // of M computed in SolveForwardDynamics.
  void PropagateDerivatives(const MatrixX<T>& dr_dtheta,
                            EigenPtr<MatrixX<T>> dv_dtheta) {
    *dv_dtheta = M_ldlt.solve(-dr_dtheta);
  }

 private:
  // Mass matrix factorization
  Eigen::LDLT<MatrixX<T>> M_ldlt;
};

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
  explicit AcrobotPlant(double dt = 0, bool fancy_gradients = false);

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
  Vector2<T> DynamicsBiasTerm(const systems::Context<T>& context) const;
  Matrix2<T> MassMatrix(const systems::Context<T>& context) const;
  ///@}

  /// Evaluates the input port and returns the scalar value of the commanded
  /// torque.  If the input port is not connected, then the torque is taken to
  /// be zero.
  const T get_tau(const systems::Context<T>& context) const {
    const systems::BasicVector<T>* u_vec = this->EvalVectorInput(context, 0);
    return u_vec ? u_vec->GetAtIndex(0) : 0.0;
  }

  static const AcrobotState<T>& get_state(
      const systems::ContinuousState<T>& cstate) {
    return dynamic_cast<const AcrobotState<T>&>(cstate.get_vector());
  }

  static const AcrobotState<T>& get_state(
      const systems::DiscreteValues<T>& dstate) {
    return dynamic_cast<const AcrobotState<T>&>(dstate.get_vector());
  }

  const AcrobotState<T>& get_state(const systems::Context<T>& context) const {
    if (time_step_ == 0) {
      return get_state(context.get_continuous_state());
    } else {
      return get_state(context.get_discrete_state());
    }
  }

  static AcrobotState<T>& get_mutable_state(
      systems::ContinuousState<T>* cstate) {
    return dynamic_cast<AcrobotState<T>&>(cstate->get_mutable_vector());
  }

  static AcrobotState<T>& get_mutable_state(
      systems::DiscreteValues<T>* dstate) {
    return dynamic_cast<AcrobotState<T>&>(dstate->get_mutable_vector());
  }

  AcrobotState<T>& get_mutable_state(systems::Context<T>* context) {
    if (time_step_ == 0) {
      return get_mutable_state(&context->get_mutable_continuous_state());
    } else {
      return get_mutable_state(&context->get_mutable_discrete_state());
    }
  }

  const AcrobotParams<T>& get_parameters(
      const systems::Context<T>& context) const {
    return this->template GetNumericParameter<AcrobotParams>(context, 0);
  }

  AcrobotParams<T>& get_mutable_parameters(systems::Context<T>* context) const {
    return this->template GetMutableNumericParameter<AcrobotParams>(context, 0);
  }

  double time_step() const { return time_step_; }

 private:
  double time_step_;

  // Flag indicating whether we're overriding gradient computation for
  // AutoDiffXd
  bool fancy_gradients_;

  T DoCalcKineticEnergy(const systems::Context<T>& context) const override;

  T DoCalcPotentialEnergy(const systems::Context<T>& context) const override;

  void DoCalcTimeDerivatives(
      const systems::Context<T>& context,
      systems::ContinuousState<T>* derivatives) const override;

  void DoCalcImplicitTimeDerivativesResidual(
      const systems::Context<T>& context,
      const systems::ContinuousState<T>& proposed_derivatives,
      EigenPtr<VectorX<T>> residual) const override;

  void DoCalcDiscreteVariableUpdates(
      const systems::Context<T>& context,
      const std::vector<const systems::DiscreteUpdateEvent<T>*>& events,
      systems::DiscreteValues<T>* next_state) const override;

  // Helper function for computing discrete updates.
  void DiscreteUpdate(const systems::Context<T>& context,
                      systems::DiscreteValues<T>* next_state) const;

  void CalcResidual(const Matrix2<T>& M, const Vector2<T>& bias,
                    const Vector2<T>& v, const Vector2<T>& v0,
                    EigenPtr<Vector2<T>> r) const;
};

/// Constructs the Acrobot with (only) encoder outputs.
///
/// @system
/// name: AcrobotWEncoder
/// input_ports:
/// - elbow_torque
/// output_ports:
/// - measured_joint_positions
/// - <span style="color:gray">acrobot_state</span>
/// @endsystem
///
/// The `acrobot_state` output port is present only if the construction
/// parameter `acrobot_state_as_second_output` is true.
///
/// @ingroup acrobot_systems
template <typename T>
class AcrobotWEncoder : public systems::Diagram<T> {
 public:
  explicit AcrobotWEncoder(bool acrobot_state_as_second_output = false);

  const AcrobotPlant<T>* acrobot_plant() const { return acrobot_plant_; }

  AcrobotState<T>& get_mutable_acrobot_state(
      systems::Context<T>* context) const;

 private:
  AcrobotPlant<T>* acrobot_plant_{nullptr};
};

/// Constructs the LQR controller for stabilizing the upright fixed point using
/// default LQR cost matrices which have been tested for this system.
/// @ingroup acrobot_systems
std::unique_ptr<systems::AffineSystem<double>> BalancingLQRController(
    const AcrobotPlant<double>& acrobot);

}  // namespace acrobot
}  // namespace examples
}  // namespace drake
