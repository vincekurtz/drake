#pragma once

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/profiler.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/inverse_dynamics_partials.h"
#include "drake/traj_opt/penta_diagonal_matrix.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/solver_parameters.h"
#include "drake/traj_opt/trajectory_optimizer_solution.h"
#include "drake/traj_opt/trajectory_optimizer_state.h"
#include "drake/traj_opt/trajectory_optimizer_workspace.h"
#include "drake/traj_opt/velocity_partials.h"

namespace drake {
namespace systems {
// Forward declaration to avoid polluting this namespace with systems:: stuff.
template <typename>
class Diagram;
}  // namespace systems

namespace traj_opt {

using internal::PentaDiagonalMatrix;
using multibody::MultibodyPlant;
using systems::Context;
using systems::Diagram;

template <typename T>
class TrajectoryOptimizer {
 public:
  /**
   * Construct a new Trajectory Optimizer object.
   *
   * @param plant A model of the system that we're trying to find an optimal
   *              trajectory for.
   * @param context A context for the plant, used to perform various multibody
   *                dynamics computations. Should be part of a larger Diagram
   *                context, and be connected to a scene graph.
   * @param prob Problem definition, including cost, initial and target states,
   *             etc.
   * @param params solver parameters, including max iterations, linesearch
   *               method, etc.
   */
  // TODO(amcastro-tri): Get rid of this constructor. Favor the new construction
  // below so that we can cache the context at each time step in the state.
  // In particular, context only gets used for
  // CalcInverseDynamicsPartialsFiniteDiff().
  TrajectoryOptimizer(const MultibodyPlant<T>* plant, Context<T>* context,
                      const ProblemDefinition& prob,
                      const SolverParameters& params = SolverParameters{});

  /**
   * Construct a new Trajectory Optimizer object.
   *
   * @param diagram Diagram for the entire model that will include the plant and
   * SceneGraph for geometric queries. Used to allocate context resources.
   * @param plant A model of the system that we're trying to find an optimal
   *              trajectory for.
   * @param prob Problem definition, including cost, initial and target states,
   *             etc.
   * @param params solver parameters, including max iterations, linesearch
   *               method, etc.
   */
  TrajectoryOptimizer(const Diagram<T>* diagram, const MultibodyPlant<T>* plant,
                      const ProblemDefinition& prob,
                      const SolverParameters& params = SolverParameters{});

  /**
   * Convienience function to get the timestep of this optimization problem.
   *
   * @return double dt, the time step for this optimization problem
   */
  double time_step() const { return plant_->time_step(); }

  /**
   * Convienience function to get the time horizon (T) of this optimization
   * problem.
   *
   * @return int the number of time steps in the optimal trajectory.
   */
  int num_steps() const { return prob_.num_steps; }

  /**
   * Convienience function to get a const reference to the multibody plant that
   * we are optimizing over.
   *
   * @return const MultibodyPlant<T>&, the plant we're optimizing over.
   */
  const MultibodyPlant<T>& plant() const { return *plant_; }

  /**
   * Create a state object which contains the decision variables (generalized
   * positions at each timestep), along with a cache of other things that are
   * computed from positions, such as velocities, accelerations, forces, and
   * various derivatives.
   *
   * @return TrajectoryOptimizerState
   */
  TrajectoryOptimizerState<T> CreateState() const {
    INSTRUMENT_FUNCTION("Creates state object with caching.");
    if (diagram_ != nullptr) {
      return TrajectoryOptimizerState<T>(num_steps(), *diagram_, plant());
    }
    return TrajectoryOptimizerState<T>(num_steps(), plant());
  }

  /**
   * Compute the gradient of the unconstrained cost L(y).
   *
   * @param state optimizer state, including q, v, tau, k, gradients, etc.
   * @param g a single VectorXd containing the partials of L w.r.t. each
   *          decision variable (y_t[i]).
   */
  void CalcGradient(const TrajectoryOptimizerState<T>& state,
                    EigenPtr<VectorX<T>> g) const;

  /**
   * Compute the Hessian of the unconstrained cost L(y) as a sparse
   * penta-diagonal matrix.
   *
   * @param state optimizer state, including q, v, tau, k, gradients, etc.
   * @param H a PentaDiagonalMatrix containing the second-order derivatives of
   *          the total cost L(y). This matrix is composed of (num_steps+1 x
   *          num_steps+1) blocks of size (ny x ny) each.
   */
  void CalcHessian(const TrajectoryOptimizerState<T>& state,
                   PentaDiagonalMatrix<T>* H) const;

  /**
   * Solve the optimization from the given initial guess, which may or may not
   * be dynamically feasible.
   *
   * @param y_guess a sequence of generalized positions and virtual force
   * stiffnesses corresponding to the initial guess
   * @param solution a container for the optimal solution, including velocities
   * and torques
   * @param stats a container for other timing and iteration-specific
   * data regarding the solve process.
   * @return SolverFlag
   */
  SolverFlag Solve(const std::vector<VectorX<T>>& y_guess,
                   TrajectoryOptimizerSolution<T>* solution,
                   TrajectoryOptimizerStats<T>* stats,
                   ConvergenceReason* reason = nullptr) const;

  // The following evaluator functions get data from the state's cache, and
  // update it if necessary.

  /**
   * Evaluate generalized positions q_t at each timestep t, t = [0, ...,
   * num_steps()].
   *
   * @param state optimizer state
   * @return const std::vector<VectorX<T>>& q_t
   */
  const std::vector<VectorX<T>>& EvalQ(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate virtual force parameters k_t at each timestep t, t = [0, ...,
   * num_steps()]. k_t=0 corresponds to no virtual force (no force at a
   * distance), while a large k_t corresponds to large virtual forces.
   *
   * @param state optimizer state
   * @return const std::vector<VectorX<T>>& k_t
   */
  const std::vector<VectorX<T>>& EvalK(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate generalized velocities
   *
   *    v_t = (q_t - q_{t-1}) / dt
   *
   * at each timestep t, t = [0, ..., num_steps()],
   *
   * where v_0 is fixed by the initial condition.
   *
   * @param state optimizer state
   * @return const std::vector<VectorX<T>>& v_t
   */
  const std::vector<VectorX<T>>& EvalV(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate generalized accelerations
   *
   *    a_t = (v_{t+1} - v_t) / dt
   *
   * at each timestep t, t = [0, ..., num_steps()-1].
   *
   * @param state optimizer state
   * @return const std::vector<VectorX<T>>& a_t
   */
  const std::vector<VectorX<T>>& EvalA(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate generalized forces
   *
   *    τ_t = ID(q_{t+1}, v_{t+1}, a_t) - J(q_{t+1})'γ(q_{t+1},v_{t+1})
   *          - J(q_{t+1})'f_v(q_{t+1}, k_t),
   *
   * where ID(q,v,a) are the inverse dynamics, J(q) is the contact Jacobian, γ
   * are contact forces, and f_v are virtual forces, at each timestep t, t = [0,
   * ..., num_steps()-1].
   *
   * @param state optimizer state
   * @return const std::vector<VectorX<T>>& τ_t
   */
  const std::vector<VectorX<T>>& EvalTau(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate partial derivatives of velocites with respect to the decision
   * variables y = [q,k] at each time step.
   *
   * @param state optimizer state
   * @return const VelocityPartials<T>& container for ∂v/∂y
   */
  const VelocityPartials<T>& EvalVelocityPartials(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the mapping from qdot to v, v = N+(q)*qdot, at each time step.
   *
   * @param state optimizer state
   * @return const std::vector<MatrixX<T>>& N+(q_t) for each time step t.
   */
  const std::vector<MatrixX<T>>& EvalNplus(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate partial derivatives of generalized forces with respect to
   * the decision variables y = [q,k] at each time step.
   *
   * @param state optimizer state
   * @return const InverseDynamicsPartials<T>& container for ∂τ/∂y
   */
  const InverseDynamicsPartials<T>& EvalInverseDynamicsPartials(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the total (unconstrained) cost of the optimization problem,
   *
   *     L(y) = x_err(T)'*Qf*x_err(T)
   *                + dt*sum_{t=0}^{T-1} x_err(t)'*Q*x_err(t) + u(t)'*R*u(t)
   *                + k(t)'*Rv*k(t),
   *
   * where:
   *      x_err(t) = x(t) - x_nom is the state error,
   *      T = num_steps() is the time horizon of the optimization problem,
   *      x(t) = [q(t); v(t)] is the system state at time t,
   *      u(t) are control inputs, and we assume (for now) that u(t) = tau(t),
   *      Q{f} = diag([Qq{f}, Qv{f}]) are a block diagonal PSD state-error
   *       weighting matrices,
   *      R is a PSD control weighting matrix.
   *      k(t) are virtual force parameters: k(t)=0 corresponds to no virtual
   *       force,
   *      Rv is a PSD penalty on the virtual force parameters
   *
   * A cached version of this cost is stored in the state. If the cache is up to
   * date, simply return that cost.
   *
   * @param state optimizer state
   * @return const double, total cost
   */
  const T EvalCost(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the Hessian of the unconstrained cost L(y) as a sparse
   * penta-diagonal matrix.
   *
   * @param state optimizer state, including q, v, tau, k, gradients, etc.
   * @return const PentaDiagonalMatrix<T>& the second-order derivatives of
   *          the total cost L(y). This matrix is composed of (num_steps+1 x
   *          num_steps+1) blocks of size (ny x ny) each.
   */
  const PentaDiagonalMatrix<T>& EvalHessian(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the gradient of the unconstrained cost L(y).
   *
   * @param state optimizer state, including q, v, tau, k, gradients, etc.
   * @return const VectorX<T>& a single vector containing the partials of L
   * w.r.t. each decision variable (y_t[i]).
   */
  const VectorX<T>& EvalGradient(
      const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate a system context for the plant at the given time step.
   *
   * @param state optimizer state
   * @param t time step
   * @return const Context<T>& context for the plant at time t
   */
  const Context<T>& EvalPlantContext(const TrajectoryOptimizerState<T>& state,
                                     int t) const;

  /**
   * Evaluate signed distance pairs for each potential contact pair at the given
   * time step
   *
   * @param state optimizer state
   * @param t time step
   * @return const std::vector<geometry::SignedDistancePair<T>>& contact
   * geometry information for each contact pair at time t
   */
  const std::vector<geometry::SignedDistancePair<T>>& EvalSignedDistancePairs(
      const TrajectoryOptimizerState<T>& state, int t) const;

  /**
   * Evaluate contact jacobians (includes all contact pairs) at each time step.
   *
   * @param state optimizer state
   * @return const TrajectoryOptimizerCache<T>::ContactJacobianData& contact
   * jacobian data
   */
  const typename TrajectoryOptimizerCache<T>::ContactJacobianData&
  EvalContactJacobianData(const TrajectoryOptimizerState<T>& state) const;

 private:
  // Friend class to facilitate testing.
  friend class TrajectoryOptimizerTester;

  // Allow different specializations to access each other's private functions.
  // In particular we want to allow TrajectoryOptimizer<double> to have access
  // to TrajectoryOptimizer<AutoDiffXd>'s functions for computing gradients.
  template <typename U>
  friend class TrajectoryOptimizer;

  /**
   * Solve the optimization problem from the given initial guess using a
   * linesearch strategy.
   *
   * @param y_guess a sequence of generalized positions and virtual force
   * parameters corresponding to the initial guess
   * @param solution a container for the optimal solution, including velocities
   * and torques
   * @param stats a container for other timing and iteration-specific
   * data regarding the solve process.
   * @return SolverFlag
   */
  SolverFlag SolveWithLinesearch(const std::vector<VectorX<T>>& y_guess,
                                 TrajectoryOptimizerSolution<T>* solution,
                                 TrajectoryOptimizerStats<T>* stats) const;

  /**
   * Solve the optimization problem from the given initial guess using a trust
   * region strategy.
   *
   * @param y_guess a sequence of generalized positions and virtual force
   * parameters corresponding to the initial guess
   * @param solution a container for the optimal solution, including velocities
   * and torques
   * @param stats a container for other timing and iteration-specific
   * data regarding the solve process.
   * @return SolverFlag
   */
  SolverFlag SolveWithTrustRegion(const std::vector<VectorX<T>>& y_guess,
                                  TrajectoryOptimizerSolution<T>* solution,
                                  TrajectoryOptimizerStats<T>* stats,
                                  ConvergenceReason* reason) const;

  // Updates `cache` to store q and v from `state`.
  void CalcContextCache(
      const TrajectoryOptimizerState<T>& state,
      typename TrajectoryOptimizerCache<T>::ContextCache* cache) const;

  /**
   * Compute all of the "trajectory data" (velocities v, accelerations a,
   * torques tau) in the state's cache to correspond to the state's generalized
   * positions q and virtual force parameters k.
   *
   * @param state optimizer state to update.
   */
  void CalcCacheTrajectoryData(const TrajectoryOptimizerState<T>& state) const;

  void CalcInverseDynamicsCache(
      const TrajectoryOptimizerState<T>& state,
      typename TrajectoryOptimizerCache<T>::InverseDynamicsCache* cache) const;

  /**
   * Compute all of the "derivatives data" (dv/dy, dtau/dy) stored in the
   * state's cache to correspond to the state's generalized positions q and
   * virtual force parameters k.
   *
   * @param state optimizer state to update.
   */
  void CalcCacheDerivativesData(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Return the total (unconstrained) cost of the optimization problem,
   *
   *     L(y) = x_err(T)'*Qf*x_err(T)
   *                + dt*sum_{t=0}^{T-1} x_err(t)'*Q*x_err(t) + u(t)'*R*u(t)
   *                + k(t)'*Rv*k(t)
   *
   * where:
   *      x_err(t) = x(t) - x_nom is the state error,
   *      T = num_steps() is the time horizon of the optimization problem,
   *      x(t) = [q(t); v(t)] is the system state at time t,
   *      u(t) are control inputs, and we assume (for now) that u(t) = tau(t),
   *      Q{f} = diag([Qq{f}, Qv{f}]) are a block diagonal PSD state-error
   *       weighting matrices,
   *      R is a PSD control weighting matrix,
   *      k(t) are virtual force parameters: k(t)=0 corresponds to no virtual
   *       force,
   *      Rv is a PSD penalty on the virtual force parameters.
   *
   * A cached version of this cost is stored in the state. If the cache is up to
   * date, simply return that cost.
   *
   * @param state optimizer state
   * @return double, total cost
   */
  T CalcCost(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Compute the total cost of the unconstrained problem (with virtual forces).
   *
   * @param q sequence of generalized positions
   * @param k sequence of virtual force parameters
   * @param v sequence of generalized velocities (consistent with q)
   * @param tau sequence of generalized forces (consistent with q, k, and v)
   * @param workspace scratch space for intermediate computations
   * @return double, total cost
   */
  T CalcCost(const std::vector<VectorX<T>>& q, const std::vector<VectorX<T>>& k,
             const std::vector<VectorX<T>>& v,
             const std::vector<VectorX<T>>& tau,
             TrajectoryOptimizerWorkspace<T>* workspace) const;

  /**
   * Compute the total cost of the unconstrained problem (without virtual
   * forces).
   *
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities (consistent with q)
   * @param tau sequence of generalized forces (consistent with q, k, and v)
   * @param workspace scratch space for intermediate computations
   * @return double, total cost
   */
  T CalcCost(const std::vector<VectorX<T>>& q,
             const std::vector<VectorX<T>>& v,
             const std::vector<VectorX<T>>& tau,
             TrajectoryOptimizerWorkspace<T>* workspace) const;

  /**
   * Compute a sequence of generalized velocities v from a sequence of
   * generalized positions, where
   *
   *     v_t = N+(q_t) * (q_t - q_{t-1}) / dt            (1)
   *
   * v and q are each vectors of length num_steps+1,
   *
   *     v = [v(0), v(1), v(2), ..., v(num_steps)],
   *     q = [q(0), q(1), q(2), ..., q(num_steps)].
   *
   * Note that v0 = v_init is defined by the initial state of the optimization
   * problem, rather than Equation (1) above.
   *
   * @param q sequence of generalized positions
   * @param Nplus the mapping from qdot to v, N+(q_t).
   * @param v sequence of generalized velocities
   */
  void CalcVelocities(const std::vector<VectorX<T>>& q,
                      const std::vector<MatrixX<T>>& Nplus,
                      std::vector<VectorX<T>>* v) const;

  /**
   * Compute a sequence of generalized accelerations a from a sequence of
   * generalized velocities,
   *
   *    a_t = (v_{t+1} - v_{t})/dt,
   *
   * where v is of length (num_steps+1) and a is of length num_steps:
   *
   *     v = [v(0), v(1), v(2), ..., v(num_steps)],
   *     a = [a(0), a(1), a(2), ..., a(num_steps-1)].
   *
   * @param v sequence of generalized velocities
   * @param a sequence of generalized accelerations
   */
  void CalcAccelerations(const std::vector<VectorX<T>>& v,
                         std::vector<VectorX<T>>* a) const;

  /**
   * Compute a sequence of generalized forces t from sequences of generalized
   * accelerations, velocities, positions, and virtual force stiffnesses, where
   * generalized forces are defined by the inverse dynamics,
   *
   *    tau_t = M*(v_{t+1}-v_t})/dt + D*v_{t+1} - k(q_t,v_t)
   *             - (1/dt) *J'*gamma(v_{t+1},q_{t+1}) - (1/dt) * J'*fv(q_t, k_t)
   *
   *
   * Note that q, v, and k have length num_steps+1,
   *
   *  q = [q(0), q(1), ..., q(num_steps)],
   *  v = [v(0), v(1), ..., v(num_steps)],
   *  k = [k(0), k(1), ..., k(num_steps)],
   *
   * while a and tau have length num_steps,
   *
   *  a = [a(0), a(1), ..., a(num_steps-1)],
   *  tau = [tau(0), tau(1), ..., tau(num_steps-1)],
   *
   * i.e., tau(t) takes us us from t to t+1.
   *
   * @param state state variable storing a context for each timestep. This
   * context in turn stores q(t) and v(t) for each timestep. k(t) is also stored
   * in the state.
   * @param a sequence of generalized accelerations
   * @param workspace scratch space for intermediate computations
   * @param tau sequence of generalized forces
   */
  void CalcInverseDynamics(const TrajectoryOptimizerState<T>& state,
                           const std::vector<VectorX<T>>& a,
                           TrajectoryOptimizerWorkspace<T>* workspace,
                           std::vector<VectorX<T>>* tau) const;

  /**
   * Helper function for computing the inverse dynamics
   *
   *  tau = ID(a, v, q, f_ext)
   *
   * at a single timestep.
   *
   * @param context system context storing q and v
   * @param a generalized acceleration
   * @param k virtual force parameters
   * @param workspace scratch space for intermediate computations
   * @param tau generalized forces
   */
  void CalcInverseDynamicsSingleTimeStep(
      const Context<T>& context, const VectorX<T>& a, const VectorX<T>& k,
      TrajectoryOptimizerWorkspace<T>* workspace, VectorX<T>* tau) const;

  /**
   * Calculate the force contribution from contacts for each body, and add them
   * into the given MultibodyForces object. This includes both the actual
   * contact forces as well as virtual forces.
   *
   * @param context system context storing q and v
   * @param k virtual force parameters
   * @param forces total forces applied to the plant, which we will add into.
   */
  void CalcContactForceContribution(const Context<T>& context,
                                    const VectorX<T>& k,
                                    MultibodyForces<T>* forces) const;

  /**
   * Compute signed distance data for all contact pairs for all time steps.
   *
   * @param state state variable storing system configurations at each time
   * step.
   * @param sdf_data signed distance data that we'll set.
   */
  void CalcSdfData(
      const TrajectoryOptimizerState<T>& state,
      typename TrajectoryOptimizerCache<T>::SdfData* sdf_data) const;

  /**
   * Helper to compute the contact Jacobian (at a particular time step) for the
   * configuration stored in `context`.
   *
   * Signed distance pairs `sdf_pairs` must be consistent with
   * `context`.
   *
   * @param context context storing q and v
   * @param sdf_pairs vector of signed distance pairs
   * @param J the jacobian to set
   * @param R_WC the rotation of each contact frame in the world
   * @param body_pairs each pair of bodies that are in contact
   */
  void CalcContactJacobian(
      const Context<T>& context,
      const std::vector<geometry::SignedDistancePair<T>>& sdf_pairs,
      MatrixX<T>* J, std::vector<math::RotationMatrix<T>>* R_WC,
      std::vector<std::pair<BodyIndex, BodyIndex>>* body_pairs) const;

  /**
   * Compute Jacobian data for all time steps.
   *
   * @param state state variable containing configurations q for each time
   * @param contact_jacobian_data jacobian data that we'll set
   */
  void CalcContactJacobianData(
      const TrajectoryOptimizerState<T>& state,
      typename TrajectoryOptimizerCache<T>::ContactJacobianData*
          contact_jacobian_data) const;

  /**
   * Compute the mapping from qdot to v, v = N+(q)*qdot, at each time step.
   *
   * @param state optimizer state
   * @param N_plus vector containing N+(q_t) for each time step t.
   */
  void CalcNplus(const TrajectoryOptimizerState<T>& state,
                 std::vector<MatrixX<T>>* N_plus) const;

  /**
   * Compute partial derivatives of the generalized velocities
   *
   *    v_t = N+(q_t) * (q_t - q_{t-1}) / dt
   *
   * and store them in the given VelocityPartials struct.
   *
   * @param q sequence of generalized positions
   * @param v_partials struct for holding dv/dy
   */
  void CalcVelocityPartials(const TrajectoryOptimizerState<T>& state,
                            VelocityPartials<T>* v_partials) const;

  /**
   * Compute partial derivatives of the inverse dynamics
   *
   *    tau_t = ID(q_{t-1}, q_t, q_{t+1})
   *
   * and store them in the given InverseDynamicsPartials struct.
   *
   * @param state state variable containing y=[q,k] for each timestep
   * @param id_partials struct for holding dtau/dy
   */
  void CalcInverseDynamicsPartials(
      const TrajectoryOptimizerState<T>& state,
      InverseDynamicsPartials<T>* id_partials) const;

  /**
   * Compute partial derivatives of the inverse dynamics
   *
   *    tau_t = ID(q_{t-1}, q_t, q_{t+1})
   *
   * using forward finite differences.
   *
   * @param state state variable storing q, v, tau, etc.
   * @param id_partials struct for holding dtau/dy
   */
  void CalcInverseDynamicsPartialsFiniteDiff(
      const TrajectoryOptimizerState<T>& state,
      InverseDynamicsPartials<T>* id_partials) const;

  /**
   * Compute partial derivatives of the inverse dynamics
   *
   *    tau_t = ID(y_{t-1}, y_t, y_{t+1})
   *
   * exactly using central differences.
   *
   * Uses second order or 4th order central differences, depending on
   * the value of params_.gradients_method.
   *
   * @param state state variable storing q, v, tau, etc.
   * @param id_partials struct for holding dtau/dy
   */
  void CalcInverseDynamicsPartialsCentralDiff(
      const TrajectoryOptimizerState<T>& state,
      InverseDynamicsPartials<T>* id_partials) const;

  /**
   * Helper to compute derivatives of tau[t-1], tau[t] and tau[t+1] w.r.t. y[t]
   * for central differences.
   *
   * @param t the timestep under consideration
   * @param state state variable storing q, k, v, tau, etc
   * @param dtaum_dqt ∂τₜ₋₁/∂yₜ
   * @param dtaut_dqt ∂τₜ/∂yₜ
   * @param dtaup_dqt ∂τₜ₊₁/∂yₜ
   */
  void CalcInverseDynamicsPartialsWrtQtCentralDiff(
      int t, const TrajectoryOptimizerState<T>& state, MatrixX<T>* dtaum_dyt,
      MatrixX<T>* dtaut_dyt, MatrixX<T>* dtaup_dyt) const;

  /**
   * Compute partial derivatives of the inverse dynamics
   *
   *    tau_t = ID(y_{t-1}, y_t, y_{t+1})
   *
   * exactly using autodiff.
   *
   * @param state state variable storing q, k, v, tau, etc.
   * @param id_partials struct for holding dtau/dy
   */
  void CalcInverseDynamicsPartialsAutoDiff(
      const TrajectoryOptimizerState<double>& state,
      InverseDynamicsPartials<double>* id_partials) const;

  /**
   * Compute the gradient of the unconstrained cost L(y) using finite
   * differences.
   *
   * Uses central differences, so with a perturbation on the order of eps^(1/3),
   * we expect errors on the order of eps^(2/3).
   *
   * For testing purposes only.
   *
   * @param state optimizer state containing decision variables y
   * @param g a single VectorX<T> containing the partials of L w.r.t. each
   *          decision variable (q_t[i]).
   */
  void CalcGradientFiniteDiff(const TrajectoryOptimizerState<T>& state,
                              EigenPtr<VectorX<T>> g) const;

  /**
   * Compute the linesearch parameter alpha given a linesearch direction
   * dq. In other words, approximately solve the optimization problem
   *
   *      min_{alpha} L(y + alpha*dy).
   *
   * @param state the optimizer state containing y and everything that we
   *              compute from y
   * @param dy search direction, stacked as one large vector
   * @param scratch_state scratch state variable used for computing L(y +
   *                      alpha*dy)
   * @return double, the linesearch parameter alpha
   * @return int, the number of linesearch iterations
   */
  std::tuple<double, int> Linesearch(
      const TrajectoryOptimizerState<T>& state, const VectorX<T>& dy,
      TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Debugging function which saves the line-search residual
   *
   *    phi(alpha) = L(y + alpha*dy)
   *
   * for various values of alpha to a file.
   *
   * This allows us to make a nice plot in python after the fact
   */
  void SaveLinesearchResidual(
      const TrajectoryOptimizerState<T>& state, const VectorX<T>& dy,
      TrajectoryOptimizerState<T>* scratch_state,
      const std::string filename = "linesearch_data.csv") const;

  /**
   * Simple backtracking linesearch strategy to find alpha that satisfies
   *
   *    L(y + alpha*dy) < L(y) + c*g'*dy
   *
   * and is (approximately) a local minimizer of L(y + alpha*dy).
   */
  std::tuple<double, int> BacktrackingLinesearch(
      const TrajectoryOptimizerState<T>& state, const VectorX<T>& dy,
      TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Simple backtracking linesearch strategy to find alpha that satisfies
   *
   *    L(y + alpha*dy) < L(y) + c*g'*dy
   */
  std::tuple<double, int> ArmijoLinesearch(
      const TrajectoryOptimizerState<T>& state, const VectorX<T>& dy,
      TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Compute the trust ratio
   *
   *           L(y) - L(y + dy)
   *    rho =  ----------------
   *             m(0) - m(dy)
   *
   * which compares the actual reduction in cost to the reduction in cost
   * predicted by the quadratic model
   *
   *    m(dy) = L + g'*dy + 1/2 dy'*H*dy
   *
   * @param state optimizer state containing y and everything computed from y
   * @param dy change in y, stacked in one large vector
   * @param scratch_state scratch state variable used to compute L(y+dy)
   * @return T, the trust region ratio
   */
  T CalcTrustRatio(const TrajectoryOptimizerState<T>& state,
                   const VectorX<T>& dy,
                   TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Compute the dogleg step δq, which approximates the solution to the
   * trust-region sub-problem
   *
   *   min_{δy} L(y) + g(y)'*δy + 1/2 δy'*H(y)*δy
   *   s.t.     ‖ δy ‖ <= Δ
   *
   * @param state the optimizer state, containing y and the ability to compute
   * g(y) and H(y)
   * @param Delta the trust region size
   * @param dy the dogleg step (change in decision variables)
   * @param dyH the Newton step
   * @return true if the step intersects the trust region
   * @return false if the step is in the interior of the trust region
   */
  bool CalcDoglegPoint(const TrajectoryOptimizerState<T>& state,
                       const double Delta, VectorX<T>* dy,
                       VectorX<T>* dyH) const;

  /**
   * Solve the scalar quadratic equation
   *
   *    a x² + b x + c = 0
   *
   * for the positive root. This problem arises from finding the intersection
   * between the trust region and the second leg of the dogleg path. Provided we
   * have properly checked that the trust region does intersect this second
   * leg, this quadratic equation has some special properties:
   *
   *     - a is strictly positive
   *     - there is exactly one positive root
   *     - this positive root is in (0,1)
   *
   * @param a the first coefficient
   * @param b the second coefficient
   * @param c the third coefficient
   * @return T the positive root
   */
  T SolveDoglegQuadratic(const T& a, const T& b, const T& c) const;

  /* Helper to solve ths system H⋅x = b with a solver as specified with
  SolverParameters. On output b is overwritten with x. */
  void SolveLinearSystemInPlace(const PentaDiagonalMatrix<T>& H,
                                VectorX<T>* b) const;

  ConvergenceReason VerifyConvergenceCriteria(
      const TrajectoryOptimizerState<T>& state, const T& previous_cost,
      const VectorX<T>& dy) const;

  /**
   * Save the cost L(y) for a variety of values of y so that we can make a
   * contour plot (later) in python.
   *
   * Only changes the first two values of y(t) at t=1, so we can plot in 2d.
   *
   * @param scratch_state State variable used to compute L(y) for a variety of
   * values of y.
   */
  void SaveContourPlotDataFirstTwoVariables(
      TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Save the cost, gradient, and Hessian accross a range of values of y(1)[0],
   * where y(1)[0] is the first state variable at timestep t=1.
   *
   * This data will be used later to make debugging plots of L, g, and H.
   *
   * @param scratch_state Optimizer state used to compute L(y), g(y), and H(y).
   */
  void SaveLinePlotDataFirstVariable(
      TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Clear the file `iteration_data.csv` and write a csv header so we can later
   * record iteration data with SaveIterationData().
   */
  void SetupIterationDataFile() const;

  /**
   * Save iteration-specific data (like cost, q, etc) to a file.
   *
   * @param iter_num iteration number
   * @param Delta trust region radius
   * @param dy change in first decision variable
   * @param rho trust ratio
   * @param state state variable used to store q and compute L(y), g(y), H(y),
   * etc
   */
  void SaveIterationData(const int iter_num, const double Delta,
                         const double dy, const double rho,
                         const TrajectoryOptimizerState<T>& state) const;

  /**
   * Clear the file `quadratic_data.csv` and write a csv header so we can later
   * record iteration data with SaveQuadraticDataFirstTwoVariables().
   */
  void SetupQuadraticDataFile() const;

  /**
   * Save the cost L(y), gradient g(y), and Hessian approximation H(y) for the
   * first two variables of the optimization problem at the given iteration.
   *
   * @warning this appends a row to `quadratic_data.csv`, without
   * establishing any csv header or clearning the file. Make sure to call
   * SetupQuadraticDataFile() first.
   *
   * @param iter_num iteration number that we're on
   * @param Delta trust region radius
   * @param dy variable step for this iteration.
   * @param state optimizer state containing y, from which we can compute L, g,
   * and H
   */
  void SaveQuadraticDataFirstTwoVariables(
      const int iter_num, const double Delta, const VectorX<T>& dy,
      const TrajectoryOptimizerState<T>& state) const;

  // Diagram of containing the plant_ model and scene graph. Needed to allocate
  // context resources.
  const Diagram<T>* diagram_{nullptr};

  // A model of the system that we are trying to find an optimal trajectory for.
  const MultibodyPlant<T>* plant_{nullptr};

  // A context corresponding to plant_, to enable dynamics computations. Must be
  // connected to a larger Diagram with a SceneGraph for systems with contact.
  // Right now only used by CalcInverseDynamicsPartialsFiniteDiff() and
  // CalcInverseDynamicsPartialsWrtQtCentralDiff().
  Context<T>* context_{nullptr};

  // Temporary workaround for when context_ is not provided at construction.
  // Right now only used by CalcInverseDynamicsPartialsFiniteDiff().
  // TODO(amcastro-tri): Get rid of context_ and owned_context_.
  std::unique_ptr<Context<T>> owned_context_;

  // Stores the problem definition, including cost, time horizon, initial state,
  // target state, etc.
  const ProblemDefinition prob_;

  // Joint damping coefficients for the plant under consideration
  VectorX<T> joint_damping_;

  // Various parameters
  const SolverParameters params_;

  // Autodiff copies of the system diagram, plant model, optimizer state, and a
  // whole optimizer for computing exact gradients.
  std::unique_ptr<Diagram<AutoDiffXd>> diagram_ad_;
  const MultibodyPlant<AutoDiffXd>* plant_ad_;
  std::unique_ptr<TrajectoryOptimizer<AutoDiffXd>> optimizer_ad_;
  std::unique_ptr<TrajectoryOptimizerState<AutoDiffXd>> state_ad_;
};

// Declare template specializations
template <>
SolverFlag TrajectoryOptimizer<double>::SolveWithLinesearch(
    const std::vector<VectorXd>&, TrajectoryOptimizerSolution<double>*,
    TrajectoryOptimizerStats<double>*) const;

template <>
SolverFlag TrajectoryOptimizer<double>::SolveWithTrustRegion(
    const std::vector<VectorXd>&, TrajectoryOptimizerSolution<double>*,
    TrajectoryOptimizerStats<double>*, ConvergenceReason*) const;

}  // namespace traj_opt
}  // namespace drake
