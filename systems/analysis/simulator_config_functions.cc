#include "drake/systems/analysis/simulator_config_functions.h"

#include <cctype>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

#include <fmt/format.h>

#include "drake/common/drake_throw.h"
#include "drake/common/never_destroyed.h"
#include "drake/common/nice_type_name.h"
#include "drake/common/unused.h"
#include "drake/systems/analysis/bogacki_shampine3_integrator.h"
#include "drake/systems/analysis/convex_integrator.h"
#include "drake/systems/analysis/explicit_euler_integrator.h"
#include "drake/systems/analysis/implicit_euler_integrator.h"
#include "drake/systems/analysis/radau_integrator.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/runge_kutta3_integrator.h"
#include "drake/systems/analysis/runge_kutta5_integrator.h"
#include "drake/systems/analysis/sdirk2_integrator.h"
#include "drake/systems/analysis/semi_explicit_euler_integrator.h"
#include "drake/systems/analysis/velocity_implicit_euler_integrator.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace {

using std::function;
using std::pair;
using std::string;
using std::vector;

// A functor that implements ConfigureIntegrator.
template <typename T>
using ConfigureIntegratorFunc = function<
    std::variant<IntegratorBase<T>*, std::unique_ptr<IntegratorBase<T>>>(
        Simulator<T>*, const System<T>&, const T& /* max_step_size */)>;

// Returns (scheme, functor) pair that implements ConfigureIntegrator.
template <typename T>
using NamedConfigureIntegratorFunc = pair<string, ConfigureIntegratorFunc<T>>;

// Converts the class name of the `Integrator` template argument into a string
// name for the scheme, e.g., FooBarIntegrator<double> becomes "foo_bar".
template <template <typename> class Integrator>
string GetIntegratorName() {
  // Get the class name, e.g., FooBarIntegrator<double>.
  string full_name = NiceTypeName::Get<Integrator<double>>();
  string class_name = NiceTypeName::RemoveNamespaces(full_name);
  if (class_name == "RadauIntegrator<double,1>") {
    class_name = "Radau1Integrator<double>";
  } else if (class_name == "RadauIntegrator<double,2>") {
    class_name = "Radau3Integrator<double>";
  }

  // Strip off "Integrator<double>" suffix to leave just "FooBar".
  const string suffix = "Integrator<double>";
  DRAKE_DEMAND(class_name.size() > suffix.size());
  const size_t suffix_begin = class_name.size() - suffix.size();
  DRAKE_DEMAND(class_name.substr(suffix_begin) == suffix);
  const string camel_name = class_name.substr(0, suffix_begin);

  // Convert "FooBar to "foo_bar".
  string result;
  for (char ch : camel_name) {
    if (std::isupper(ch)) {
      if (!result.empty()) {
        result.push_back('_');
      }
      result.push_back(std::tolower(ch));
    } else {
      result.push_back(ch);
    }
  }
  return result;
}

// A hollow shell of a System.  TODO(jeremy.nimmer) Move into drake primitives.
template <typename T>
class DummySystem final : public LeafSystem<T> {
 public:
  DummySystem() {}
};
// N.B In a roundabout way the string returned here is generated by
// GetIntegratorName().
template <typename T>
string GetIntegrationSchemeName(const IntegratorBase<T>& integrator) {
  const string current_type = NiceTypeName::Get(integrator);
  Simulator<T> dummy_simulator(std::make_unique<DummySystem<T>>());
  for (const auto& scheme : GetIntegrationSchemes()) {
    ResetIntegratorFromFlags(&dummy_simulator, scheme, T(0.001));
    if (NiceTypeName::Get(dummy_simulator.get_integrator()) == current_type) {
      return scheme;
    }
  }
  throw std::runtime_error("Unrecognized integration scheme " + current_type);
}

// Returns the (scheme, functor) configurator pair for this integrator type.
// The functor is null if the scheme does not support the scalar type T.
// The functor returns IntegratorBase<T>* if the parameter `simulator` is
// provided, returns std::unique_ptr<IntegratorBase<T>> otherwise.
template <typename T, template <typename> class Integrator>
NamedConfigureIntegratorFunc<T> MakeConfigurator() {
  constexpr bool is_fixed_step =
      std::is_constructible_v<Integrator<T>, const System<T>&, T, Context<T>*>;
  constexpr bool is_error_controlled =
      std::is_constructible_v<Integrator<T>, const System<T>&, Context<T>*>;
  static_assert(is_fixed_step ^ is_error_controlled);

  // Among currently-existing set of integrators, all fixed-step
  // integrators support all default scalars; all error-controlled
  // integrators support only default nonsymbolic scalars.
  if constexpr (std::is_same_v<T, symbolic::Expression> && !is_fixed_step) {
    return NamedConfigureIntegratorFunc<T>(GetIntegratorName<Integrator>(),
                                           nullptr);
  } else {
    return NamedConfigureIntegratorFunc<T>(
        GetIntegratorName<Integrator>(),
        [](Simulator<T>* simulator, const System<T>& system,
           const T& max_step_size)
            -> std::variant<IntegratorBase<T>*,
                            std::unique_ptr<IntegratorBase<T>>> {
          // The functor returns the integrator owned by the simulator if
          // simulator is non-null.
          if (simulator != nullptr) {
            if constexpr (is_fixed_step) {
              IntegratorBase<T>& result =
                  simulator->template reset_integrator<Integrator<T>>(
                      max_step_size);
              return &result;
            } else {
              IntegratorBase<T>& result =
                  simulator->template reset_integrator<Integrator<T>>();
              result.set_maximum_step_size(max_step_size);
              return &result;
            }
          }
          // The functor returns a new integrator if simulator is null.
          if constexpr (is_fixed_step) {
            auto result =
                std::make_unique<Integrator<T>>(system, max_step_size, nullptr);
            return result;
          } else {
            auto result = std::make_unique<Integrator<T>>(system, nullptr);
            result->set_maximum_step_size(max_step_size);
            return result;
          }
        });
  }
}

// Returns the full list of supported (scheme, functor) pairs.  N.B. The list
// here must be kept in sync with the help string in simulator_gflags.cc.
template <typename T>
const vector<NamedConfigureIntegratorFunc<T>>&
GetAllNamedConfigureIntegratorFuncs() {
  static const never_destroyed<vector<NamedConfigureIntegratorFunc<T>>> result{
      std::initializer_list<NamedConfigureIntegratorFunc<T>>{
          // Keep this list sorted alphabetically.
          MakeConfigurator<T, BogackiShampine3Integrator>(),
          MakeConfigurator<T, ConvexIntegrator>(),
          MakeConfigurator<T, ExplicitEulerIntegrator>(),
          MakeConfigurator<T, ImplicitEulerIntegrator>(),
          MakeConfigurator<T, Radau1Integrator>(),
          MakeConfigurator<T, Radau3Integrator>(),
          MakeConfigurator<T, RungeKutta2Integrator>(),
          MakeConfigurator<T, RungeKutta3Integrator>(),
          MakeConfigurator<T, RungeKutta5Integrator>(),
          MakeConfigurator<T, Sdirk2Integrator>(),
          MakeConfigurator<T, SemiExplicitEulerIntegrator>(),
          MakeConfigurator<T, VelocityImplicitEulerIntegrator>(),
      }};
  return result.access();
}

}  // namespace

template <typename T>
IntegratorBase<T>& ResetIntegratorFromFlags(Simulator<T>* simulator,
                                            const string& scheme,
                                            const T& max_step_size) {
  static_assert(
      !std::is_same_v<T, symbolic::Expression>,
      "Simulator<T> does not support symbolic::Expression scalar type");
  DRAKE_THROW_UNLESS(simulator != nullptr);

  const auto& name_func_pairs = GetAllNamedConfigureIntegratorFuncs<T>();
  for (const auto& [one_name, one_func] : name_func_pairs) {
    if (scheme == one_name) {
      return *std::get<IntegratorBase<T>*>(
          one_func(simulator, simulator->get_system(), max_step_size));
    }
  }
  throw std::runtime_error(
      fmt::format("Unknown integration scheme: {}", scheme));
}

const vector<string>& GetIntegrationSchemes() {
  static const never_destroyed<vector<string>> result{[]() {
    vector<string> names;
    const auto& name_func_pairs = GetAllNamedConfigureIntegratorFuncs<double>();
    for (const auto& [one_name, one_func] : name_func_pairs) {
      names.push_back(one_name);
      unused(one_func);
    }
    return names;
  }()};
  return result.access();
}

template <typename T>
void ApplySimulatorConfig(const SimulatorConfig& config,
                          Simulator<T>* simulator) {
  DRAKE_THROW_UNLESS(simulator != nullptr);
  IntegratorBase<T>& integrator = ResetIntegratorFromFlags(
      simulator, config.integration_scheme, T(config.max_step_size));
  if (integrator.supports_error_estimation()) {
    integrator.set_fixed_step_mode(!config.use_error_control);
  }
  if (!integrator.get_fixed_step_mode()) {
    integrator.set_target_accuracy(config.accuracy);
  }
  simulator->set_target_realtime_rate(config.target_realtime_rate);
  // It is almost always the case we want these two next flags to be either both
  // true or both false. Otherwise we could miss the first publish at t = 0.
  simulator->set_publish_at_initialization(config.publish_every_time_step);
  simulator->set_publish_every_time_step(config.publish_every_time_step);
}

template <typename T>
SimulatorConfig ExtractSimulatorConfig(const Simulator<T>& simulator) {
  SimulatorConfig result;
  const IntegratorBase<T>& integrator = simulator.get_integrator();
  result.integration_scheme = GetIntegrationSchemeName(integrator);
  result.max_step_size =
      ExtractDoubleOrThrow(integrator.get_maximum_step_size());
  if (integrator.supports_error_estimation()) {
    result.use_error_control = !integrator.get_fixed_step_mode();
    const double accuracy_in_use =
        ExtractDoubleOrThrow(integrator.get_accuracy_in_use());
    DRAKE_DEMAND(!std::isnan(accuracy_in_use));
    result.accuracy = accuracy_in_use;
  } else {
    result.use_error_control = false;
    result.accuracy = 0.0;
  }
  result.target_realtime_rate =
      ExtractDoubleOrThrow(simulator.get_target_realtime_rate());
  result.publish_every_time_step = simulator.get_publish_every_time_step();
  return result;
}

template <typename T>
std::unique_ptr<IntegratorBase<T>> CreateIntegratorFromConfig(
    const System<T>* system, const SimulatorConfig& config) {
  DRAKE_THROW_UNLESS(system != nullptr);
  const auto& name_func_pairs = GetAllNamedConfigureIntegratorFuncs<T>();
  for (const auto& [one_name, one_func] : name_func_pairs) {
    if (config.integration_scheme == one_name) {
      if (!one_func) {
        throw std::logic_error(fmt::format(
            "Integration scheme '{}' does not support scalar type {}", one_name,
            NiceTypeName::Get<T>()));
      }
      auto integrator = std::get<std::unique_ptr<IntegratorBase<T>>>(
          one_func(nullptr, *system, config.max_step_size));
      if (integrator->supports_error_estimation()) {
        integrator->set_fixed_step_mode(!config.use_error_control);
        integrator->set_target_accuracy(config.accuracy);
      }
      return integrator;
    }
  }
  throw std::runtime_error(
      fmt::format("Unknown integration scheme: {}", config.integration_scheme));
}

// We can't support T=symbolic::Expression because Simulator doesn't support it.
DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ResetIntegratorFromFlags<T>, &ApplySimulatorConfig<T>,
     &ExtractSimulatorConfig<T>));

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&CreateIntegratorFromConfig<T>));

}  // namespace systems
}  // namespace drake
