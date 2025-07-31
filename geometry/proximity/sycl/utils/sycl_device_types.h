#pragma once

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Enum class to differentiate between GPU and CPU execution contexts
// for SYCL device function templating
enum class DeviceType { GPU, CPU };

// Template struct to wrap device type as a compile-time constant
template <DeviceType device_type>
struct DeviceTraits {
  static constexpr bool GPU = (device_type == DeviceType::GPU);
  static constexpr bool CPU = (device_type == DeviceType::CPU);
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake