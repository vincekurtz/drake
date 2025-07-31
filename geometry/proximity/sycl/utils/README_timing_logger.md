# SYCL Timing Logger

This directory contains a timing logger for SYCL kernels that helps identify performance bottlenecks in the SYCL proximity engine.

## Files

- `sycl_timing_logger.h` - Main timing logger class
- `sycl_timing_logger_example.h` - Example usage
- `sycl_timing_integration.h` - Integration guide for the proximity engine

## Usage

### 1. Enable Timing

To enable timing, define the macro before including the header:

```cpp
#define DRAKE_SYCL_TIMING_ENABLED
#include "drake/geometry/proximity/sycl/utils/sycl_timing_logger.h"
```

When this macro is not defined, the timing logger becomes a no-op class with zero overhead.

### 2. Basic Usage

```cpp
#include "drake/geometry/proximity/sycl/utils/sycl_timing_logger.h"

SyclTimingLogger logger;

// Time a kernel using SYCL's built-in profiling
auto event = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> idx) {
        // Your kernel code
    });
});
logger.TimeSyclEvent("my_kernel", event);

// Print all timing statistics
logger.PrintStats();
```

### 3. Integration with SYCL Proximity Engine

To add timing to your existing SYCL proximity engine:

1. Add the timing logger as a member to your `SyclProximityEngine::Impl` class:

```cpp
class SyclProximityEngine::Impl {
private:
    SyclTimingLogger timing_logger_;
    // ... other members
};
```

2. Add timing calls around your kernels in `ComputeSYCLHydroelasticSurface`:

```cpp
// Transform vertices kernel
timing_logger_.StartKernel("transform_vertices");
auto transform_vertices_event = q_device_.submit([&](sycl::handler& h) {
    // Your existing transform vertices kernel
});
timing_logger_.TimeSyclEvent("transform_vertices", transform_vertices_event);

// Element AABB computation kernel
timing_logger_.StartKernel("compute_element_aabbs");
auto element_aabb_event = q_device_.submit([&](sycl::handler& h) {
    // Your existing element AABB computation kernel
});
timing_logger_.TimeSyclEvent("compute_element_aabbs", element_aabb_event);

// ... continue for other kernels
```

3. Print statistics at the end of your computation:

```cpp
timing_logger_.PrintStats();
```

## Features

- **Conditional compilation**: Zero overhead when timing is disabled
- **SYCL built-in profiling**: Uses SYCL's native event profiling for accurate timing
- **Statistics tracking**: Tracks min, max, average, and total times for each kernel
- **Multiple timing methods**: Support for kernel timing, event timing, and SYCL event timing
- **Clean API**: Simple interface that doesn't pollute your public API

## Output Example

When timing is enabled, you'll see output like:

```
=== SYCL Kernel Timing Statistics ===
Kernel: transform_vertices
  Calls: 10
  Min: 45.2 μs
  Max: 52.1 μs
  Avg: 48.7 μs
  Total: 487.0 μs

Kernel: compute_element_aabbs
  Calls: 10
  Min: 23.1 μs
  Max: 28.9 μs
  Avg: 25.4 μs
  Total: 254.0 μs

SYCL Event: generate_collision_filter
  Calls: 10
  Min: 12.3 μs
  Max: 15.7 μs
  Avg: 13.8 μs
  Total: 138.0 μs
```

## Performance Considerations

- When `DRAKE_SYCL_TIMING_ENABLED` is not defined, all timing calls are no-ops
- SYCL event profiling has minimal overhead and provides accurate device timing
- The logger uses high-resolution clocks for host-side timing when needed
- Statistics are computed on-demand to avoid unnecessary computation

## Notes

- SYCL event profiling requires the queue to be created with profiling enabled
- The timing logger is thread-safe for basic operations but not for concurrent access to the same logger instance
- All times are reported in microseconds (μs) for consistency 