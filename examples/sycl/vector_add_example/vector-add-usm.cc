//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of SYCL. This sample runs on both CPU and GPU. When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data shared between CPU and offload device.
// •	A device queue and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <array>
#include <iostream>
#include <string>

#include <sycl/sycl.hpp>

using namespace sycl;

// Number of repetitions for timing
size_t num_repetitions = 6;
// Array size for this example.
size_t array_size = 10000000;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const& e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const& e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

//************************************
// Vector add in SYCL on device: returns sum in 4th parameter "sum".
//************************************
double VectorAdd(queue& q, const int* a, const int* b, int* sum, size_t size) {
  // Create the range object for the arrays.

  range<1> num_items{size};
  double total_kernel_time = 0;

  for (size_t i = 0; i < num_repetitions; i++) {
    // Use parallel_for to run vector addition in parallel on device. This
    // executes the kernel.
    //    1st parameter is the number of work items.
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. the parameter of the lambda is the work item id.
    // SYCL supports unnamed lambda kernel by default.
    auto e = q.parallel_for(num_items, [=](auto i) {
      sum[i] = a[i] + b[i];
    });

    // q.parallel_for() is an asynchronous call. SYCL runtime enqueues and runs
    // the kernel asynchronously. Wait for the asynchronous call to complete.
    e.wait();

    double step_kernel_time =
        (e.template get_profiling_info<
             sycl::info::event_profiling::command_end>() -
         e.template get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1e6;

    if (i != 0) total_kernel_time += step_kernel_time;
    if (i == 0) {
      std::cout << "Warmup done" << std::endl;
    }
    std::cout << "Run " << i + 1 << " Kernel time: " << step_kernel_time
              << " ms" << std::endl;
  }

  // Return average kernel time, excluding warmup
  return total_kernel_time / (num_repetitions - 1);
}

//************************************
// Initialize the array from 0 to array_size - 1
//************************************
void InitializeArray(int* a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change array_size if it was passed as argument
  if (argc > 1) array_size = std::stoi(argv[1]);
  // Change num_repetitions if it was passed as argument
  if (argc > 2) num_repetitions = std::stoi(argv[2]);

  // Create device selectors for the devices of interest.
  auto gpu_selector = gpu_selector_v;
  auto cpu_selector = cpu_selector_v;

  double kernel_time_gpu = 0;
  double kernel_time_cpu = 0;

  try {
    // GPU queue with profiling enabled
    queue gpu_q(gpu_selector, exception_handler,
                property::queue::enable_profiling());

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << gpu_q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << array_size << "\n";

    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    int* a = malloc_shared<int>(array_size, gpu_q);
    int* b = malloc_shared<int>(array_size, gpu_q);
    int* sum_sequential = malloc_shared<int>(array_size, gpu_q);
    int* sum_parallel_gpu = malloc_shared<int>(array_size, gpu_q);
    int* sum_parallel_cpu = malloc_shared<int>(array_size, gpu_q);

    if ((a == nullptr) || (b == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel_gpu == nullptr) || (sum_parallel_cpu == nullptr)) {
      if (a != nullptr) free(a, gpu_q);
      if (b != nullptr) free(b, gpu_q);
      if (sum_sequential != nullptr) free(sum_sequential, gpu_q);
      if (sum_parallel_gpu != nullptr) free(sum_parallel_gpu, gpu_q);
      if (sum_parallel_cpu != nullptr) free(sum_parallel_cpu, gpu_q);

      std::cout << "Shared memory allocation failure.\n";
      return -1;
    }

    // Initialize input arrays with values from 0 to array_size - 1
    InitializeArray(a, array_size);
    InitializeArray(b, array_size);

    // Compute the sum of two arrays in sequential for validation.
    for (size_t i = 0; i < array_size; i++) sum_sequential[i] = a[i] + b[i];

    // Vector addition in SYCL on GPU
    kernel_time_gpu = VectorAdd(gpu_q, a, b, sum_parallel_gpu, array_size);

    // CPU queue with profiling enabled
    queue cpu_q(cpu_selector, exception_handler,
                property::queue::enable_profiling());

    // Print out the CPU device information
    std::cout << "Running on device: "
              << cpu_q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << array_size << "\n";

    // Vector addition in SYCL on CPU
    kernel_time_cpu = VectorAdd(cpu_q, a, b, sum_parallel_cpu, array_size);

    // Verify that the arrays are equal.
    for (size_t i = 0; i < array_size; i++) {
      if (sum_parallel_gpu[i] != sum_sequential[i]) {
        std::cout << "Vector add failed on GPU device.\n";
        return -1;
      }
      if (sum_parallel_cpu[i] != sum_sequential[i]) {
        std::cout << "Vector add failed on CPU device.\n";
        return -1;
      }
    }

    int indices[]{0, 1, 2, (static_cast<int>(array_size) - 1)};
    constexpr size_t indices_size = sizeof(indices) / sizeof(int);

    // Print out the result of vector add.
    for (size_t i = 0; i < indices_size; i++) {
      int j = indices[i];
      if (i == indices_size - 1) std::cout << "...\n";
      std::cout << "[" << j << "]: " << a[j] << " + " << b[j] << " = "
                << sum_sequential[j] << " (cpu) " << sum_parallel_gpu[j]
                << " (gpu)\n";
    }

    free(a, gpu_q);
    free(b, gpu_q);
    free(sum_sequential, gpu_q);
    free(sum_parallel_gpu, gpu_q);
    free(sum_parallel_cpu, gpu_q);

    std::cout << "Average Kernel time CPU: " << kernel_time_cpu << " ms"
              << std::endl;
    std::cout << "Average Kernel time GPU: " << kernel_time_gpu << " ms"
              << std::endl;

  } catch (exception const& e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
