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
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

// num_repetitions: How many times to repeat the kernel invocation
size_t num_repetitions = 6;
// Vector type and data size for this example.
size_t vector_size = 10000000;
typedef std::vector<int> IntVector;

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
// Vector add in SYCL on device: returns sum in 4th parameter "sum_parallel".
//************************************
double VectorAdd(queue& q, const IntVector& a_vector, const IntVector& b_vector,
                 IntVector& sum_parallel) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{a_vector.size()};

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer a_buf(a_vector);
  buffer b_buf(b_vector);
  buffer sum_buf(sum_parallel.data(), num_items);
  double total_kernel_time = 0;
  for (size_t i = 0; i < num_repetitions; i++) {
    // Submit a command group to the queue by a lambda function that contains
    // the data access permission and device computation (kernel).
    auto e = q.submit([&](handler& h) {
      // Create an accessor for each buffer with access permission: read, write
      // or read/write. The accessor is a mean to access the memory in the
      // buffer.
      accessor a(a_buf, h, read_only);
      accessor b(b_buf, h, read_only);

      // The sum_accessor is used to store (with write permission) the sum data.
      accessor sum(sum_buf, h, write_only, no_init);

      // Use parallel_for to run vector addition in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // SYCL supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](auto i) {
        sum[i] = a[i] + b[i];
      });
    });
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
  };
  // Wait until compute tasks on GPU done
  return total_kernel_time / (num_repetitions - 1);
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
void InitializeVector(IntVector& a) {
  for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change num_repetitions if it was passed as argument
  if (argc > 2) num_repetitions = std::stoi(argv[2]);
  // Change vector_size if it was passed as argument
  if (argc > 1) vector_size = std::stoi(argv[1]);
  // Create device selector for the device of your interest.
  // The default device selector will select the most performant device.
  auto gpu_selector = gpu_selector_v;
  auto cpu_selector = cpu_selector_v;
  auto selector = default_selector_v;

  // Create vector objects with "vector_size" to store the input and output
  // data.
  IntVector a, b, sum_sequential, sum_parallel_cpu, sum_parallel_gpu;
  a.resize(vector_size);
  b.resize(vector_size);
  sum_sequential.resize(vector_size);
  sum_parallel_cpu.resize(vector_size);
  sum_parallel_gpu.resize(vector_size);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector(a);
  InitializeVector(b);

  double kernel_time_gpu = 0;
  double kernel_time_cpu = 0;
  try {
    queue gpu_q(gpu_selector, exception_handler,
                property::queue::enable_profiling());

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << gpu_q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << a.size() << "\n";

    // Vector addition in SYCL
    kernel_time_gpu = VectorAdd(gpu_q, a, b, sum_parallel_gpu);

    queue cpu_q(cpu_selector, exception_handler,
                property::queue::enable_profiling());

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << cpu_q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << a.size() << "\n";

    // Vector addition in SYCL
    kernel_time_cpu = VectorAdd(cpu_q, a, b, sum_parallel_cpu);
  } catch (exception const& e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }

  // Compute the sum of two vectors in sequential for validation.
  for (size_t i = 0; i < sum_sequential.size(); i++)
    sum_sequential.at(i) = a.at(i) + b.at(i);

  // Verify that the two vectors are equal.
  for (size_t i = 0; i < sum_sequential.size(); i++) {
    if (sum_parallel_cpu.at(i) != sum_sequential.at(i)) {
      std::cout << "Vector add failed on device cpu.\n";
      return -1;
    }
    if (sum_parallel_gpu.at(i) != sum_sequential.at(i)) {
      std::cout << "Vector add failed on device gpu.\n";
      return -1;
    }
  }

  int indices[]{0, 1, 2, (static_cast<int>(a.size()) - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // Print out the result of vector add.
  for (size_t i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "]: " << a[j] << " + " << b[j] << " = "
              << sum_parallel_cpu[j] << " (cpu) " << sum_parallel_gpu[j]
              << " (gpu)\n";
  }

  a.clear();
  b.clear();
  sum_sequential.clear();
  sum_parallel_cpu.clear();
  sum_parallel_gpu.clear();

  std::cout << "Average Kernel time CPU: " << kernel_time_cpu << " ms"
            << std::endl;
  std::cout << "Average Kernel time GPU: " << kernel_time_gpu << " ms"
            << std::endl;

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
