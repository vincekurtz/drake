#include "drake/examples/sycl/simple.h"

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>

namespace drake {
namespace examples {
namespace simple {


void print_platform_devices() {
  for (const auto& platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << "\n";
    for (const auto& device : platform.get_devices()) {
      std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                << " | Type: "
                << (device.is_gpu()   ? "GPU"
                    : device.is_cpu() ? "CPU"
                                      : "Other")
                << " | Vendor: "
                << device.get_info<sycl::info::device::vendor>() << "\n";
    }
  }
}
void InitializeVector(IntVector &a) {
  for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
}

void InitializeArray(int *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = i;
}

void VectorAdd_buffer(sycl::queue &q, const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel, size_t num_repetitions) {
  // Create the range object for the vectors managed by the buffer.
  sycl::range<1> num_items{a_vector.size()};

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  sycl::buffer a_buf(a_vector);
  sycl::buffer b_buf(b_vector);
  sycl::buffer sum_buf(sum_parallel.data(), num_items);

  for (size_t i = 0; i < num_repetitions; i++ ) {

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](sycl::handler &h) {
      // Create an accessor for each buffer with access permission: read, write or
      // read/write. The accessor is a mean to access the memory in the buffer.
      sycl::accessor a(a_buf, h, sycl::read_only);
      sycl::accessor b(b_buf, h, sycl::read_only);
  
      // The sum_accessor is used to store (with write permission) the sum data.
      sycl::accessor sum(sum_buf, h, sycl::write_only, sycl::no_init);
  
      // Use parallel_for to run vector addition in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // SYCL supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
    });
  };
  // Wait until compute tasks on GPU done
  q.wait();
}

void VectorAdd_usm(sycl::queue &q, int *a, int *b, int *sum_parallel, size_t size) {
  // Create the range object for the arrays.
  sycl::range<1> num_items{size};

  // Use parallel_for to run vector addition in parallel on device. This
  // executes the kernel.
  //    1st parameter is the number of work items.
  //    2nd parameter is the kernel, a lambda that specifies what to do per
  //    work item. the parameter of the lambda is the work item id.
  // SYCL supports unnamed lambda kernel by default.
  auto e = q.parallel_for(num_items, [=](auto i) { sum_parallel[i] = a[i] + b[i]; });

  // q.parallel_for() is an asynchronous call. SYCL runtime enqueues and runs
  // the kernel
  e.wait();
}

int sum_ids() {
  print_platform_devices();
  // Creating buffer of 4 ints to be used inside the kernel code
  sycl::buffer<int, 1> Buffer{4};

  // Creating SYCL queue
  sycl::queue Queue{};

  // Size of index space for kernel
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler& cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      // Fill buffer with indexes
      Accessor[WIid] = static_cast<int>(WIid.get(0));
    });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  auto HostAccessor = Buffer.get_host_access();

  // Check the results
  bool MismatchFound{false};
  int sum = 0;
  for (size_t I{0}; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != static_cast<int>(I)) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
    sum += HostAccessor[I];
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return sum;
}

int vector_add_buffer() {
  // print_platform_devices();
  // num_repetitions: How many times to repeat the kernel invocation
  size_t num_repetitions = 1;
  // Vector type and data size for this example.
  size_t vector_size = 10000;
  

  // Create vector objects with "vector_size" to store the input and output data.
  IntVector a, b, sum_sequential, sum_parallel;
  a.resize(vector_size);
  b.resize(vector_size);
  sum_sequential.resize(vector_size);
  sum_parallel.resize(vector_size);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector(a);
  InitializeVector(b);

  sycl::queue q{};
  VectorAdd_buffer(q, a, b, sum_parallel, num_repetitions);

  // Compute the sum of two vectors in sequential for validation.
  for (size_t i = 0; i < sum_sequential.size(); i++)
    sum_sequential.at(i) = a.at(i) + b.at(i);

  // Verify that the two vectors are equal.  
  for (size_t i = 0; i < sum_sequential.size(); i++) {
    if (sum_parallel.at(i) != sum_sequential.at(i)) {
      std::cout << "Vector add failed on device.\n";
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
              << sum_parallel[j] << "\n";
  }

  a.clear();
  b.clear();
  sum_sequential.clear();
  sum_parallel.clear();
  std::cout << "Vector add successfully completed on device with buffer.\n";
  return 0;
}

int vector_add_usm() {

  // print_platform_devices();

  size_t vector_size = 10000;

  sycl::queue q{};

  // Create arrays with "array_size" to store input and output data. Allocate
  // unified shared memory so that both CPU and device can access them.
  int *a = malloc_shared<int>(vector_size, q);
  int *b = malloc_shared<int>(vector_size, q);
  int *sum_sequential = malloc_shared<int>(vector_size, q);
  int *sum_parallel = malloc_shared<int>(vector_size, q);


  if ((a == nullptr) || (b == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel == nullptr)) {
      if (a != nullptr) free(a, q);
      if (b != nullptr) free(b, q);
      if (sum_sequential != nullptr) free(sum_sequential, q);
      if (sum_parallel != nullptr) free(sum_parallel, q);

      std::cout << "Shared memory allocation failure.\n";
    return -1;
  }
  // Initialize input arrays with values from 0 to array_size - 1
  InitializeArray(a, vector_size);
  InitializeArray(b, vector_size);


  // Compute the sum of two arrays in sequential for validation.
  for (size_t i = 0; i < vector_size; i++) sum_sequential[i] = a[i] + b[i];
  VectorAdd_usm(q, a, b, sum_parallel, vector_size);

  // Verify that the two arrays are equal.
  for (size_t i = 0; i < vector_size; i++) {
    if (sum_parallel[i] != sum_sequential[i]) {
      std::cout << "Vector add failed on device.\n";
      return -1;
    }
  }

  int indices[]{0, 1, 2, (static_cast<int>(vector_size) - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // Print out the result of vector add.
  for (size_t i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "]: " << j << " + " << j << " = "
              << sum_sequential[j] << "\n";
  }

  free(a, q);
  free(b, q);
  free(sum_sequential, q);
  free(sum_parallel, q);


  std::cout << "Vector add successfully completed on device with USM.\n";
  return 0;
}  // End of vector_add_usm()

}  // namespace simple
}  // namespace examples
}  // namespace drake