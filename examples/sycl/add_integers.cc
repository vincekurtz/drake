#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

/**
 * Helper function for printing every element of a vector.
 */
void print_vector(const std::vector<int>& v) {
  std::cout << "[";
  for (const auto& e : v) {
    std::cout << e << " ";
  }
  std::cout << "]" << std::endl;
}

int main() {
  // Create a simple SYCL queue. We'll request the GPU device.
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "Using device: "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  std::cout << "Driver version: "
            << Q.get_device().get_info<sycl::info::device::driver_version>()
            << std::endl;
  

  // Create some initial data on the host. We'll be adding 1 to each element
  // of this vector.
  const int N = 16;
  std::vector<int> data(N);
  for (int i = 0; i < N; ++i) {
    data[i] = i;
  }
  std::cout << "Initial data:     ";
  print_vector(data);

  // Create a new scope for the buffer. When the buffer goes out of scope, SYCL
  // will (1) wait for any kernels to finish, and (2) copy data back to the host
  {
    // Create a buffer. This will be used to access the data on the device.
    sycl::buffer B{data};

    // Submit a kernel (as an anonymous lambda function) to the queue.
    Q.submit([&](sycl::handler& cgh) {
      // Create an accessor, which will allow us to read and write to the
      // buffer.
      sycl::accessor A{B, cgh};

      // The kernel itself. This will be executed on the device.
      cgh.parallel_for(N, [=](auto& i) {
        A[i] += 1;
      });
    });

    // At this point, the kernel has been submitted, but the results may not be
    // ready yet, since the kernel will be executed asynchronously.
    std::cout << "Kernel submitted: ";
    print_vector(data);

    // Let's block the host until the kernel is finished.
    Q.wait();

    // Even after waiting, the results will not be available on the host. That's
    // because the results of the computation are still on the device only!
    std::cout << "Kernel finshed:   ";
    print_vector(data);
  }

  // To get the results back to the host, we can either use sycl::host_accessor,
  // or destroy the buffer by causing it to go out of scope.
  std::cout << "Final result:     ";
  print_vector(data);

  return 0;
}