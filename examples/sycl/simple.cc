#include <sycl/sycl.hpp>

int main() {
  // Creating buffer of 4 ints to be used inside the kernel code
  sycl::buffer<int, 1> Buffer{4};

  // Creating SYCL queue
  sycl::queue Queue{};

  // Size of index space for kernel
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](sycl::id<1> WIid) {
          // Fill buffer with indexes
          Accessor[WIid] = static_cast<int>(WIid.get(0));
        });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  auto HostAccessor = Buffer.get_host_access();

  // Check the results
  bool MismatchFound{false};
  for (size_t I{0}; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != static_cast<int>(I)) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return MismatchFound;
}
