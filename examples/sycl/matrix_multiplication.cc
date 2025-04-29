#include <chrono>
#include <iostream>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include <gflags/gflags.h>
#include <sycl/sycl.hpp>

// Parse command line arguments
DEFINE_int32(N, 1024, "Size of the matrix and vector to multiply");

// Compute c = A * b using Eigen.
void EigenMatMul(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                 Eigen::VectorXd* c) {
  (*c) = A * b;
}

// Compute c = A * b using a naive SYCL kernel.
void NaiveMatMul(sycl::queue& Q, const Eigen::MatrixXd& A,
                 const Eigen::VectorXd& b, Eigen::VectorXd* c) {
  // Create buffers for A, b, and c, initializing them to point at the raw data.
  const int N = A.rows();
  sycl::buffer A_buf{A.data(), sycl::range<2>(N, N)};
  sycl::buffer b_buf{b.data(), sycl::range<1>(N)};
  sycl::buffer c_buf{c->data(), sycl::range<1>(N)};

  // Submit a kernel to the queue.
  Q.submit([&](sycl::handler& cgh) {
    auto A_acc = A_buf.get_access<sycl::access::mode::read>(cgh);
    auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
    auto c_acc = c_buf.get_access<sycl::access::mode::write>(cgh);

    cgh.parallel_for(N, [=](auto i) {
      c_acc[i] = 0;
      for (int j = 0; j < N; ++j) {
        // Eigen uses column-major storage
        c_acc[i] += A_acc[j][i] * b_acc[j];
      }
    });
  });

  // Buffers will go out of scope at the end of this function, forcing the
  // result to be copied back to c on the host.
}

int main(int argc, char* argv[]) {
  // Parse command line arguments
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const int N = FLAGS_N;

  // Spin up a SYCL queue.
  sycl::queue Q;
  std::cout << "Using device "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // Generate a random test problem of size N.
  std::cout << "Generating random test problem of size " << N << "\n";
  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
  const Eigen::VectorXd b = Eigen::VectorXd::Random(N);

  std::cout << std::endl;

  // Compute c = A * b using Eigen, and time the operation.
  Eigen::VectorXd c_eigen(N);
  auto start = std::chrono::high_resolution_clock::now();
  EigenMatMul(A, b, &c_eigen);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Eigen time: " << elapsed.count() << " seconds\n";

  // Compute c = A * b using a naive SYCL kernel, and time the operation.
  Eigen::VectorXd c_sycl(N);
  start = std::chrono::high_resolution_clock::now();
  NaiveMatMul(Q, A, b, &c_sycl);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "SYCL time: " << elapsed.count() << " seconds\n";

  // Verify that the results are the same.
  DRAKE_DEMAND(c_eigen.isApprox(c_sycl));

  return 0;
}
