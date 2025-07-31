#pragma once

#include <vector>
#include <cstddef>
#include <sycl/sycl.hpp>

namespace drake {
namespace examples {
namespace simple {

typedef std::vector<int> IntVector; 

void print_platform_devices();
void InitializeVector(IntVector &a);
void InitializeArray(int *a, size_t size);
int vector_add_buffer();
int vector_add_usm();


void VectorAdd_buffer(sycl::queue &q, const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel, size_t num_repetitions);
void VectorAdd_usm(sycl::queue &q, int *a, int *b, int *sum_parallel, size_t size);
int sum_ids();



}  // namespace simple
}  // namespace examples
}  // namespace drake