#include "drake/examples/sycl/simple.h"
#include <fmt/format.h>

int main() {
  const int result = drake::examples::simple::sum_ids();
  fmt::print("Result from calling sycl function: {}\n", result);

  const int result2 = drake::examples::simple::vector_add_buffer();

  const int result3 = drake::examples::simple::vector_add_usm();

  return 0;
}