#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif
  
// Check if we have open_mp available. Note that we need to use --config=omp
// option when compiling.
#if defined(_OPENMP)
constexpr bool has_openmp = true;
#else
constexpr bool has_openmp = false;
int omp_get_thread_num() { return 0; }
#endif

int main() {

  std::cout << "has_openmp: " << has_openmp << std::endl;

  // Run a simple for loop that doubles the values in a list
  const int N = 20;
  std::vector<double> values;
  for (int i=0; i<N; ++i) {
    values.push_back(i);
  }

  // Record which omp thread computed which value
  std::vector<int> outputs(N, 0);

  auto start_time = std::chrono::high_resolution_clock::now();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i=0; i<N; ++i) {
    values[i] = 2*values[i];
    outputs[i] = omp_get_thread_num();
    // Sleep for a while to pretend this was complicated
    std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_time = end_time-start_time;   
  std::cout << "Time: " << total_time.count() << " seconds" << std::endl;

  std::cout << "Values: " << std::endl;
  for (int i=0; i<N; ++i) {
    std::cout << values[i] << ", ";
  }
  std::cout << std::endl;
  
  std::cout << "Thread used to compute each value: " << std::endl;
  for (int i=0; i<N; ++i) {
    std::cout << outputs[i] << ", ";
  }
  std::cout << std::endl;

  return 0;
}
