#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "drake/examples/sycl/simple_mesh_example/simple_mesh.h"
#include "drake/math/rigid_transform.h"

using namespace drake;
using namespace drake::math;

// Default values when sweeping across parameter space
DEFINE_int32(sweep, 1, "Run full parameter sweep (1) or fixed parameters (0)");
DEFINE_int32(num_meshes, 0, "Fixed number of meshes (0 to sweep)");
DEFINE_int32(points_per_mesh, 0, "Fixed points per mesh (0 to sweep)");
DEFINE_int32(elements_per_mesh, 0, "Fixed elements per mesh (0 to sweep)");
DEFINE_int32(num_runs, 5,
             "Number of runs to include in average (excluding warm-up)");
DEFINE_bool(use_sycl_vec, false,
            "Use sycl::vec<double, 3> instead of Vector3<double>.");
DEFINE_bool(
    use_host_memory, false,
    "Use malloc_host instead of malloc_shared for USM memory allocation.");

// Populate allocated USM memory with random positions - templated version
template <typename VectorType>
void InitializeRandomPositions(VectorType* p_MV, size_t num_points) {
  for (size_t i = 0; i < num_points; i++) {
    if constexpr (std::is_same_v<VectorType, Vector3<double>>) {
      p_MV[i] = Vector3<double>(rand() % 100, rand() % 100, rand() % 100);
    } else {
      p_MV[i] = sycl::vec<double, 3>(rand() % 100, rand() % 100, rand() % 100);
    }
  }
}

// Populate allocated USM memory with random elements
void InitializeRandomElements(int* elements, size_t num_elements) {
  // Assuming 4 vertices per element (tet)
  for (size_t i = 0; i < num_elements * 4; i++) {
    elements[i] = rand() % 100;
  }
}

// Run the mesh transformation benchmark with the given queue - templated
// version
template <typename VectorType>
std::pair<double, double> runBenchmarkOnce(sycl::queue& q,
                                           const std::string& device_name,
                                           size_t num_meshes,
                                           size_t points_per_mesh,
                                           size_t elements_per_mesh,
                                           bool print_results = false) {
  auto start = std::chrono::high_resolution_clock::now();

  // Allocate memory for meshes using USM
  SimpleMesh<VectorType>* meshes =
      FLAGS_use_host_memory
          ? sycl::malloc_host<SimpleMesh<VectorType>>(num_meshes, q)
          : sycl::malloc_shared<SimpleMesh<VectorType>>(num_meshes, q);

  // Allocate memory for transforms
  RigidTransformd* X_MBs =
      FLAGS_use_host_memory
          ? sycl::malloc_host<RigidTransformd>(num_meshes, q)
          : sycl::malloc_shared<RigidTransformd>(num_meshes, q);

  // Allocate memory for vertices and elements directly using USM
  VectorType** vertex_arrays =
      FLAGS_use_host_memory ? sycl::malloc_host<VectorType*>(num_meshes, q)
                            : sycl::malloc_shared<VectorType*>(num_meshes, q);
  int** element_arrays = FLAGS_use_host_memory
                             ? sycl::malloc_host<int*>(num_meshes, q)
                             : sycl::malloc_shared<int*>(num_meshes, q);

  // Initialize data and create meshes with pre-allocated memory
  for (size_t i = 0; i < num_meshes; i++) {
    // Allocate memory for vertices
    vertex_arrays[i] =
        FLAGS_use_host_memory
            ? sycl::malloc_host<VectorType>(points_per_mesh, q)
            : sycl::malloc_shared<VectorType>(points_per_mesh, q);
    InitializeRandomPositions(vertex_arrays[i], points_per_mesh);

    // Allocate memory for elements (4 vertices per element)
    element_arrays[i] =
        FLAGS_use_host_memory
            ? sycl::malloc_host<int>(elements_per_mesh * 4, q)
            : sycl::malloc_shared<int>(elements_per_mesh * 4, q);
    InitializeRandomElements(element_arrays[i], elements_per_mesh);

    // Create mesh with pre-allocated memory
    meshes[i] = SimpleMesh<VectorType>(vertex_arrays[i], element_arrays[i],
                                       points_per_mesh, elements_per_mesh);

    // Initialize transform
    X_MBs[i] = RigidTransformd::Identity();
  }

  // Transform the meshes with a kernel
  // Parallelize across the meshes
  sycl::range<1> num_items{num_meshes};

  auto e = q.parallel_for(num_items, [=](auto idx) {
    // Get the mesh and transform
    SimpleMesh<VectorType>& mesh = meshes[idx];
    const auto& transform = X_MBs[idx].GetAsMatrix34();

    // Extract transformation matrix elements
    const double r00 = transform(0, 0), r01 = transform(0, 1),
                 r02 = transform(0, 2), tx = transform(0, 3);
    const double r10 = transform(1, 0), r11 = transform(1, 1),
                 r12 = transform(1, 2), ty = transform(1, 3);
    const double r20 = transform(2, 0), r21 = transform(2, 1),
                 r22 = transform(2, 2), tz = transform(2, 3);

    // Apply the transform to the mesh
    for (size_t i = 0; i < mesh.num_points(); ++i) {
      if constexpr (std::is_same_v<VectorType, Vector3<double>>) {
        const double x = mesh.p_MV()[i][0];
        const double y = mesh.p_MV()[i][1];
        const double z = mesh.p_MV()[i][2];

        mesh.p_MV()[i][0] = r00 * x + r01 * y + r02 * z + tx;
        mesh.p_MV()[i][1] = r10 * x + r11 * y + r12 * z + ty;
        mesh.p_MV()[i][2] = r20 * x + r21 * y + r22 * z + tz;
      } else {
        const double x = mesh.p_MV()[i].x();
        const double y = mesh.p_MV()[i].y();
        const double z = mesh.p_MV()[i].z();

        mesh.p_MV()[i] = sycl::vec<double, 3>(r00 * x + r01 * y + r02 * z + tx,
                                              r10 * x + r11 * y + r12 * z + ty,
                                              r20 * x + r21 * y + r22 * z + tz);
      }
    }
  });

  q.wait();

  // Profiler returns in nanoseconds, convert to milliseconds
  double kernel_time = (e.template get_profiling_info<
                            sycl::info::event_profiling::command_end>() -
                        e.template get_profiling_info<
                            sycl::info::event_profiling::command_start>()) /
                       1e6;
  // Deallocate memory in reverse order of allocation
  for (size_t i = 0; i < num_meshes; i++) {
    sycl::free(element_arrays[i], q);
    sycl::free(vertex_arrays[i], q);
  }
  sycl::free(element_arrays, q);
  sycl::free(vertex_arrays, q);
  sycl::free(X_MBs, q);
  sycl::free(meshes, q);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> total_duration = end - start;

  if (print_results) {
    fmt::print("  Run results:\n");
    fmt::print("    - Kernel execution time: {:.3f} ms\n", kernel_time);
    fmt::print("    - Total execution time:  {:.3f} ms\n",
               total_duration.count());
  }

  return {kernel_time, total_duration.count()};
}

template <typename VectorType>
std::pair<double, double> runBenchmark(
    sycl::queue& q, const std::string& device_name, size_t num_meshes,
    size_t points_per_mesh, size_t elements_per_mesh, int num_runs = 5,
    bool print_details = false) {
  if (print_details) {
    const char* vector_type_name =
        std::is_same_v<VectorType, Vector3<double>> ? "Vector3" : "sycl::vec";
    fmt::print(
        "{} benchmark for meshes={}, points={}, elements={} (using {}):\n",
        device_name, num_meshes, points_per_mesh, elements_per_mesh,
        vector_type_name);
  }

  std::vector<double> kernel_times;
  std::vector<double> total_times;

  // Add one extra run as warm-up but print all runs
  int total_runs = num_runs + 1;

  for (int i = 0; i < total_runs; i++) {
    if (print_details) {
      fmt::print("  Run {} of {} {}:\n", i + 1, total_runs,
                 (i == 0 ? "(warm-up, excluded from average)" : ""));
    }
    auto [kernel_time, total_time] = runBenchmarkOnce<VectorType>(
        q, device_name, num_meshes, points_per_mesh, elements_per_mesh,
        print_details);

    // Only include the last num_runs (exclude the first/warm-up run)
    if (i > 0) {
      kernel_times.push_back(kernel_time);
      total_times.push_back(total_time);
    }
  }

  // Calculate averages (only from runs after the warm-up)
  double avg_kernel_time =
      std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0) / num_runs;
  double avg_total_time =
      std::accumulate(total_times.begin(), total_times.end(), 0.0) / num_runs;

  if (print_details) {
    const char* vector_type_name =
        std::is_same_v<VectorType, Vector3<double>> ? "Vector3" : "sycl::vec";
    fmt::print(
        "\n{} AVERAGE RESULTS (over {} runs, excluding warm-up, using {}):\n",
        device_name, num_runs, vector_type_name);
    fmt::print("  - Average kernel execution time: {:.3f} ms\n",
               avg_kernel_time);
    fmt::print("  - Average total execution time:  {:.3f} ms\n\n",
               avg_total_time);
  }

  return {avg_kernel_time, avg_total_time};
}

// Helper function to run benchmarks with the selected vector type
template <typename VectorType>
void runBenchmarksForType(sycl::queue& gpu_queue, sycl::queue& cpu_queue,
                          const std::vector<size_t>& num_meshes_values,
                          const std::vector<size_t>& points_per_mesh_values,
                          const std::vector<size_t>& elements_per_mesh_values,
                          std::ofstream& results_file, size_t& completed,
                          size_t total_combinations) {
  const char* vector_type_suffix =
      std::is_same_v<VectorType, Vector3<double>> ? "_eigen" : "_sycl";

  // Test all combinations with both GPU and CPU
  for (size_t num_meshes : num_meshes_values) {
    for (size_t points_per_mesh : points_per_mesh_values) {
      for (size_t elements_per_mesh : elements_per_mesh_values) {
        // Skip combinations that could cause out-of-memory issues
        if (num_meshes * points_per_mesh * elements_per_mesh > 1e9) {
          fmt::print(
              "Skipping potentially out-of-memory combination: meshes={}, "
              "points={}, elements={}\n",
              num_meshes, points_per_mesh, elements_per_mesh);
          completed += 2;  // Count as completed for both GPU and CPU
          continue;
        }

        fmt::print(
            "Running benchmark ({}/{}) for meshes={}, points={}, "
            "elements={} ({})\n",
            completed + 1, total_combinations, num_meshes, points_per_mesh,
            elements_per_mesh, vector_type_suffix);

        try {
          // GPU benchmark
          auto [gpu_kernel_time, gpu_total_time] = runBenchmark<VectorType>(
              gpu_queue, "GPU", num_meshes, points_per_mesh, elements_per_mesh,
              FLAGS_num_runs);

          // Write GPU results to CSV
          results_file << "GPU" << vector_type_suffix << "," << num_meshes
                       << "," << points_per_mesh << "," << elements_per_mesh
                       << "," << gpu_kernel_time << "," << gpu_total_time
                       << "\n";
          completed++;

          fmt::print(
              "Running benchmark ({}/{}) for meshes={}, points={}, "
              "elements={} ({})\n",
              completed + 1, total_combinations, num_meshes, points_per_mesh,
              elements_per_mesh, vector_type_suffix);

          // CPU benchmark
          auto [cpu_kernel_time, cpu_total_time] = runBenchmark<VectorType>(
              cpu_queue, "CPU", num_meshes, points_per_mesh, elements_per_mesh,
              FLAGS_num_runs);

          // Write CPU results to CSV
          results_file << "CPU" << vector_type_suffix << "," << num_meshes
                       << "," << points_per_mesh << "," << elements_per_mesh
                       << "," << cpu_kernel_time << "," << cpu_total_time
                       << "\n";
          completed++;

          // Flush results to file periodically
          results_file.flush();

        } catch (std::exception const& e) {
          fmt::print(
              "Exception during benchmark for meshes={}, points={}, "
              "elements={}: {}\n",
              num_meshes, points_per_mesh, elements_per_mesh, e.what());
          completed += 2;  // Count as completed for both GPU and CPU
        }
      }
    }
  }
}

// Construct a bunch of simple meshes and do transforms on them using SYCL
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Default parameter values for sweeping
  std::vector<size_t> parameter_values = {1,     10,     100,    1000,
                                          10000, 100000, 1000000};

  // Initialize parameter vectors based on flags
  std::vector<size_t> num_meshes_values;
  std::vector<size_t> points_per_mesh_values;
  std::vector<size_t> elements_per_mesh_values;

  // If sweep is enabled, use the full parameter range for any parameter set to
  // 0 Otherwise, use the single value specified (or 1 if 0 was specified)
  if (FLAGS_sweep) {
    // For each parameter, either use the full range or just the specified value
    if (FLAGS_num_meshes == 0) {
      num_meshes_values = parameter_values;
    } else {
      num_meshes_values = {static_cast<size_t>(FLAGS_num_meshes)};
    }

    if (FLAGS_points_per_mesh == 0) {
      points_per_mesh_values = parameter_values;
    } else {
      points_per_mesh_values = {static_cast<size_t>(FLAGS_points_per_mesh)};
    }

    if (FLAGS_elements_per_mesh == 0) {
      elements_per_mesh_values = parameter_values;
    } else {
      elements_per_mesh_values = {static_cast<size_t>(FLAGS_elements_per_mesh)};
    }
  } else {
    // In non-sweep mode, use specified values (default to 1 if 0)
    num_meshes_values = {
        static_cast<size_t>(FLAGS_num_meshes > 0 ? FLAGS_num_meshes : 1)};
    points_per_mesh_values = {static_cast<size_t>(
        FLAGS_points_per_mesh > 0 ? FLAGS_points_per_mesh : 1)};
    elements_per_mesh_values = {static_cast<size_t>(
        FLAGS_elements_per_mesh > 0 ? FLAGS_elements_per_mesh : 1)};
  }

  // Report benchmark configuration
  const char* vector_type =
      FLAGS_use_sycl_vec ? "sycl::vec<double, 3>" : "Vector3<double>";
  const char* memory_type =
      FLAGS_use_host_memory ? "malloc_host" : "malloc_shared";
  if (FLAGS_sweep) {
    fmt::print("Running parameter sweep benchmark with:\n");
    if (num_meshes_values.size() == 1) {
      fmt::print("  - Fixed number of meshes: {}\n", num_meshes_values[0]);
    } else {
      fmt::print("  - Variable number of meshes: {} - {}\n",
                 parameter_values.front(), parameter_values.back());
    }

    if (points_per_mesh_values.size() == 1) {
      fmt::print("  - Fixed points per mesh: {}\n", points_per_mesh_values[0]);
    } else {
      fmt::print("  - Variable points per mesh: {} - {}\n",
                 parameter_values.front(), parameter_values.back());
    }

    if (elements_per_mesh_values.size() == 1) {
      fmt::print("  - Fixed elements per mesh: {}\n",
                 elements_per_mesh_values[0]);
    } else {
      fmt::print("  - Variable elements per mesh: {} - {}\n",
                 parameter_values.front(), parameter_values.back());
    }
  } else {
    fmt::print("Running fixed parameter benchmark with:\n");
    fmt::print("  - Number of meshes: {}\n", num_meshes_values[0]);
    fmt::print("  - Points per mesh: {}\n", points_per_mesh_values[0]);
    fmt::print("  - Elements per mesh: {}\n", elements_per_mesh_values[0]);
  }
  fmt::print("  - Vector type: {}\n", vector_type);
  fmt::print("  - Memory allocation: {}\n", memory_type);

  try {
    // Create a descriptive filename based on what parameters were fixed
    std::string result_description = "";
    if (num_meshes_values.size() == 1) {
      result_description += fmt::format("_meshes{}", num_meshes_values[0]);
    }
    if (points_per_mesh_values.size() == 1) {
      result_description += fmt::format("_points{}", points_per_mesh_values[0]);
    }
    if (elements_per_mesh_values.size() == 1) {
      result_description +=
          fmt::format("_elements{}", elements_per_mesh_values[0]);
    }

    // Add vector type and memory type to filename
    const char* vector_suffix = FLAGS_use_sycl_vec ? "_syclvec" : "_eigen";
    const char* memory_suffix = FLAGS_use_host_memory ? "_host" : "_shared";

    // If all parameters vary, use the default name
    std::string output_filename =
        result_description.empty()
            ? fmt::format("mesh_benchmark{}{}_results.csv", vector_suffix,
                          memory_suffix)
            : fmt::format("mesh_benchmark{}{}{}_results.csv",
                          result_description, vector_suffix, memory_suffix);

    // Create output file for results
    std::ofstream results_file(output_filename);
    if (!results_file) {
      fmt::print("Error opening results file: {}!\n", output_filename);
      return 1;
    }

    // Write CSV header
    results_file << "device,num_meshes,points_per_mesh,elements_per_mesh,"
                    "kernel_time_ms,total_time_ms\n";

    // Create GPU queue
    sycl::queue gpu_queue(sycl::gpu_selector_v,
                          sycl::property::queue::enable_profiling{});
    fmt::print("Running on GPU: {}\n",
               gpu_queue.get_device().get_info<sycl::info::device::name>());

    // Create CPU queue
    sycl::queue cpu_queue(sycl::cpu_selector_v,
                          sycl::property::queue::enable_profiling{});
    fmt::print("Running on CPU: {}\n",
               cpu_queue.get_device().get_info<sycl::info::device::name>());

    fmt::print("\nRunning benchmark for specified parameter combinations...\n");
    fmt::print("Each configuration will run {} times plus one warm-up run\n",
               FLAGS_num_runs);

    // Total number of combinations (multiply by 2 for GPU and CPU)
    size_t total_combinations = num_meshes_values.size() *
                                points_per_mesh_values.size() *
                                elements_per_mesh_values.size() * 2;
    size_t completed = 0;

    // Run benchmarks with the selected vector type
    if (FLAGS_use_sycl_vec) {
      runBenchmarksForType<sycl::vec<double, 3>>(
          gpu_queue, cpu_queue, num_meshes_values, points_per_mesh_values,
          elements_per_mesh_values, results_file, completed,
          total_combinations);
    } else {
      runBenchmarksForType<Vector3<double>>(
          gpu_queue, cpu_queue, num_meshes_values, points_per_mesh_values,
          elements_per_mesh_values, results_file, completed,
          total_combinations);
    }

    fmt::print("\nBenchmark completed. Results written to {}\n",
               output_filename);

    return 0;
  } catch (sycl::exception const& e) {
    fmt::print("SYCL exception caught: {}\n", e.what());
    return 1;
  } catch (std::exception const& e) {
    fmt::print("Standard exception caught: {}\n", e.what());
    return 1;
  }
}
