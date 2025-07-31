#include <chrono>
#include <numeric>
#include <random>
#include <vector>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "drake/examples/sycl/simple_mesh_example/simple_mesh.h"
#include "drake/math/rigid_transform.h"

using namespace drake;
using namespace drake::math;

DEFINE_int32(num_meshes, 1000, "Number of meshes to transform.");
DEFINE_int32(points_per_mesh, 100, "Number of points per mesh.");
DEFINE_int32(elements_per_mesh, 100, "Number of elements per mesh.");
DEFINE_int32(num_runs, 5,
             "Number of runs to include in average (excluding warm-up).");
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
                                           bool print_results = true) {
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
    // fmt::print("    - Kernel execution time: {:.3f} ms\n",
    // kernel_duration.count());
    fmt::print("    - Kernel execution time: {:.3f} ms\n", kernel_time);
    fmt::print("    - Total execution time:  {:.3f} ms\n",
               total_duration.count());
  }

  // return {kernel_duration.count(), total_duration.count()};
  return {kernel_time, total_duration.count()};
}

template <typename VectorType>
void runBenchmark(sycl::queue& q, const std::string& device_name,
                  size_t num_meshes, size_t points_per_mesh,
                  size_t elements_per_mesh, int num_runs = 5) {
  const char* vector_type_name =
      std::is_same_v<VectorType, Vector3<double>> ? "Vector3" : "sycl::vec";
  fmt::print("{} benchmark (using {}):\n", device_name, vector_type_name);

  std::vector<double> kernel_times;
  std::vector<double> total_times;

  // Add one extra run as warm-up but print all runs
  int total_runs = num_runs + 1;

  for (int i = 0; i < total_runs; i++) {
    fmt::print("  Run {} of {} {}:\n", i + 1, total_runs,
               (i == 0 ? "(warm-up, excluded from average)" : ""));
    auto [kernel_time, total_time] = runBenchmarkOnce<VectorType>(
        q, device_name, num_meshes, points_per_mesh, elements_per_mesh);

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

  fmt::print(
      "\n{} AVERAGE RESULTS (over {} runs, excluding warm-up, using {}):\n",
      device_name, num_runs, vector_type_name);
  fmt::print("  - Average kernel execution time: {:.3f} ms\n", avg_kernel_time);
  fmt::print("  - Average total execution time:  {:.3f} ms\n\n",
             avg_total_time);
}

// Construct a bunch of simple meshes and do transforms on them using SYCL
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  size_t num_meshes = FLAGS_num_meshes;
  size_t points_per_mesh = FLAGS_points_per_mesh;
  size_t elements_per_mesh = FLAGS_elements_per_mesh;
  int num_runs = FLAGS_num_runs;

  try {
    const char* vector_type =
        FLAGS_use_sycl_vec ? "sycl::vec<double, 3>" : "Vector3<double>";
    const char* memory_type =
        FLAGS_use_host_memory ? "malloc_host" : "malloc_shared";
    fmt::print(
        "Running on {} meshes with {} points per mesh and {} elements per "
        "mesh\n",
        num_meshes, points_per_mesh, elements_per_mesh);
    fmt::print("Using vector type: {}\n", vector_type);
    fmt::print("Using memory allocation: {}\n", memory_type);
    fmt::print(
        "Each benchmark will run {} times plus one warm-up run (total: {})\n",
        num_runs, num_runs + 1);
    fmt::print("Warm-up run will be excluded from average calculations\n\n");

    // Create GPU queue
    // sycl::queue gpu_queue(sycl::gpu_selector_v);
    sycl::queue gpu_queue(sycl::gpu_selector_v,
                          sycl::property::queue::enable_profiling{});
    fmt::print("Running on GPU: {}\n",
               gpu_queue.get_device().get_info<sycl::info::device::name>());

    // Create CPU queue
    // sycl::queue cpu_queue(sycl::cpu_selector_v);
    sycl::queue cpu_queue(sycl::cpu_selector_v,
                          sycl::property::queue::enable_profiling{});
    fmt::print("Running on CPU: {}\n",
               cpu_queue.get_device().get_info<sycl::info::device::name>());

    fmt::print("\nRunning performance comparison...\n\n");

    // Run benchmarks with the selected vector type
    if (FLAGS_use_sycl_vec) {
      runBenchmark<sycl::vec<double, 3>>(gpu_queue, "GPU", num_meshes,
                                         points_per_mesh, elements_per_mesh,
                                         num_runs);
      // runBenchmark<sycl::vec<double, 3>>(cpu_queue, "CPU", num_meshes,
      //                                    points_per_mesh, elements_per_mesh,
      //                                    num_runs);
    } else {
      runBenchmark<Vector3<double>>(gpu_queue, "GPU", num_meshes,
                                    points_per_mesh, elements_per_mesh,
                                    num_runs);
      // runBenchmark<Vector3<double>>(cpu_queue, "CPU", num_meshes,
      //                               points_per_mesh, elements_per_mesh,
      //                               num_runs);
    }

    return 0;
  } catch (sycl::exception const& e) {
    fmt::print("SYCL exception caught: {}\n", e.what());
    return 1;
  } catch (std::exception const& e) {
    fmt::print("Standard exception caught: {}\n", e.what());
    return 1;
  }
}
