#include "drake/geometry/proximity/volume_mesh.h"

#include <chrono>
#include <numeric>
#include <random>
#include <vector>

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <sycl/sycl.hpp>

#include "drake/math/rigid_transform.h"

using namespace drake;
using namespace drake::math;
using namespace drake::geometry;

DEFINE_int32(num_meshes, 1000, "Number of meshes to transform.");
DEFINE_int32(points_per_mesh, 100, "Number of points per mesh.");
DEFINE_int32(elements_per_mesh, 100, "Number of elements per mesh.");
DEFINE_int32(num_runs, 5,
             "Number of runs to include in average (excluding warm-up).");

// Populate allocated USM memory with random positions
void InitializeRandomPositions(Vector3<double>* p_MV, size_t num_points) {
  for (size_t i = 0; i < num_points; i++) {
    p_MV[i] = Vector3<double>(rand() % 100, rand() % 100, rand() % 100);
  }
}

// Populate allocated USM memory with random elements
void InitializeRandomElements(VolumeElement* elements, size_t num_elements,
                              size_t num_points) {
  // Assuming 4 vertices per element (tetrahedron)
  for (size_t i = 0; i < num_elements; i++) {
    elements[i] = VolumeElement(rand() % num_points, rand() % num_points,
                                rand() % num_points, rand() % num_points);
  }
}

void InitializeRandomTransforms(RigidTransformd* X_MBs, size_t num_meshes) {
  for (size_t i = 0; i < num_meshes; i++) {
    X_MBs[i] = RigidTransformd::Identity();
  }
}

// Run the mesh transformation benchmark with the given queue
std::pair<double, double> runBenchmarkOnce(sycl::queue& q,
                                           const std::string& device_name,
                                           size_t num_meshes,
                                           size_t points_per_mesh,
                                           size_t elements_per_mesh,
                                           bool print_results = true) {
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<VolumeMesh<double>> meshes;
  // Create the meshes with vertices and elements
  for (size_t i = 0; i < num_meshes; i++) {
    std::vector<Vector3<double>> mesh_vertices;
    mesh_vertices.resize(points_per_mesh, Vector3<double>(0, 0, 0));
    InitializeRandomPositions(mesh_vertices.data(), points_per_mesh);
    std::vector<VolumeElement> mesh_elements;
    mesh_elements.resize(elements_per_mesh, VolumeElement(0, 0, 0, 0));
    InitializeRandomElements(mesh_elements.data(), elements_per_mesh,
                             points_per_mesh);

    // Create the mesh
    meshes.emplace_back(std::move(mesh_elements), std::move(mesh_vertices));
  }

  // Create bunch of transforms
  std::vector<RigidTransformd> X_MBs(num_meshes);
  InitializeRandomTransforms(X_MBs.data(), num_meshes);

  // Move everything into device vectors
  auto meshes_d = sycl::malloc_device<VolumeMesh<double>>(num_meshes, q);
  auto X_MBs_d = sycl::malloc_device<RigidTransformd>(num_meshes, q);
  q.memcpy(meshes_d, meshes.data(), num_meshes * sizeof(VolumeMesh<double>));
  q.memcpy(X_MBs_d, X_MBs.data(), num_meshes * sizeof(RigidTransformd));

  // Wait for all the memcpy to finish
  q.wait();

  // Transform the meshes with a kernel
  // Parallelize across the meshes
  sycl::range<1> num_items{num_meshes};

  auto e = q.parallel_for(num_items, [=](auto idx) {
    // Get the mesh and transform
    VolumeMesh<double>& mesh = meshes_d[idx];
    RigidTransformd& X_MB = X_MBs_d[idx];
    mesh.TransformVertices(X_MB);
  });

  q.wait();
  // Profiler returns in nanoseconds, convert to milliseconds
  double kernel_time = (e.template get_profiling_info<
                            sycl::info::event_profiling::command_end>() -
                        e.template get_profiling_info<
                            sycl::info::event_profiling::command_start>()) /
                       1e6;

  // Deallocate memory in reverse order of allocation
  sycl::free(meshes_d, q);
  sycl::free(X_MBs_d, q);

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

void runBenchmark(sycl::queue& q, const std::string& device_name,
                  size_t num_meshes, size_t points_per_mesh,
                  size_t elements_per_mesh, int num_runs = 5) {
  fmt::print("{} benchmark:\n", device_name);

  std::vector<double> kernel_times;
  std::vector<double> total_times;

  // Add one extra run as warm-up but print all runs
  int total_runs = num_runs + 1;

  for (int i = 0; i < total_runs; i++) {
    fmt::print("  Run {} of {} {}:\n", i + 1, total_runs,
               (i == 0 ? "(warm-up, excluded from average)" : ""));
    auto [kernel_time, total_time] = runBenchmarkOnce(
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

  fmt::print("\n{} AVERAGE RESULTS (over {} runs, excluding warm-up):\n",
             device_name, num_runs);
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
    fmt::print(
        "Running on {} meshes with {} points per mesh and {} elements per "
        "mesh\n",
        num_meshes, points_per_mesh, elements_per_mesh);
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

    // Run benchmarks
    runBenchmark(gpu_queue, "GPU", num_meshes, points_per_mesh,
                 elements_per_mesh, num_runs);
    runBenchmark(cpu_queue, "CPU", num_meshes, points_per_mesh,
                 elements_per_mesh, num_runs);

    return 0;
  } catch (sycl::exception const& e) {
    fmt::print("SYCL exception caught: {}\n", e.what());
    return 1;
  } catch (std::exception const& e) {
    fmt::print("Standard exception caught: {}\n", e.what());
    return 1;
  }
}
