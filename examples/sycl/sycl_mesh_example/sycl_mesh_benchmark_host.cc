#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

constexpr size_t transform_size = 12;
constexpr size_t num_frames = 1000;
int main(int argc, char* argv[]) {
  size_t num_meshes = 30;
  size_t vertices_per_mesh = 4000;
  size_t vertices_per_work_item = 20;

  if (argc >= 4) {
    num_meshes = std::stoul(argv[1]);
    vertices_per_mesh = std::stoul(argv[2]);
    vertices_per_work_item = std::stoul(argv[3]);
  } else {
    std::cout << "Usage: " << argv[0]
              << " <num_meshes> <vertices_per_mesh> <vertices_per_work_item>\n";
    std::cout << "Using default values.\n";
  }

  std::vector<sycl::device> devices = sycl::device::get_devices();

  for (const auto& dev : devices) {
    std::cout << "\n== Running on device: "
              << dev.get_info<sycl::info::device::name>() << " ==\n";

    sycl::queue q{dev};

    double* mesh_vertices = sycl::malloc_device<double>(
        num_meshes * vertices_per_mesh * 3, q);  // XYZ per vertex

    std::vector<double> host_vertices(num_meshes * vertices_per_mesh * 3);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : host_vertices) v = dist(rng);

    q.memcpy(mesh_vertices, host_vertices.data(),
             sizeof(double) * host_vertices.size())
        .wait();

    double* transforms =
        sycl::malloc_host<double>(num_meshes * transform_size, q);

    for (size_t i = 0; i < num_meshes * transform_size; ++i)
      transforms[i] = 0.0;

    std::chrono::duration<double, std::milli> total_update_time{0};
    std::chrono::duration<double, std::milli> total_kernel_time{0};

    for (int frame = 0; frame <= static_cast<int>(num_frames); ++frame) {
      auto t0 = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < num_meshes; ++i) {
        for (size_t j = 0; j < transform_size; ++j) {
          transforms[i * transform_size + j] +=
              static_cast<double>(frame + i + j) * 0.001;
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      if (frame > 0) total_update_time += (t1 - t0);

      auto kernel_event = q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::range<1>{num_meshes * vertices_per_mesh /
                           vertices_per_work_item},
            [=](sycl::id<1> idx) {
              const size_t vertices_start = idx[0] * vertices_per_work_item;
              const size_t mesh_id = vertices_start / vertices_per_mesh;

              double T[transform_size];
              for (size_t j = 0; j < transform_size; ++j)
                T[j] = transforms[mesh_id * transform_size + j];

              for (size_t v = 0; v < vertices_per_work_item; ++v) {
                double& x = mesh_vertices[vertices_start * 3 + v * 3 + 0];
                double& y = mesh_vertices[vertices_start * 3 + v * 3 + 1];
                double& z = mesh_vertices[vertices_start * 3 + v * 3 + 2];

                double new_x = T[0] * x + T[1] * y + T[2] * z + T[3];
                double new_y = T[4] * x + T[5] * y + T[6] * z + T[7];
                double new_z = T[8] * x + T[9] * y + T[10] * z + T[11];

                x = new_x;
                y = new_y;
                z = new_z;
              }
            });
      });

      kernel_event.wait();

      auto t2 = std::chrono::high_resolution_clock::now();
      if (frame > 0) total_kernel_time += (t2 - t1);

      q.memcpy(host_vertices.data(), mesh_vertices,
               sizeof(double) * host_vertices.size())
          .wait();
    }

    std::cout << "Average CPU-side transform update time: "
              << (total_update_time.count() / num_frames) << " ms\n";
    std::cout << "Average kernel execution time: "
              << (total_kernel_time.count() / num_frames) << " ms\n";

    sycl::free(mesh_vertices, q);
    sycl::free(transforms, q);
  }

  return 0;
}