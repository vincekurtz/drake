#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

constexpr size_t transform_size = 12;
constexpr size_t num_frames = 1000;

// Parse range specification of form "start end step+" or "start end step*"
std::vector<size_t> parse_range(const std::string& range_spec) {
  std::vector<size_t> values;
  std::istringstream ss(range_spec);
  size_t start, end;
  std::string step_str;

  if (ss >> start >> end >> step_str) {
    size_t step_val = std::stoul(step_str.substr(0, step_str.size() - 1));
    char op = step_str.back();

    if (op == '+') {
      for (size_t val = start; val <= end; val += step_val) {
        values.push_back(val);
      }
    } else if (op == '*') {
      for (size_t val = start; val <= end; val *= step_val) {
        values.push_back(val);
      }
    } else {
      std::cerr << "Invalid range operator. Use '+' for addition or '*' for "
                   "multiplication.\n";
      values = {start};  // Default to just the start value
    }
  } else {
    // If parsing fails, assume it's a single value
    values = {start};
  }

  return values;
}

int main(int argc, char* argv[]) {
  std::vector<size_t> num_meshes_values = {30};
  std::vector<size_t> vertices_per_mesh_values = {4000};
  std::vector<size_t> vertices_per_work_item_values = {20};
  std::string output_csv = "benchmark_results.csv";

  if (argc < 4) {
    std::cout
        << "Usage: " << argv[0]
        << " <meshes_range> <vertices_range> <work_items_range> [output_csv]\n";
    std::cout << "Range format: 'start end step+' for addition or 'start end "
                 "step*' for multiplication\n";
    std::cout << "Examples: '10 1000 10*' or '5 100 5+'\n";
    std::cout << "Single values are also accepted\n";
    std::cout << "Using default values.\n";
  } else {
    num_meshes_values = parse_range(argv[1]);
    vertices_per_mesh_values = parse_range(argv[2]);
    vertices_per_work_item_values = parse_range(argv[3]);

    if (argc >= 5) {
      output_csv = argv[4];
    }
  }

  // Create and initialize CSV file
  std::ofstream csv_file(output_csv);
  if (!csv_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_csv << std::endl;
    return 1;
  }

  csv_file << "meshes,vertices,vertices_per_work_item,"
           << "CPU_update_time,CPU_memcpy_time,GPU_kernel_time,GPU_memcpy_time,"
           << "device_name\n";

  std::vector<sycl::device> devices = sycl::device::get_devices();

  for (const auto& dev : devices) {
    std::string device_name = dev.get_info<sycl::info::device::name>();
    std::cout << "\n== Running on device: " << device_name << " ==\n";

    for (size_t num_meshes : num_meshes_values) {
      for (size_t vertices_per_mesh : vertices_per_mesh_values) {
        for (size_t vertices_per_work_item : vertices_per_work_item_values) {
          std::cout << "\nRunning with: " << num_meshes << " meshes, "
                    << vertices_per_mesh << " vertices/mesh, "
                    << vertices_per_work_item << " vertices/work_item\n";

          // Skip invalid configurations
          //   if (num_meshes * vertices_per_mesh % vertices_per_work_item != 0)
          //   {
          //     std::cout << "  Skipping: num_meshes * vertices_per_mesh must
          //     be "
          //                  "divisible by vertices_per_work_item\n";
          //     continue;
          //   }

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

          // Device allocated transforms
          double* transforms =
              sycl::malloc_device<double>(num_meshes * transform_size, q);

          // Host side transforms for updating
          std::vector<double> host_transforms(num_meshes * transform_size, 0.0);

          std::chrono::duration<double, std::milli> total_update_time{0};
          std::chrono::duration<double, std::milli> total_kernel_time{0};
          std::chrono::duration<double, std::milli> total_memcpy_time{0};

          for (int frame = 0; frame <= static_cast<int>(num_frames); ++frame) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < num_meshes; ++i) {
              for (size_t j = 0; j < transform_size; ++j) {
                host_transforms[i * transform_size + j] +=
                    static_cast<double>(frame + i + j) * dist(rng);
              }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (frame > 0) total_update_time += (t1 - t0);

            // Memcpy the transforms to device
            auto memcpy_start = std::chrono::high_resolution_clock::now();
            q.memcpy(transforms, host_transforms.data(),
                     sizeof(double) * host_transforms.size())
                .wait();
            auto memcpy_end = std::chrono::high_resolution_clock::now();
            if (frame > 0) total_memcpy_time += (memcpy_end - memcpy_start);

            auto kernel_event = q.submit([&](sycl::handler& h) {
              h.parallel_for(
                  sycl::range<1>{num_meshes * vertices_per_mesh /
                                 vertices_per_work_item},
                  [=](sycl::id<1> idx) {
                    const size_t vertices_start =
                        idx[0] * vertices_per_work_item;
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
            if (frame > 0) total_kernel_time += (t2 - memcpy_end);

            q.memcpy(host_vertices.data(), mesh_vertices,
                     sizeof(double) * host_vertices.size())
                .wait();
          }

          double cpu_update_time = total_update_time.count() / num_frames;
          double cpu_memcpy_time = total_memcpy_time.count() / num_frames;
          double gpu_kernel_time = total_kernel_time.count() / num_frames;

          std::cout << "Average CPU-side transform update time: "
                    << cpu_update_time << " ms\n";
          std::cout << "Average transforms memcpy time: " << cpu_memcpy_time
                    << " ms\n";
          std::cout << "Average kernel execution time: " << gpu_kernel_time
                    << " ms\n";

          // Write results to CSV
          csv_file
              << num_meshes << "," << vertices_per_mesh << ","
              << vertices_per_work_item << "," << cpu_update_time << ","
              << cpu_memcpy_time << "," << gpu_kernel_time << "," << 0.0
              << ","  // GPU memcpy time (currently not measured separately)
              << "\"" << device_name << "\""
              << "\n";
          csv_file.flush();

          sycl::free(mesh_vertices, q);
          sycl::free(transforms, q);
        }
      }
    }
  }

  csv_file.close();
  std::cout << "Results written to: " << output_csv << std::endl;
  return 0;
}