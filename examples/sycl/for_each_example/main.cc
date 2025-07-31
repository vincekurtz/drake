#include <algorithm>
#include <cassert>
#include <chrono>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>

// Helper function to compare two vectors element-wise
bool compare_vectors(const std::vector<uint32_t>& vec1,
                     const std::vector<uint32_t>& vec2,
                     const std::string& name) {
  if (vec1.size() != vec2.size()) {
    std::cout << "ERROR: " << name << " sizes don't match: " << vec1.size()
              << " vs " << vec2.size() << std::endl;
    return false;
  }

  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec1[i] != vec2[i]) {
      std::cout << "ERROR: " << name << " mismatch at index " << i << ": "
                << vec1[i] << " vs " << vec2[i] << std::endl;
      return false;
    }
  }
  return true;
}

// Helper function to print first few elements of a vector
void print_vector_preview(const std::vector<uint32_t>& vec,
                          const std::string& name, size_t max_elements = 10) {
  std::cout << name << " (first " << std::min(max_elements, vec.size())
            << " elements): ";
  for (size_t i = 0; i < std::min(max_elements, vec.size()); ++i) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  size_t num_meshes = 10;           // Reduced for testing
  size_t elements_per_mesh = 1000;  // Reduced for testing
  if (argc >= 3) {
    num_meshes = std::stoul(argv[1]);
    elements_per_mesh = std::stoul(argv[2]);
  } else {
    std::cout << "Usage: " << argv[0] << " <num_meshes> <elements_per_mesh>\n";
    std::cout << "Using default values: num_meshes=" << num_meshes
              << ", elements_per_mesh=" << elements_per_mesh << "\n";
  }
  size_t total_elements = num_meshes * elements_per_mesh;

  sycl::queue q(sycl::gpu_selector_v);
  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // Generate random keys and indices
  std::vector<uint32_t> keysAll(total_elements);
  std::vector<uint32_t> indicesAll(total_elements);
  std::mt19937 rng(42);
  std::uniform_int_distribution<uint32_t> key_dist(0, 1000000);
  for (size_t i = 0; i < total_elements; ++i) {
    keysAll[i] = key_dist(rng);
    indicesAll[i] = i;
  }

  // Create reference data for comparison
  std::vector<uint32_t> reference_keys = keysAll;
  std::vector<uint32_t> reference_indices = indicesAll;

  // Offsets and counts for each mesh
  std::vector<uint32_t> element_offsets(num_meshes);
  std::vector<uint32_t> element_counts(num_meshes, elements_per_mesh);
  for (size_t i = 0; i < num_meshes; ++i) {
    element_offsets[i] = i * elements_per_mesh;
  }

  // Sort reference data using std::sort for each mesh
  std::cout << "\n=== Sorting reference data with std::sort ===" << std::endl;
  for (size_t mesh_id = 0; mesh_id < num_meshes; ++mesh_id) {
    uint32_t start = element_offsets[mesh_id];
    uint32_t count = element_counts[mesh_id];

    // Create pairs for sort_by_key equivalent
    std::vector<std::pair<uint32_t, uint32_t>> pairs(count);
    for (size_t i = 0; i < count; ++i) {
      pairs[i] = {reference_keys[start + i], reference_indices[start + i]};
    }

    // Sort by key
    std::sort(pairs.begin(), pairs.end());

    // Update reference arrays
    for (size_t i = 0; i < count; ++i) {
      reference_keys[start + i] = pairs[i].first;
      reference_indices[start + i] = pairs[i].second;
    }
  }

  // Copy data to device
  uint32_t* d_keysAll = sycl::malloc_device<uint32_t>(total_elements, q);
  uint32_t* d_indicesAll = sycl::malloc_device<uint32_t>(total_elements, q);
  uint32_t* d_element_offsets = sycl::malloc_host<uint32_t>(num_meshes, q);
  uint32_t* d_element_counts = sycl::malloc_host<uint32_t>(num_meshes, q);
  q.memcpy(d_keysAll, keysAll.data(), total_elements * sizeof(uint32_t)).wait();
  q.memcpy(d_indicesAll, indicesAll.data(), total_elements * sizeof(uint32_t))
      .wait();
  q.memcpy(d_element_offsets, element_offsets.data(),
           num_meshes * sizeof(uint32_t))
      .wait();
  q.memcpy(d_element_counts, element_counts.data(),
           num_meshes * sizeof(uint32_t))
      .wait();

  // Sort using SYCL/oneAPI DPL
  std::cout << "\n=== Sorting with SYCL/oneAPI DPL ===" << std::endl;
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  std::vector<int> mesh_ids(num_meshes);
  std::iota(mesh_ids.begin(), mesh_ids.end(), 0);

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int mesh_id : mesh_ids) {
    uint32_t this_geom_start = d_element_offsets[mesh_id];
    uint32_t this_geom_count = d_element_counts[mesh_id];
    auto keys_begin = d_keysAll + this_geom_start;
    auto keys_end = keys_begin + this_geom_count;
    auto indices_begin = d_indicesAll + this_geom_start;
    oneapi::dpl::sort_by_key(policy, keys_begin, keys_end, indices_begin);

    uint32_t* random_allo1 = sycl::malloc_device<uint32_t>(this_geom_count, q);
    q.memset(random_allo1, 0, this_geom_count * sizeof(uint32_t)).wait();
    uint32_t* random_alloc2 = sycl::malloc_device<uint32_t>(this_geom_count, q);
    q.memset(random_alloc2, 0, this_geom_count * sizeof(uint32_t)).wait();
    sycl::free(random_allo1, q);
    sycl::free(random_alloc2, q);
  }

  q.wait();
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "Sort-by-key for " << num_meshes << " meshes of "
            << elements_per_mesh << " elements took " << ms << " ms\n";

  // Copy results back from device
  std::vector<uint32_t> sycl_keys(total_elements);
  std::vector<uint32_t> sycl_indices(total_elements);
  q.memcpy(sycl_keys.data(), d_keysAll, total_elements * sizeof(uint32_t))
      .wait();
  q.memcpy(sycl_indices.data(), d_indicesAll, total_elements * sizeof(uint32_t))
      .wait();

  // Element-wise comparison
  std::cout << "\n=== Element-wise Comparison ===" << std::endl;

  // Compare keys
  bool keys_match = compare_vectors(reference_keys, sycl_keys, "Keys");
  if (keys_match) {
    std::cout << "✓ Keys match perfectly!" << std::endl;
  } else {
    std::cout << "✗ Keys do not match!" << std::endl;
    print_vector_preview(reference_keys, "Reference keys");
    print_vector_preview(sycl_keys, "SYCL keys");
  }

  // Compare indices
  bool indices_match =
      compare_vectors(reference_indices, sycl_indices, "Indices");
  if (indices_match) {
    std::cout << "✓ Indices match perfectly!" << std::endl;
  } else {
    std::cout << "✗ Indices do not match!" << std::endl;
    print_vector_preview(reference_indices, "Reference indices");
    print_vector_preview(sycl_indices, "SYCL indices");
  }

  // Detailed analysis for first mesh
  if (num_meshes > 0) {
    std::cout << "\n=== Detailed Analysis of First Mesh ===" << std::endl;
    size_t first_mesh_start = element_offsets[0];
    size_t first_mesh_count = element_counts[0];

    std::cout << "First mesh: " << first_mesh_count
              << " elements starting at index " << first_mesh_start
              << std::endl;

    // Check if keys are sorted
    bool keys_sorted = true;
    for (size_t i = first_mesh_start + 1;
         i < first_mesh_start + first_mesh_count; ++i) {
      if (sycl_keys[i] < sycl_keys[i - 1]) {
        std::cout << "ERROR: Keys not sorted at index " << i << ": "
                  << sycl_keys[i - 1] << " > " << sycl_keys[i] << std::endl;
        keys_sorted = false;
        break;
      }
    }
    if (keys_sorted) {
      std::cout << "✓ Keys are properly sorted" << std::endl;
    }

    // Check if indices correspond to sorted keys
    bool indices_correct = true;
    for (size_t i = first_mesh_start + 1;
         i < first_mesh_start + first_mesh_count; ++i) {
      if (sycl_keys[i] == sycl_keys[i - 1] &&
          sycl_indices[i] < sycl_indices[i - 1]) {
        std::cout << "WARNING: Indices not stable at index " << i
                  << " for equal keys " << sycl_keys[i] << std::endl;
        indices_correct = false;
      }
    }
    if (indices_correct) {
      std::cout << "✓ Indices are stable for equal keys" << std::endl;
    }
  }

  // Overall test result
  bool all_tests_passed = keys_match && indices_match;
  std::cout << "\n=== Test Summary ===" << std::endl;
  if (all_tests_passed) {
    std::cout << "✓ ALL TESTS PASSED: SYCL sort matches std::sort exactly!"
              << std::endl;
  } else {
    std::cout << "✗ TESTS FAILED: SYCL sort does not match std::sort"
              << std::endl;
  }

  // Cleanup
  sycl::free(d_keysAll, q);
  sycl::free(d_indicesAll, q);
  sycl::free(d_element_offsets, q);
  sycl::free(d_element_counts, q);

  return all_tests_passed ? 0 : 1;
}
