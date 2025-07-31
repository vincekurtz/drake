#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Define this macro to enable timing logging
#define DRAKE_SYCL_TIMING_ENABLED

#ifdef DRAKE_SYCL_TIMING_ENABLED

// Timing logger class for SYCL kernels
class SyclTimingLogger {
 public:
  // Constructor
  SyclTimingLogger() = default;

  // Start timing a kernel
  void StartKernel(const std::string& kernel_name) {
    auto start_time = std::chrono::high_resolution_clock::now();
    kernel_start_times_[kernel_name] = start_time;
  }

  // End timing a kernel and record the duration
  void EndKernel(const std::string& kernel_name) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_time = kernel_start_times_[kernel_name];
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    kernel_timings_[kernel_name].push_back(duration.count());
  }

  // Start timing a SYCL event
  void StartEvent(const std::string& event_name) {
    auto start_time = std::chrono::high_resolution_clock::now();
    event_start_times_[event_name] = start_time;
  }

  // End timing a SYCL event and record the duration
  void EndEvent(const std::string& event_name) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_time = event_start_times_[event_name];
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    event_timings_[event_name].push_back(duration.count());
  }

  // Time a SYCL event using SYCL's built-in timing
  void TimeSyclEvent(const std::string& event_name, sycl::event& event) {
    // Wait for the event to complete
    event.wait_and_throw();

    // Get the timing using SYCL's profiling info
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();

    // Convert from nanoseconds to microseconds
    auto duration_us = (end_time - start_time) / 1000.0;
    sycl_event_timings_[event_name].push_back(static_cast<double>(duration_us));
  }

  // Get timing statistics for a kernel
  struct TimingStats {
    double min_time_us = 0.0;
    double max_time_us = 0.0;
    double avg_time_us = 0.0;
    double total_time_us = 0.0;
    size_t call_count = 0;
  };

  TimingStats GetKernelStats(const std::string& kernel_name) const {
    TimingStats stats;
    auto it = kernel_timings_.find(kernel_name);
    if (it == kernel_timings_.end()) return stats;

    const auto& timings = it->second;
    if (timings.empty()) return stats;

    stats.call_count = timings.size();
    stats.min_time_us = *std::min_element(timings.begin(), timings.end());
    stats.max_time_us = *std::max_element(timings.begin(), timings.end());
    stats.total_time_us = std::accumulate(timings.begin(), timings.end(), 0.0);
    stats.avg_time_us = stats.total_time_us / stats.call_count;

    return stats;
  }

  TimingStats GetEventStats(const std::string& event_name) const {
    TimingStats stats;
    auto it = event_timings_.find(event_name);
    if (it == event_timings_.end()) return stats;

    const auto& timings = it->second;
    if (timings.empty()) return stats;

    stats.call_count = timings.size();
    stats.min_time_us = *std::min_element(timings.begin(), timings.end());
    stats.max_time_us = *std::max_element(timings.begin(), timings.end());
    stats.total_time_us = std::accumulate(timings.begin(), timings.end(), 0.0);
    stats.avg_time_us = stats.total_time_us / stats.call_count;

    return stats;
  }

  TimingStats GetSyclEventStats(const std::string& event_name) const {
    TimingStats stats;
    auto it = sycl_event_timings_.find(event_name);
    if (it == sycl_event_timings_.end()) return stats;

    const auto& timings = it->second;
    if (timings.empty()) return stats;

    stats.call_count = timings.size();
    stats.min_time_us = *std::min_element(timings.begin(), timings.end());
    stats.max_time_us = *std::max_element(timings.begin(), timings.end());
    stats.total_time_us = std::accumulate(timings.begin(), timings.end(), 0.0);
    stats.avg_time_us = stats.total_time_us / stats.call_count;

    return stats;
  }

  // Print all timing statistics
  void PrintStats() const {
    std::cout << "\n=== SYCL Kernel Timing Statistics ===" << std::endl;

    // Print kernel timings
    for (const auto& [kernel_name, _] : kernel_timings_) {
      auto stats = GetKernelStats(kernel_name);
      std::cout << "Kernel: " << kernel_name << std::endl;
      std::cout << "  Calls: " << stats.call_count << std::endl;
      std::cout << "  Min: " << stats.min_time_us << " μs" << std::endl;
      std::cout << "  Max: " << stats.max_time_us << " μs" << std::endl;
      std::cout << "  Avg: " << stats.avg_time_us << " μs" << std::endl;
      std::cout << "  Total: " << stats.total_time_us << " μs" << std::endl;
      std::cout << std::endl;
    }

    // Print event timings
    for (const auto& [event_name, _] : event_timings_) {
      auto stats = GetEventStats(event_name);
      std::cout << "Event: " << event_name << std::endl;
      std::cout << "  Calls: " << stats.call_count << std::endl;
      std::cout << "  Min: " << stats.min_time_us << " μs" << std::endl;
      std::cout << "  Max: " << stats.max_time_us << " μs" << std::endl;
      std::cout << "  Avg: " << stats.avg_time_us << " μs" << std::endl;
      std::cout << "  Total: " << stats.total_time_us << " μs" << std::endl;
      std::cout << std::endl;
    }

    // Print SYCL event timings
    for (const auto& [event_name, _] : sycl_event_timings_) {
      auto stats = GetSyclEventStats(event_name);
      std::cout << "SYCL Event: " << event_name << std::endl;
      std::cout << "  Calls: " << stats.call_count << std::endl;
      std::cout << "  Min: " << stats.min_time_us << " μs" << std::endl;
      std::cout << "  Max: " << stats.max_time_us << " μs" << std::endl;
      std::cout << "  Avg: " << stats.avg_time_us << " μs" << std::endl;
      std::cout << "  Total: " << stats.total_time_us << " μs" << std::endl;
      std::cout << std::endl;
    }
  }

  // Print all timing statistics in JSON format
  void PrintStatsJson(const std::string& path = "") const {
    std::ostringstream oss;
    oss << "{\n";
    // Kernel timings
    oss << "  \"kernel_timings\": {\n";
    bool first = true;
    for (const auto& [kernel_name, timings] : kernel_timings_) {
      if (!first) oss << ",\n";
      first = false;
      auto stats = GetKernelStats(kernel_name);
      oss << "    \"" << kernel_name << "\": {"
          << "\"calls\": " << stats.call_count << ", "
          << "\"min_us\": " << stats.min_time_us << ", "
          << "\"max_us\": " << stats.max_time_us << ", "
          << "\"avg_us\": " << stats.avg_time_us << ", "
          << "\"total_us\": " << stats.total_time_us << "}";
      
      // Save individual kernel timings to separate txt file
      if (!path.empty()) {
        std::string txt_path = path;
        // Remove .json extension if present and add kernel name
        if (txt_path.length() > 5 && txt_path.substr(txt_path.length() - 5) == ".json") {
          txt_path = txt_path.substr(0, txt_path.length() - 5);
        }
        txt_path += "_" + kernel_name + ".txt";
        
        std::ofstream txt_ofs(txt_path);
        if (txt_ofs.is_open()) {
          txt_ofs << "# Kernel: " << kernel_name << std::endl;
          txt_ofs << "# Format: time_step timing_us" << std::endl;
          for (size_t i = 0; i < timings.size(); ++i) {
            txt_ofs << i << " " << timings[i] << std::endl;
          }
          txt_ofs.close();
        } else {
          std::cerr << "Failed to open file for writing: " << txt_path << std::endl;
        }
      }
    }
    oss << "\n  },\n";
    // Event timings
    oss << "  \"event_timings\": {\n";
    first = true;
    for (const auto& [event_name, _] : event_timings_) {
      if (!first) oss << ",\n";
      first = false;
      auto stats = GetEventStats(event_name);
      oss << "    \"" << event_name << "\": {"
          << "\"calls\": " << stats.call_count << ", "
          << "\"min_us\": " << stats.min_time_us << ", "
          << "\"max_us\": " << stats.max_time_us << ", "
          << "\"avg_us\": " << stats.avg_time_us << ", "
          << "\"total_us\": " << stats.total_time_us << "}";
    }
    oss << "\n  },\n";
    // SYCL event timings
    oss << "  \"sycl_event_timings\": {\n";
    first = true;
    for (const auto& [event_name, _] : sycl_event_timings_) {
      if (!first) oss << ",\n";
      first = false;
      auto stats = GetSyclEventStats(event_name);
      oss << "    \"" << event_name << "\": {"
          << "\"calls\": " << stats.call_count << ", "
          << "\"min_us\": " << stats.min_time_us << ", "
          << "\"max_us\": " << stats.max_time_us << ", "
          << "\"avg_us\": " << stats.avg_time_us << ", "
          << "\"total_us\": " << stats.total_time_us << "}";
    }
    oss << "\n  }\n";
    oss << "}";

    if (!path.empty()) {
      std::ofstream ofs(path);
      if (ofs.is_open()) {
        ofs << oss.str();
        ofs.close();
      } else {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
      }
    } else {
      std::cout << oss.str() << std::endl;
    }
  }

  // Clear all timing data
  void Clear() {
    kernel_timings_.clear();
    event_timings_.clear();
    sycl_event_timings_.clear();
    kernel_start_times_.clear();
    event_start_times_.clear();
  }

  // Enable/disable timing
  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool IsEnabled() const { return enabled_; }

 private:
  bool enabled_ = true;
  std::map<std::string, std::chrono::high_resolution_clock::time_point>
      kernel_start_times_;
  std::map<std::string, std::chrono::high_resolution_clock::time_point>
      event_start_times_;
  std::map<std::string, std::vector<double>> kernel_timings_;
  std::map<std::string, std::vector<double>> event_timings_;
  std::map<std::string, std::vector<double>> sycl_event_timings_;
};

#else

// Dummy class when timing is disabled
class SyclTimingLogger {
 public:
  void StartKernel(const std::string&) {}
  void EndKernel(const std::string&) {}
  void StartEvent(const std::string&) {}
  void EndEvent(const std::string&) {}
  void TimeSyclEvent(const std::string&, sycl::event&) {}

  struct TimingStats {
    double min_time_us = 0.0;
    double max_time_us = 0.0;
    double avg_time_us = 0.0;
    double total_time_us = 0.0;
    size_t call_count = 0;
  };

  TimingStats GetKernelStats(const std::string&) const { return TimingStats{}; }
  TimingStats GetEventStats(const std::string&) const { return TimingStats{}; }
  TimingStats GetSyclEventStats(const std::string&) const {
    return TimingStats{};
  }
  void PrintStats() const {}
  void Clear() {}
  void SetEnabled(bool) {}
  bool IsEnabled() const { return false; }
};

#endif  // DRAKE_SYCL_TIMING_ENABLED

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake