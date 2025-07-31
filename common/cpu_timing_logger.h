#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace drake {
namespace common {

// Define this macro to enable CPU timing logging
#define DRAKE_CPU_TIMING_ENABLED

#ifdef DRAKE_CPU_TIMING_ENABLED

class CpuTimingLogger {
 public:
  struct TimingStats {
    double min_time_us = 0.0;
    double max_time_us = 0.0;
    double avg_time_us = 0.0;
    double total_time_us = 0.0;
    size_t call_count = 0;
  };

  static CpuTimingLogger& GetInstance() {
    static CpuTimingLogger instance;
    return instance;
  }

  void AddTiming(const std::string& name, double microseconds) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (first_run_seen_.find(name) == first_run_seen_.end()) {
      first_run_seen_.insert(name);
      return;  // Ignore the first run.
    }
    timings_[name].push_back(microseconds);
  }

  TimingStats GetStats(const std::string& name) const {
    TimingStats stats;
    auto it = timings_.find(name);
    if (it == timings_.end()) return stats;
    const auto& times = it->second;
    if (times.empty()) return stats;
    stats.call_count = times.size();
    stats.min_time_us = *std::min_element(times.begin(), times.end());
    stats.max_time_us = *std::max_element(times.begin(), times.end());
    stats.total_time_us = std::accumulate(times.begin(), times.end(), 0.0);
    // Always use call count from the HydroelasticQuery if available
    size_t call_count = stats.call_count;
    auto HydroI = timings_.find("HydroelasticQuery");
    if (HydroI != timings_.end()) {
      const auto& HydroTimes = HydroI->second;
      call_count = HydroTimes.size();
    }
    stats.avg_time_us = stats.total_time_us / call_count;
    return stats;
  }

  void PrintStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "\n=== CPU Timing Statistics ===" << std::endl;
    for (const auto& [name, _] : timings_) {
      auto stats = GetStats(name);
      std::cout << "Timer: " << name << std::endl;
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
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"timings\": {\n";
    bool first = true;
    for (const auto& [name, _] : timings_) {
      if (!first) oss << ",\n";
      first = false;
      auto stats = GetStats(name);
      oss << "    \"" << name << "\": {"
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

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.clear();
  }

  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool IsEnabled() const { return enabled_; }

 private:
  CpuTimingLogger() = default;
  mutable std::mutex mutex_;
  std::map<std::string, std::vector<double>> timings_;
  bool enabled_ = true;
  std::set<std::string> first_run_seen_;
};

class ScopedCpuTimer {
 public:
  explicit ScopedCpuTimer(const std::string& name)
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
  ~ScopedCpuTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    double duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
            .count();
    CpuTimingLogger::GetInstance().AddTiming(name_, duration_us);
  }

 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};

#else  // DRAKE_CPU_TIMING_ENABLED

class CpuTimingLogger {
 public:
  struct TimingStats {
    double min_time_us = 0.0;
    double max_time_us = 0.0;
    double avg_time_us = 0.0;
    double total_time_us = 0.0;
    size_t call_count = 0;
  };
  static CpuTimingLogger& GetInstance() {
    static CpuTimingLogger instance;
    return instance;
  }
  void AddTiming(const std::string&, double) {}
  TimingStats GetStats(const std::string&) const { return TimingStats{}; }
  void PrintStats() const {}
  void Clear() {}
  void SetEnabled(bool) {}
  bool IsEnabled() const { return false; }
};

class ScopedCpuTimer {
 public:
  explicit ScopedCpuTimer(const std::string&) {}
};

#endif  // DRAKE_CPU_TIMING_ENABLED

// Macro for easy use
#define DRAKE_CPU_SCOPED_TIMER(name) \
  drake::common::ScopedCpuTimer timer_##__LINE__(name)

}  // namespace common
}  // namespace drake