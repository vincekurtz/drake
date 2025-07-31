#pragma once

#include <algorithm>
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

// Define this macro to enable problem size logging
#define DRAKE_PROBLEM_SIZE_LOGGING_ENABLED

#ifdef DRAKE_PROBLEM_SIZE_LOGGING_ENABLED

class ProblemSizeLogger {
 public:
  struct SizeStats {
    int64_t min = 0;
    int64_t max = 0;
    double avg = 0.0;
    int64_t total = 0;
    size_t call_count = 0;
  };

  static ProblemSizeLogger& GetInstance() {
    static ProblemSizeLogger instance;
    return instance;
  }

  // Add a count for a named event (e.g., number of candidate tets)
  void AddCount(const std::string& name, int64_t count) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    counts_[name].push_back(count);
  }

  // Increment a count for a named event (e.g., faces inserted)
  void Increment(const std::string& name, int64_t delta = 1) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t new_count = delta;
    if (!counts_[name].empty()) {
      new_count += counts_[name].back();
    }
    counts_[name].push_back(new_count);
  }

  SizeStats GetStats(const std::string& name) const {
    SizeStats stats;
    auto it = counts_.find(name);
    if (it == counts_.end()) return stats;
    const auto& vals = it->second;
    if (vals.empty()) return stats;
    stats.call_count = vals.size();
    stats.min = *std::min_element(vals.begin(), vals.end());
    stats.max = *std::max_element(vals.begin(), vals.end());
    stats.total = std::accumulate(vals.begin(), vals.end(), int64_t(0));
    // Always use call count from the HydroelasticQuery if available
    size_t call_count = stats.call_count;
    auto HydroI = counts_.find("HydroelasticQuery");
    if (HydroI != counts_.end()) {
      const auto& HydroTimes = HydroI->second;
      call_count = HydroTimes.size();
    }
    stats.avg = static_cast<double>(stats.total) / call_count;
    return stats;
  }

  void PrintStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "\n=== Problem Size Statistics ===" << std::endl;
    for (const auto& [name, _] : counts_) {
      auto stats = GetStats(name);
      std::cout << "Problem Size: " << name << std::endl;
      std::cout << "  Calls: " << stats.call_count << std::endl;
      std::cout << "  Min: " << stats.min << std::endl;
      std::cout << "  Max: " << stats.max << std::endl;
      std::cout << "  Avg: " << stats.avg << std::endl;
      std::cout << "  Total: " << stats.total << std::endl;
      std::cout << std::endl;
    }
  }

  // Print all problem size statistics in JSON format
  void PrintStatsJson(const std::string& path = "") const {
    PrintStatsJson(path, "");
  }

  // Overload: Print all problem size statistics in JSON format, with extra JSON
  // fields
  void PrintStatsJson(const std::string& path,
                      const std::string& extra_json) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"problem_sizes\": {\n";
    bool first = true;
    for (const auto& [name, _] : counts_) {
      if (!first) oss << ",\n";
      first = false;
      auto stats = GetStats(name);
      oss << "    \"" << name << "\": {"
          << "\"calls\": " << stats.call_count << ", "
          << "\"min\": " << stats.min << ", "
          << "\"max\": " << stats.max << ", "
          << "\"avg\": " << stats.avg << ", "
          << "\"total\": " << stats.total << "}";
    }
    oss << "\n  }";
    if (!extra_json.empty()) {
      oss << ",\n" << extra_json << "\n";
    } else {
      oss << "\n";
    }
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
    counts_.clear();
  }

  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool IsEnabled() const { return enabled_; }

 private:
  ProblemSizeLogger() = default;
  mutable std::mutex mutex_;
  std::map<std::string, std::vector<int64_t>> counts_;
  bool enabled_ = true;
};

#else  // DRAKE_PROBLEM_SIZE_LOGGING_ENABLED

class ProblemSizeLogger {
 public:
  struct SizeStats {
    int64_t min = 0;
    int64_t max = 0;
    double avg = 0.0;
    int64_t total = 0;
    size_t call_count = 0;
  };
  static ProblemSizeLogger& GetInstance() {
    static ProblemSizeLogger instance;
    return instance;
  }
  void AddCount(const std::string&, int64_t) {}
  void Increment(const std::string&, int64_t = 1) {}
  SizeStats GetStats(const std::string&) const { return SizeStats{}; }
  void PrintStats() const {}
  void Clear() {}
  void SetEnabled(bool) {}
  bool IsEnabled() const { return false; }
};

#endif  // DRAKE_PROBLEM_SIZE_LOGGING_ENABLED

}  // namespace common
}  // namespace drake