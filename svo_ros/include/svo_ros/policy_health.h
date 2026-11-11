#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace svo {
// Stateful reduction shared by every ROS frontend frame, even without subscribers.
// Input extraction deliberately mirrors svo_pybind::latest_features camera 0.
struct PolicyHealthValues {
  std::size_t count = 0;
  double survival = std::numeric_limits<double>::quiet_NaN();
  double reprojection = std::numeric_limits<double>::quiet_NaN();
  double age = 0.;
  std::array<double, 4> normalized{{0., 0., 1.5, 0.}};
};
class PolicyHealthAccumulator {
 public:
  void reset() { ages_.clear(); }
  PolicyHealthValues update(std::size_t count,
                           const std::unordered_set<int64_t>& ids,
                           std::vector<double> errors) {
    PolicyHealthValues out;
    out.count = count;
    std::unordered_map<int64_t, uint64_t> next;
    std::vector<double> ages;
    std::size_t surviving = 0;
    for (const auto id : ids) {
      const auto old = ages_.find(id);
      const uint64_t age = old == ages_.end() ? 1 : old->second + 1;
      surviving += old != ages_.end();
      next[id] = age;
      ages.push_back(static_cast<double>(age));
    }
    if (!ages_.empty()) out.survival = double(surviving) / ages_.size();
    ages_.swap(next);
    errors.erase(std::remove_if(errors.begin(), errors.end(),
                               [](double x) { return !std::isfinite(x); }), errors.end());
    if (!errors.empty()) out.reprojection = median(errors);
    if (!ages.empty()) out.age = median(ages);
    out.normalized = {{clip(double(count) / 100., 1.5),
      std::isfinite(out.survival) ? clip(out.survival, 1.) : 0.,
      std::isfinite(out.reprojection) ? clip(out.reprojection / 2., 1.5) : 1.5,
      clip(out.age / 10., 1.5)}};
    return out;
  }
 private:
  static double clip(double x, double high) { return std::max(0., std::min(high, x)); }
  static double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.;
  }
  std::unordered_map<int64_t, uint64_t> ages_;
};
}  // namespace svo
