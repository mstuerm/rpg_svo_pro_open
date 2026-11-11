// Keep assertions active in Release catkin builds.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <svo_ros/policy_health.h>
#include <cassert>
#include <cmath>
#include <limits>
int main() {
  svo::PolicyHealthAccumulator health;
  auto first = health.update(3, {1, 2}, {1., 3., std::numeric_limits<double>::quiet_NaN()});
  assert(first.count == 3 && std::isnan(first.survival));
  assert(first.reprojection == 2. && first.age == 1.);
  assert(first.normalized[0] == .03 && first.normalized[1] == 0.);
  assert(first.normalized[2] == 1. && first.normalized[3] == .1);
  auto next = health.update(4, {2, 3}, {0., 1., 8.});
  assert(next.survival == .5 && next.age == 1.5 && next.reprojection == 1.);
  auto empty = health.update(0, {}, {});
  assert(empty.survival == 0. && std::isnan(empty.reprojection));
  assert(empty.normalized[2] == 1.5 && empty.age == 0.);
  auto new_tracks = health.update(200, {2}, {10.});
  assert(std::isnan(new_tracks.survival) && new_tracks.age == 1.);
  assert(new_tracks.normalized[0] == 1.5 && new_tracks.normalized[2] == 1.5);
  for (int i = 0; i < 20; ++i) next = health.update(1, {2}, {0.});
  assert(next.normalized[3] == 1.5 && next.survival == 1.);
  health.reset();
  auto reset = health.update(1, {2}, {});
  assert(std::isnan(reset.survival) && reset.age == 1.);
}
