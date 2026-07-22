// A from-scratch reference implementation of classic FABRIK (Forward And
// Backward Reaching Inverse Kinematics; Aristidou & Lazarus, 2011) for a
// single open kinematic chain with a fixed base.
//
// This is deliberately the simplest, most classical form of FABRIK:
//   - The chain is a sequence of points connected by segments of fixed
//     length -- no rotation-axis constraints, no joint angle limits. Every
//     joint is a free ball joint that can bend in any direction, as long as
//     the segment lengths on either side of it are preserved.
//   - Only the tip (last point) is given a target position. Intermediate
//     points have no target of their own; classic FABRIK has no mechanism
//     to fit them (this is inherent to the algorithm, not a shortcut taken
//     here -- see ../README.md).
//
// This means a chain built from the NeuroMechFly body plan's per-joint
// offset lengths has strictly more freedom than the real leg mechanism
// (whose joints are constrained to rotate about specific local axes), so
// solved configurations may be physically impossible for the actual leg.
// See extern/fabrik/README.md for the full discussion of this tradeoff.

#pragma once

#include <cmath>
#include <vector>

namespace fabrik {

struct Vec3 {
  double x = 0, y = 0, z = 0;

  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  double norm() const { return std::sqrt(x * x + y * y + z * z); }
};

inline Vec3 lerp(const Vec3 &a, const Vec3 &b, double t) { return a + (b - a) * t; }

// A single open chain: `lengths[i]` is the fixed distance between
// `points[i]` and `points[i+1]`, so a chain with N segments has N+1 points.
// `points[0]` is the fixed base (e.g. the thorax attachment point) and is
// never moved by `solve()`.
struct FabrikChain {
  std::vector<double> lengths;
  double total_reach = 0.0;

  int n_points() const { return static_cast<int>(lengths.size()) + 1; }

  // Solves in place, starting from whatever configuration `points` already
  // holds (the caller controls cold- vs. warm-starting by what it puts
  // there beforehand). Returns the number of iterations actually run (0 if
  // the target was unreachable and the chain was simply stretched toward
  // it, since that case has no meaningful "iteration" to converge).
  int solve(std::vector<Vec3> &points, const Vec3 &target, int max_iterations, double tolerance) const {
    const int n = static_cast<int>(lengths.size());
    const Vec3 base = points[0];

    // Unreachable target: classic FABRIK's documented fallback is to fully
    // extend the chain in a straight line toward the target rather than
    // iterate (iterating a chain that can never reach won't converge).
    if ((target - base).norm() > total_reach) {
      for (int i = 0; i < n; i++) {
        double r = (target - points[i]).norm();
        double lambda = (r > 1e-12) ? lengths[i] / r : 0.0;
        points[i + 1] = lerp(points[i], target, lambda);
      }
      return 0;
    }

    for (int iter = 0; iter < max_iterations; iter++) {
      if ((points[n] - target).norm() < tolerance) return iter;

      // Forward reaching: snap the tip onto the target, then walk back
      // toward the base, keeping every segment's length fixed.
      points[n] = target;
      for (int i = n - 1; i >= 0; i--) {
        double r = (points[i] - points[i + 1]).norm();
        double lambda = (r > 1e-12) ? lengths[i] / r : 0.0;
        points[i] = lerp(points[i + 1], points[i], lambda);
      }

      // Backward reaching: snap the base back to its fixed position, then
      // walk forward toward the tip, again keeping lengths fixed.
      points[0] = base;
      for (int i = 0; i < n; i++) {
        double r = (points[i + 1] - points[i]).norm();
        double lambda = (r > 1e-12) ? lengths[i] / r : 0.0;
        points[i + 1] = lerp(points[i], points[i + 1], lambda);
      }
    }
    return max_iterations;
  }
};

}  // namespace fabrik
