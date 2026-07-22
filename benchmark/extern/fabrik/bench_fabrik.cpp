// Throughput/latency benchmark for the from-scratch reference FABRIK solver
// (fabrik.hpp), mirroring ../../fastik_cpp/bench_cpp.cpp's methodology
// (single-frame latency, single-thread sequence throughput, multi-thread
// sequence throughput) and ../../plot/RESULTS_SCHEMA.md's output format, so
// results are directly comparable to fastik's own benchmarks.
//
// Formulation (see README.md for the full rationale):
//   - Thorax is a FIXED base at the origin -- no floating-base solving.
//   - Each of the 6 legs is one independent single-chain FABRIK problem,
//     targeting only its claw (tip) position (classic FABRIK has no notion
//     of fitting intermediate keypoints).
//   - Every joint is an unconstrained free ball joint (no rotation-axis
//     limits from the body plan are enforced) -- see fabrik.hpp's header
//     comment and README.md for why this makes accuracy non-comparable to
//     axis-constrained solvers; only raw speed is a fair comparison here.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "fabrik.hpp"
#include "json.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using fabrik::FabrikChain;
using fabrik::Vec3;

// FABRIK iteration cap and early-stop tolerance (model units), matching the
// "early stop, n_iterations as ceiling" spirit of fastik's own default
// solver config.
constexpr int kMaxIterations = 15;
constexpr double kTolerance = 1e-4;

// One independent single-chain FABRIK problem per leg: thorax (fixed base,
// origin) -> thorax_coxa -> coxa_trochanterfemur -> trochanterfemur_tibia ->
// tibia_tarsus -> claw (tip). Segment lengths come from the norm of each
// joint's `offset_pos` in the body-plan JSON.
struct LegChain {
  std::string name;  // leg prefix, e.g. "lf"
  FabrikChain chain;
  std::vector<Vec3> neutral_points;  // n_points() entries; [0] = thorax
};

double elapsed_us(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

Vec3 to_vec3(const Json &p) { return {p[0].as_number(), p[1].as_number(), p[2].as_number()}; }

// Builds the 6 leg chains from the body plan, grouped via fixtures.json's
// `leg_joint_names` (30 names = 6 legs x 5 joints each, in a fixed
// thorax_coxa/coxa_trochanterfemur/trochanterfemur_tibia/tibia_tarsus/claw
// order per leg). Initial ("neutral") positions are the cumulative sum of
// each joint's `offset_pos`, assuming zero rotation -- a valid starting
// configuration for FABRIK (which has no angle concept at all, only point
// positions), not a claim about the mechanism's real neutral pose.
std::vector<LegChain> build_leg_chains(const Json &body_plan, const Json &fixtures) {
  auto find_offset = [&](const std::string &joint_name) -> Vec3 {
    for (auto &j : body_plan["joints"].as_array()) {
      if (j["name"].as_string() == joint_name) return to_vec3(j["offset_pos"]);
    }
    throw std::runtime_error("joint not found in body plan: " + joint_name);
  };

  auto &names = fixtures["leg_joint_names"].as_array();
  if (names.size() % 5 != 0) throw std::runtime_error("leg_joint_names not a multiple of 5");

  std::vector<LegChain> legs;
  for (size_t leg = 0; leg * 5 < names.size(); leg++) {
    LegChain lc;
    const std::string &first_name = names[leg * 5].as_string();
    lc.name = first_name.substr(0, first_name.find('_'));
    lc.neutral_points.push_back({0, 0, 0});  // thorax: fixed base, always at the origin
    for (size_t k = 0; k < 5; k++) {
      Vec3 offset = find_offset(names[leg * 5 + k].as_string());
      double len = offset.norm();
      lc.chain.lengths.push_back(len);
      lc.chain.total_reach += len;
      lc.neutral_points.push_back(lc.neutral_points.back() + offset);
    }
    legs.push_back(std::move(lc));
  }
  return legs;
}

// Extracts each leg's claw (tip) target from a `target_ego` array (30
// entries = 6 legs x 5 joints, `leg_joint_names` order); the claw is the
// last of each leg's group of 5 -- every intermediate joint target is
// ignored, since classic FABRIK only ever fits the chain's end effector.
std::vector<Vec3> claw_targets(const Json &target_ego) {
  auto &arr = target_ego.as_array();
  std::vector<Vec3> targets;
  for (size_t leg = 0; leg * 5 < arr.size(); leg++) targets.push_back(to_vec3(arr[leg * 5 + 4]));
  return targets;
}

// One "frame" of work: solve all 6 legs' chains in place, each starting
// from whatever `points[leg]` already holds (the caller controls cold vs.
// warm start).
void solve_frame(const std::vector<LegChain> &legs, std::vector<std::vector<Vec3>> &points,
                  const std::vector<Vec3> &targets) {
  for (size_t leg = 0; leg < legs.size(); leg++) {
    legs[leg].chain.solve(points[leg], targets[leg], kMaxIterations, kTolerance);
  }
}

std::vector<std::vector<Vec3>> neutral_state(const std::vector<LegChain> &legs) {
  std::vector<std::vector<Vec3>> points;
  points.reserve(legs.size());
  for (auto &lc : legs) points.push_back(lc.neutral_points);
  return points;
}

// Prints the usual latency/throughput summary and returns the mean in
// microseconds, mirroring bench_cpp.cpp's `summarize`.
double summarize(const std::string &label, std::vector<double> samples_us) {
  std::sort(samples_us.begin(), samples_us.end());
  size_t n = samples_us.size();
  double mean = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / n;
  auto pct = [&](double p) { return samples_us[static_cast<size_t>(std::round((n - 1) * p))]; };
  std::printf(
      "%-42s n=%-7zu mean=%9.3fus  median=%9.3fus  p95=%9.3fus  p99=%9.3fus  min=%9.3fus  "
      "max=%9.3fus  throughput=%10.1f calls/s\n",
      label.c_str(), n, mean, pct(0.50), pct(0.95), pct(0.99), samples_us.front(), samples_us.back(), 1e6 / mean);
  return mean;
}

// =============================================================================
//  1. single_frame_latency_us
// =============================================================================
// A fresh/neutral chain state solved against a fixed target every call (no
// warm start) -- one frame's cost is all 6 legs' independent chain solves.
double bench_single_frame_latency(const std::vector<LegChain> &legs, const std::vector<Vec3> &target, int n_calls) {
  for (int i = 0; i < 1000; i++) {
    auto points = neutral_state(legs);
    solve_frame(legs, points, target);
  }
  std::vector<double> samples;
  samples.reserve(n_calls);
  for (int i = 0; i < n_calls; i++) {
    auto points = neutral_state(legs);
    auto t0 = Clock::now();
    solve_frame(legs, points, target);
    samples.push_back(elapsed_us(t0));
  }
  return summarize("fabrik::solve (6 legs, cold)", samples);
}

// =============================================================================
//  2. single_thread_throughput_fps
// =============================================================================
// Warm-started sequence solve: each leg's chain is seeded from that same
// leg's previous frame's converged points. A fresh (neutral) state is used
// for the timed pass after a full untimed warmup pass, matching perf.rs /
// bench_cpp.cpp's "second sequence solver for the timed run" pattern.
std::vector<double> bench_single_thread_sequence(const std::vector<LegChain> &legs,
                                                  const std::vector<std::vector<Vec3>> &all_targets) {
  auto warmup_points = neutral_state(legs);
  for (auto &targets : all_targets) solve_frame(legs, warmup_points, targets);

  auto points = neutral_state(legs);
  std::vector<double> samples;
  samples.reserve(all_targets.size());
  for (auto &targets : all_targets) {
    auto t0 = Clock::now();
    solve_frame(legs, points, targets);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// =============================================================================
//  3. multi_thread_throughput_fps
// =============================================================================
constexpr int kMultithreadNThreads = 8;
// A longer tiled sequence (repeating the 300-frame native-rate fixture) so
// each of the 8 threads gets a nontrivial, evenly-sized contiguous chunk;
// 2400 divides evenly into 8 chunks of 300 frames each.
constexpr int kMultithreadTotalFrames = 2400;

double bench_multithread_sequence(const std::vector<LegChain> &legs,
                                   const std::vector<std::vector<Vec3>> &native_targets) {
  const int chunk_len = kMultithreadTotalFrames / kMultithreadNThreads;

  // Each thread solves its own contiguous chunk independently: warm-started
  // within the chunk (frame-to-frame), but cold (neutral pose) at the
  // chunk's start -- there is no cross-chunk state sharing, matching a
  // segmented-parallel solve with no overlap/blending (FABRIK's per-leg
  // chains need none, unlike fastik's segmented solver).
  auto worker = [&](int start_frame, int count) {
    auto points = neutral_state(legs);
    for (int i = 0; i < count; i++) {
      const auto &targets = native_targets[(start_frame + i) % native_targets.size()];
      solve_frame(legs, points, targets);
    }
  };

  auto run_once = [&] {
    std::vector<std::thread> threads;
    threads.reserve(kMultithreadNThreads);
    for (int t = 0; t < kMultithreadNThreads; t++) threads.emplace_back(worker, t * chunk_len, chunk_len);
    for (auto &th : threads) th.join();
  };

  run_once();  // warm up
  auto t0 = Clock::now();
  run_once();
  return elapsed_us(t0) / 1e3;  // ms
}

// Writes ../../plot/results/fabrik.json, matching ../../plot/RESULTS_SCHEMA.md.
void write_results_json(double single_frame_latency_us, double single_thread_throughput_fps,
                         double multi_thread_throughput_fps) {
  std::filesystem::path out_dir = std::filesystem::path(__FILE__).parent_path() / "../../plot/results";
  std::filesystem::create_directories(out_dir);
  std::ofstream out(out_dir / "fabrik.json");
  const char *notes =
      "Fixed-base, per-leg, tip-only formulation: thorax is a fixed base at "
      "the origin (no floating-base solve); each of the 6 legs is solved as "
      "an independent single open-chain FABRIK problem targeting only its "
      "claw (tip) position, ignoring intermediate joint keypoints -- same "
      "asymmetry as the TRAC-IK benchmark, since classic FABRIK (like "
      "TRAC-IK) has no notion of a floating base, branching tree, or "
      "intermediate-keypoint fitting. This is UNCONSTRAINED positional "
      "FABRIK (Aristidou and Lazarus 2011): every joint is a free ball "
      "joint with only its segment length fixed; no rotation-axis "
      "constraints or joint limits from the body plan are enforced, so "
      "this solver has strictly more freedom than fastik/KDL/TRAC-IK/"
      "Pinocchio's axis-constrained joints. Its keypoint-fit quality is "
      "therefore not apples-to-apples with those -- only its raw solve "
      "speed is a fair comparison point.";
  out << "{\n"
      << "  \"name\": \"fabrik\",\n"
      << "  \"language\": \"cpp\",\n"
      << "  \"formulation\": \"fixed-base-per-leg\",\n"
      << "  \"single_frame_latency_us\": " << single_frame_latency_us << ",\n"
      << "  \"single_thread_throughput_fps\": " << single_thread_throughput_fps << ",\n"
      << "  \"multi_thread_throughput_fps\": " << multi_thread_throughput_fps << ",\n"
      << "  \"multi_thread_n_threads\": " << kMultithreadNThreads << ",\n"
      << "  \"notes\": \"" << notes << "\"\n"
      << "}\n";
}

}  // namespace

int main() {
#ifndef FASTIK_ASSETS_DIR
#error "FASTIK_ASSETS_DIR must be defined at compile time"
#endif
  const std::string assets_dir = FASTIK_ASSETS_DIR;

  Json body_plan = parse_json_file(assets_dir + "/neuromechfly_ypr_legs.json");
  Json fixtures = parse_json_file(assets_dir + "/fixtures.json");
  auto legs = build_leg_chains(body_plan, fixtures);

  std::printf("Reference FABRIK benchmark: %zu independent leg chains (fixed base, tip-only)\n\n", legs.size());
  for (auto &lc : legs) {
    std::printf("  %-3s: %zu segments, total_reach=%.4f\n", lc.name.c_str(), lc.chain.lengths.size(),
                lc.chain.total_reach);
  }

  auto target = claw_targets(fixtures["synthetic_frames"][0]["target_ego"]);
  std::printf("\n-- single-frame time (latency), max_iterations=%d tolerance=%g --\n", kMaxIterations, kTolerance);
  double single_frame_latency_us = bench_single_frame_latency(legs, target, 20000);

  std::vector<std::vector<Vec3>> native_targets;
  for (auto &f : fixtures["native_rate_frames"].as_array()) native_targets.push_back(claw_targets(f["target_ego"]));
  std::printf("\n-- single-thread sequence throughput (native-rate frames, warm-started) --\n");
  double single_thread_mean_us = summarize("solve_frame (6 legs, warm)", bench_single_thread_sequence(legs, native_targets));

  std::printf("\n-- multi-thread sequence throughput (%d threads, cold per chunk, warm within chunk) --\n",
              kMultithreadNThreads);
  double elapsed_ms = bench_multithread_sequence(legs, native_targets);
  double multithread_fps = kMultithreadTotalFrames / (elapsed_ms / 1e3);
  std::printf("solve_frame (parallel chunks)            n_frames=%-6d elapsed=%9.3fms  throughput=%10.1f frames/s\n",
              kMultithreadTotalFrames, elapsed_ms, multithread_fps);

  write_results_json(single_frame_latency_us, 1e6 / single_thread_mean_us, multithread_fps);
  return 0;
}
