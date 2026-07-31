// Correctness cross-check and throughput/latency benchmark for quickik's C++
// bindings, mirroring ../quickik_rust/src/{correctness,perf}.rs and
// ../quickik_python/bench.py so all three are directly comparable. See that
// Python script's own header comment for why an independent FK replica is
// used here too (FK isn't exposed to C++, same as Python).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "quickik.h"
#include "forward_kinematics.hpp"
#include "json.hpp"

namespace {

using Clock = std::chrono::steady_clock;

// Defaults matching the old `SolverConfig::default()`.
constexpr size_t kNIterations = 10;
constexpr float kNeutralWeight = 1e-3f;
constexpr float kPositionTolerance = 1e-3f;
constexpr float kAngleTolerance = 1e-3f;
constexpr float kDamping = 1e-6f;

// One body to benchmark: a name (used for the results filename and printed
// headers) plus the paths to its body plan and fixtures.
struct BodySpec {
  std::string name;
  std::string body_plan_path;
  std::string fixtures_path;
};

std::vector<Vec3> to_vec3s(const Json &target_ego) {
  std::vector<Vec3> out;
  for (auto &p : target_ego.as_array()) {
    out.push_back({static_cast<float>(p[0].as_number()), static_cast<float>(p[1].as_number()),
                    static_cast<float>(p[2].as_number())});
  }
  return out;
}

/// `target_ego` covers every joint but the free-floating root (which has no
/// mocap keypoint of its own): prepend `Missing` for it.
std::vector<quickik::KeypointObservation> build_observations(const std::vector<Vec3> &target_ego) {
  std::vector<quickik::KeypointObservation> obs;
  obs.push_back(quickik::keypoint_missing());
  for (auto &p : target_ego) {
    obs.push_back(quickik::keypoint_position_3d({p.x, p.y, p.z}, 1.0f));
  }
  return obs;
}

rust::Slice<const quickik::KeypointObservation> slice_of(const std::vector<quickik::KeypointObservation> &v) {
  return rust::Slice<const quickik::KeypointObservation>(v.data(), v.size());
}

std::vector<float> to_std_vec(const rust::Vec<float> &v) { return std::vector<float>(v.begin(), v.end()); }

std::vector<Vec3> solved_keypoints(const BodyPlan &plan, const quickik::SolverResult &result) {
  auto &p = result.root_pos;
  auto &q = result.root_rot;
  return forward_kinematics(plan, to_std_vec(result.dof_angles), Vec3{p[0], p[1], p[2]}, Quat{q[0], q[1], q[2], q[3]});
}

// a has one entry per joint (root included); b covers only leg joints (root
// excluded), so compare a[1..] against b.
std::pair<float, float> residual_stats(const std::vector<Vec3> &a, const std::vector<Vec3> &b) {
  std::vector<float> dists;
  for (size_t i = 0; i < b.size(); i++) dists.push_back((a[i + 1] - b[i]).norm());
  float sum_sq = 0;
  for (float d : dists) sum_sq += d * d;
  float rms = std::sqrt(sum_sq / dists.size());
  float max = *std::max_element(dists.begin(), dists.end());
  return {rms, max};
}

float angle_error_deg(const std::vector<float> &solved, const std::vector<float> &ground_truth) {
  float worst = 0.0f;
  constexpr float pi = 3.14159265358979323846f;
  for (size_t i = 0; i < solved.size(); i++) {
    float d = solved[i] - ground_truth[i];
    float wrapped = std::fmod(d + pi, 2.0f * pi);
    if (wrapped < 0) wrapped += 2.0f * pi;
    wrapped -= pi;
    worst = std::max(worst, std::abs(wrapped) * 180.0f / pi);
  }
  return worst;
}

// =============================================================================
//  Correctness (mirrors correctness.rs / bench_python.py's run_correctness)
// =============================================================================

void run_correctness(const BodyPlan &plan, const Json &fixtures, rust::Box<quickik::KinematicTree> &tree) {
  std::printf("== Synthetic exact-fit frames (bug hunt) ==\n");
  std::printf("%6s %16s %16s %18s %20s\n", "frame", "kpt rms", "kpt max", "angle err deg", "angle err deg (w=0)");

  auto default_solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                             kPositionTolerance, kAngleTolerance, kDamping);
  auto zero_reg_solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f, kPositionTolerance,
                                              kAngleTolerance, kDamping);

  size_t i = 0;
  for (auto &frame : fixtures["synthetic_frames"].as_array()) {
    auto target = to_vec3s(frame["target_ego"]);
    std::vector<float> ground_truth;
    for (auto &leg : frame["ground_truth_dof_angles_per_leg"].as_array()) {
      for (auto &a : leg.as_array()) ground_truth.push_back(static_cast<float>(a.as_number()));
    }
    auto obs = build_observations(target);

    auto state = quickik::state_neutral_pose(*tree);
    auto result = default_solver->solve(*state, slice_of(obs), false, false);
    auto solved_pts = solved_keypoints(plan, result);
    auto [rms, max] = residual_stats(solved_pts, target);
    float angle_err = angle_error_deg(to_std_vec(result.dof_angles), ground_truth);

    auto state0 = quickik::state_neutral_pose(*tree);
    auto result0 = zero_reg_solver->solve(*state0, slice_of(obs), false, false);
    float angle_err0 = angle_error_deg(to_std_vec(result0.dof_angles), ground_truth);

    std::printf("%6zu %16.6f %16.6f %18.4f %20.6f\n", i, rms, max, angle_err, angle_err0);
    i++;
  }
  std::printf(
      "(kpt rms/max: 3D distance to target, via an independent from-JSON FK replica, model "
      "units. angle err: max abs error over all %zu DOFs, degrees, mod 2*pi. \"w=0\" = "
      "weight=0.)\n\n",
      tree->n_dofs());

  std::printf("== Real mocap frames (cross-solver vs. flygym.ik) ==\n");
  auto &real_frames = fixtures["real_frames"].as_array();
  size_t n_joints = tree->n_joints();
  std::vector<quickik::KeypointObservation> flat;
  flat.reserve(real_frames.size() * n_joints);
  for (auto &frame : real_frames) {
    auto obs = build_observations(to_vec3s(frame["target_ego"]));
    flat.insert(flat.end(), obs.begin(), obs.end());
  }
  auto seq = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                           kPositionTolerance, kAngleTolerance, kDamping);
  auto results = seq->solve(slice_of(flat), n_joints, false, false);

  std::vector<float> quickik_rms, quickik_max, cross_rms, cross_max;
  for (size_t idx = 0; idx < real_frames.size(); idx++) {
    auto &frame = real_frames[idx];
    auto target = to_vec3s(frame["target_ego"]);
    auto result = results->at(idx);
    auto solved_pts = solved_keypoints(plan, result);

    auto [rms, max] = residual_stats(solved_pts, target);
    quickik_rms.push_back(rms);
    quickik_max.push_back(max);

    // Not every body has an independent reference solver to cross-check
    // against (e.g. G1's real frames have no flygym.ik reconstruction).
    if (frame.has("flygym_ik_reconstructed_ego")) {
      auto cross_target = to_vec3s(frame["flygym_ik_reconstructed_ego"]);
      auto [crms, cmax] = residual_stats(solved_pts, cross_target);
      cross_rms.push_back(crms);
      cross_max.push_back(cmax);
    }
  }
  auto mean = [](const std::vector<float> &v) { return std::accumulate(v.begin(), v.end(), 0.0f) / v.size(); };
  auto rms_of = [](const std::vector<float> &v) {
    float s = 0;
    for (float x : v) s += x * x;
    return std::sqrt(s / v.size());
  };
  auto max_of = [](const std::vector<float> &v) { return *std::max_element(v.begin(), v.end()); };
  std::printf("over %zu frames:\n", real_frames.size());
  std::printf("  quickik fit residual to target:      rms=%.5f  mean=%.5f  max=%.5f\n", rms_of(quickik_rms),
              mean(quickik_rms), max_of(quickik_max));
  if (!cross_rms.empty()) {
    std::printf("  cross-solver agreement (vs flygym.ik): rms=%.5f  mean=%.5f  max=%.5f\n\n", rms_of(cross_rms),
                mean(cross_rms), max_of(cross_max));
  } else {
    std::printf("  cross-solver agreement (vs flygym.ik): n/a (no reference solver output for this body)\n\n");
  }
}

// =============================================================================
//  Performance (mirrors perf.rs / bench_python.py's run_performance)
// =============================================================================

// Prints the usual latency/throughput summary and returns the mean in
// microseconds, for callers that also want the number for
// ../plot/results/quickik-cpp.json.
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

double elapsed_us(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// Single-frame latency: a fresh State::neutral_pose() solved against a fixed
// real target every call (no warm start).
std::vector<double> bench_single_frame_latency(rust::Box<quickik::KinematicTree> &tree,
                                                const std::vector<quickik::KeypointObservation> &obs, int n_calls,
                                                float neutral_weight, float position_tolerance,
                                                float angle_tolerance) {
  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, neutral_weight, position_tolerance,
                                     angle_tolerance, kDamping);
  for (int i = 0; i < 500; i++) {
    auto state = quickik::state_neutral_pose(*tree);
    solver->solve(*state, slice_of(obs), false, false);
  }
  std::vector<double> samples;
  samples.reserve(n_calls);
  for (int i = 0; i < n_calls; i++) {
    auto state = quickik::state_neutral_pose(*tree);
    auto t0 = Clock::now();
    solver->solve(*state, slice_of(obs), false, false);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// Single-thread sequence throughput: `SequenceSolver::solve` warm started
// across a tiled native-rate sequence, one frame per call (the same
// frame-by-frame interface a continuous tracking pipeline would use). A
// second, fresh `SequenceSolver` is used for the timed pass after warming up
// once, so the sequence's own frame-to-frame warm-starting is what's
// measured.
std::vector<double> bench_solve_sequence(rust::Box<quickik::KinematicTree> &tree,
                                          const std::vector<std::vector<quickik::KeypointObservation>> &all_obs) {
  size_t n_joints = all_obs.front().size();
  auto seq = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                           kPositionTolerance, kAngleTolerance, kDamping);
  for (auto &obs : all_obs) seq->solve(slice_of(obs), n_joints, false, false);

  auto timed_seq = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                                 kPositionTolerance, kAngleTolerance, kDamping);
  std::vector<double> samples;
  samples.reserve(all_obs.size());
  for (auto &obs : all_obs) {
    auto t0 = Clock::now();
    timed_seq->solve(slice_of(obs), n_joints, false, false);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// Frames per segment/worker, matching perf.rs exactly (same stride, so a
// `n_segments`-worker run gets exactly one segment per worker).
constexpr size_t kSegmentLen = 200;
// Worker count for the main "multi-thread sequence throughput" metric,
// passed explicitly to `solve_segments_parallel`'s `n_workers`, fixed rather
// than detected, matching perf.rs (see its comment).
constexpr size_t kMultithreadNThreads = 8;
// Frame count for the single-thread sequence-throughput metric, tiled from
// the 300-frame native-rate fixture: larger than the multi-thread metric's
// per-worker segment since this one has no worker count to divide by.
constexpr size_t kSingleThreadNFrames = 1000;

size_t frames_for_n_segments(size_t n_segments) { return kSegmentLen * n_segments; }

std::vector<std::vector<quickik::KeypointObservation>> tiled_native_rate_sequence(const Json &fixtures,
                                                                                   size_t length) {
  std::vector<std::vector<quickik::KeypointObservation>> base;
  for (auto &f : fixtures["native_rate_frames"].as_array()) base.push_back(build_observations(to_vec3s(f["target_ego"])));
  std::vector<std::vector<quickik::KeypointObservation>> out;
  out.reserve(length);
  for (size_t i = 0; i < length; i++) out.push_back(base[i % base.size()]);
  return out;
}

// `solve_segments_parallel` never reads or writes its `SequenceSolver`'s own
// running state, so (unlike `bench_solve_sequence`) the same instance is
// reused for both the warm-up and timed calls without biasing the result.
double bench_multithread_sequence_throughput(rust::Box<quickik::KinematicTree> &tree,
                                              const std::vector<std::vector<quickik::KeypointObservation>> &sequence,
                                              rust::isize n_workers) {
  size_t n_joints = sequence.front().size();
  std::vector<quickik::KeypointObservation> flat;
  flat.reserve(sequence.size() * n_joints);
  for (auto &obs : sequence) flat.insert(flat.end(), obs.begin(), obs.end());

  auto seq = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                           kPositionTolerance, kAngleTolerance, kDamping);
  auto run_once = [&] { return seq->solve_segments_parallel(slice_of(flat), n_joints, n_workers, false, false); };
  run_once();  // warm up
  auto t0 = Clock::now();
  auto results = run_once();
  double elapsed_ms = elapsed_us(t0) / 1e3;
  (void)results;
  return elapsed_ms;
}

// Writes ../plot/results/quickik-cpp-<body>.json for ../plot/plot_comparison.py
// to pick up. Hand-written (not using json.hpp, which is read-only) since
// this is the only place this binary needs to produce JSON.
void write_results_json(const std::string &body, double single_frame_latency_us, double single_frame_latency_max_us,
                         double single_thread_throughput_fps, double multi_thread_throughput_fps) {
  std::filesystem::path out_dir = std::filesystem::path(__FILE__).parent_path() / "../plot/results";
  std::filesystem::create_directories(out_dir);
  std::ofstream out(out_dir / ("quickik-cpp-" + body + ".json"));
  out << "{\n"
      << "  \"name\": \"quickik-cpp\",\n"
      << "  \"body\": \"" << body << "\",\n"
      << "  \"language\": \"cpp\",\n"
      << "  \"formulation\": \"whole-tree\",\n"
      << "  \"single_frame_latency_us\": " << single_frame_latency_us << ",\n"
      << "  \"single_frame_latency_max_us\": " << single_frame_latency_max_us << ",\n"
      << "  \"single_thread_throughput_fps\": " << single_thread_throughput_fps << ",\n"
      << "  \"multi_thread_throughput_fps\": " << multi_thread_throughput_fps << ",\n"
      << "  \"multi_thread_n_threads\": " << kMultithreadNThreads << ",\n"
      << "  \"notes\": null\n"
      << "}\n";
}

void run_performance(const std::string &body, rust::Box<quickik::KinematicTree> &tree, const Json &fixtures) {
  std::printf("quickik C++-bindings benchmark (state_dim=%zu)\n\n", tree->state_dim());

  // Same fixture-derived target used by the Rust and Python benchmarks, so
  // this number is directly comparable across all three.
  auto target = to_vec3s(fixtures["synthetic_frames"][0]["target_ego"]);
  auto obs = build_observations(target);
  std::printf("-- single-frame time (latency), default config (adaptive early stop) --\n");
  double single_frame_latency_us = summarize(
      "solve()",
      bench_single_frame_latency(tree, obs, 10000, kNeutralWeight, kPositionTolerance, kAngleTolerance));

  // Early stop disabled (tolerances = 0), so every call runs the full
  // n_iterations, the worst case if a frame never converges early.
  std::printf("\n-- single-frame time (latency), early stop disabled (%zu iterations) --\n", kNIterations);
  double single_frame_latency_max_us =
      summarize("solve() (forced max iterations)",
                bench_single_frame_latency(tree, obs, 10000, kNeutralWeight, 0.0f, 0.0f));

  std::printf("\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --\n");
  auto single_thread_sequence = tiled_native_rate_sequence(fixtures, kSingleThreadNFrames);
  double single_thread_mean_us = summarize("SequenceSolver.solve", bench_solve_sequence(tree, single_thread_sequence));

  std::printf("\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop, %zu threads) --\n",
              kMultithreadNThreads);
  auto sequence = tiled_native_rate_sequence(fixtures, frames_for_n_segments(kMultithreadNThreads));
  double elapsed_ms =
      bench_multithread_sequence_throughput(tree, sequence, static_cast<rust::isize>(kMultithreadNThreads));
  double multithread_fps = sequence.size() / (elapsed_ms / 1e3);
  std::printf("solve_segments_parallel             n_frames=%-6zu elapsed=%9.3fms  throughput=%10.1f frames/s\n",
              sequence.size(), elapsed_ms, multithread_fps);

  write_results_json(body, single_frame_latency_us, single_frame_latency_max_us, 1e6 / single_thread_mean_us,
                      multithread_fps);
}

}  // namespace

int main() {
#ifndef QUICKIK_ASSETS_DIR
#error "QUICKIK_ASSETS_DIR must be defined at compile time"
#endif
  const std::string assets_dir = QUICKIK_ASSETS_DIR;

  const std::vector<BodySpec> bodies = {
      {"neuromechfly", assets_dir + "/neuromechfly_ypr_legs.json", assets_dir + "/fixtures.json"},
      {"g1", assets_dir + "/g1_body_plan.json", assets_dir + "/fixtures_g1.json"},
  };

  for (auto &body : bodies) {
    std::printf("======================================================================\n");
    std::printf("Body: %s\n", body.name.c_str());
    std::printf("======================================================================\n\n");

    auto tree = quickik::kinematic_tree_from_json_file(body.body_plan_path);
    BodyPlan plan = load_body_plan(body.body_plan_path);
    Json fixtures = parse_json_file(body.fixtures_path);

    std::printf("Loaded body plan: %zu joints, %zu dofs, state_dim=%zu\n\n", tree->n_joints(), tree->n_dofs(),
                tree->state_dim());

    run_correctness(plan, fixtures, tree);
    run_performance(body.name, tree, fixtures);
  }
  return 0;
}
