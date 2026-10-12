// Correctness cross-check and throughput/latency benchmark for quickik's C++
// bindings, mirroring ../quickik_python/bench.py and
// ../quickik_rust/src/{correctness,perf}.rs so all three are directly
// comparable. Built along with the bindings' tests (from the repo root):
//
//   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
//   cmake --build cpp/build -j
//   ./cpp/build/quickik_cpp_benchmark
//
// For the keypoint comparisons, forward_kinematics.hpp reimplements forward
// kinematics directly from the JSON body plan rather than calling
// `quickik::forward_kinematics`, so they are an independent check of quickik's
// solved state.
//
// Solver settings: the default `SolverConfig` plus a DOF neutral prior of
// weight `kNeutralWeight`, matching the Python benchmark.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "forward_kinematics.hpp"
#include "json.hpp"
#include "quickik.hpp"

namespace {

using Clock = std::chrono::steady_clock;

// Weight of the DOF neutral prior.
constexpr float kNeutralWeight = 1e-3f;

// One body to benchmark: a name (used for the results filename and printed
// headers) plus the paths to its body plan and fixtures.
struct BodySpec {
  std::string name;
  std::string body_plan_path;
  std::string fixtures_path;
};

/// The default config plus a DOF neutral prior. `early_stop = false` sets both
/// tolerances to 0, to force the full iteration count.
std::unique_ptr<quickik::SolverConfig> solver_config(
    float neutral_weight = kNeutralWeight, bool early_stop = true) {
  auto config = quickik::SolverConfig::create();
  if (!early_stop)
    config = quickik::SolverConfig::create(config->n_iterations(), 0.0f, 0.0f,
                                           config->damping());
  quickik::unwrap(config->add_dof_neutral_prior(neutral_weight));
  return config;
}

std::vector<Vec3> to_vec3s(const Json& target_ego) {
  std::vector<Vec3> out;
  for (auto& p : target_ego.as_array()) {
    out.push_back({static_cast<float>(p[0].as_number()),
                   static_cast<float>(p[1].as_number()),
                   static_cast<float>(p[2].as_number())});
  }
  return out;
}

/// Adds every keypoint but the root's (joint 0, which `target_ego` doesn't
/// cover and isn't observed) to `observations`.
void add_observations(quickik::KeypointObservationList& observations,
                      const std::vector<Vec3>& target_ego) {
  for (size_t i = 0; i < target_ego.size(); i++) {
    const auto& p = target_ego[i];
    observations.add_position_3d(i + 1, {p.x, p.y, p.z}, 1.0f);
  }
}

std::unique_ptr<quickik::KeypointObservationList> build_observations(
    const std::vector<Vec3>& target_ego) {
  auto observations = quickik::KeypointObservationList::create();
  add_observations(*observations, target_ego);
  return observations;
}

/// `frames[i % frames.size()]["target_ego"]` for each of `length` frames, as
/// one sequence.
std::unique_ptr<quickik::ObservationSequence> tiled_sequence(
    const std::vector<Json>& frames, size_t length) {
  auto observations = quickik::KeypointObservationList::create();
  std::vector<size_t> frame_lengths;
  for (size_t i = 0; i < length; i++) {
    auto target = to_vec3s(frames[i % frames.size()]["target_ego"]);
    add_observations(*observations, target);
    frame_lengths.push_back(target.size());
  }
  return quickik::unwrap(
      quickik::ObservationSequence::from_flat(*observations, frame_lengths));
}

std::vector<Vec3> solved_keypoints(const BodyPlan& plan,
                                   const quickik::State& state) {
  auto dof_values = state.dof_values();
  auto p = state.root_pos();
  auto q = state.root_quat();
  return forward_kinematics(
      plan, std::vector<float>(dof_values.begin(), dof_values.end()),
      Vec3{p.x, p.y, p.z}, Quat{q.w, q.x, q.y, q.z});
}

// a has one entry per joint (root included); b covers only leg joints (root
// excluded), so compare a[1..] against b.
std::pair<float, float> residual_stats(const std::vector<Vec3>& a,
                                       const std::vector<Vec3>& b) {
  std::vector<float> dists;
  for (size_t i = 0; i < b.size(); i++)
    dists.push_back((a[i + 1] - b[i]).norm());
  float sum_sq = 0;
  for (float d : dists) sum_sq += d * d;
  float rms = std::sqrt(sum_sq / dists.size());
  float max = *std::max_element(dists.begin(), dists.end());
  return {rms, max};
}

float angle_error_deg(quickik::diplomat::span<const float> solved,
                      const std::vector<float>& ground_truth) {
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
//  Correctness (mirrors bench.py's run_correctness)
// =============================================================================

void run_correctness(const BodyPlan& plan, const Json& fixtures,
                     const quickik::KinematicTree& tree) {
  std::printf("== Synthetic exact-fit frames (bug hunt) ==\n");
  std::printf("%6s %16s %16s %18s %20s\n", "frame", "kpt rms", "kpt max",
              "angle err deg", "angle err deg (w=0)");

  auto default_solver = quickik::unwrap(
      quickik::SingleFrameSolver::create(tree, *solver_config()));
  auto zero_reg_solver = quickik::unwrap(
      quickik::SingleFrameSolver::create(tree, *solver_config(0.0f)));

  size_t i = 0;
  for (auto& frame : fixtures["synthetic_frames"].as_array()) {
    auto target = to_vec3s(frame["target_ego"]);
    std::vector<float> ground_truth;
    for (auto& leg : frame["ground_truth_dof_angles_per_leg"].as_array()) {
      for (auto& a : leg.as_array())
        ground_truth.push_back(static_cast<float>(a.as_number()));
    }
    auto observations = build_observations(target);

    auto state = quickik::State::neutral_pose(tree);
    auto result = quickik::unwrap(
        default_solver->solve(*state, *observations, false, false));
    auto [rms, max] =
        residual_stats(solved_keypoints(plan, result->state()), target);
    float angle_err =
        angle_error_deg(result->state().dof_values(), ground_truth);

    auto state0 = quickik::State::neutral_pose(tree);
    auto result0 = quickik::unwrap(
        zero_reg_solver->solve(*state0, *observations, false, false));
    float angle_err0 =
        angle_error_deg(result0->state().dof_values(), ground_truth);

    std::printf("%6zu %16.6f %16.6f %18.4f %20.6f\n", i, rms, max, angle_err,
                angle_err0);
    i++;
  }
  std::printf(
      "(kpt rms/max: 3D distance to target, via an independent from-JSON FK "
      "replica, model "
      "units. angle err: max abs error over all %zu DOFs, degrees, mod 2*pi. "
      "\"w=0\" = "
      "weight=0.)\n\n",
      tree.n_dofs());

  std::printf("== Real mocap frames (cross-solver vs. flygym.ik) ==\n");
  auto& real_frames = fixtures["real_frames"].as_array();
  auto sequence = tiled_sequence(real_frames, real_frames.size());
  auto solver = quickik::unwrap(
      quickik::SequentialSolver::create(tree, *solver_config(), nullptr));
  auto results = quickik::unwrap(solver->solve(*sequence, false, false));

  // Per-frame (rms, max) first, then aggregated across frames, matching the
  // Rust and Python benchmarks.
  std::vector<float> quickik_rms, quickik_max, cross_rms, cross_max;
  for (size_t idx = 0; idx < real_frames.size(); idx++) {
    auto& frame = real_frames[idx];
    auto solved_pts = solved_keypoints(plan, (*results)[idx]->state());

    auto [rms, max] = residual_stats(solved_pts, to_vec3s(frame["target_ego"]));
    quickik_rms.push_back(rms);
    quickik_max.push_back(max);

    // Not every body has an independent reference solver to cross-check
    // against (e.g. G1's real frames have no flygym.ik reconstruction).
    if (frame.has("flygym_ik_reconstructed_ego")) {
      auto [crms, cmax] = residual_stats(
          solved_pts, to_vec3s(frame["flygym_ik_reconstructed_ego"]));
      cross_rms.push_back(crms);
      cross_max.push_back(cmax);
    }
  }
  auto mean = [](const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
  };
  auto rms_of = [](const std::vector<float>& v) {
    float s = 0;
    for (float x : v) s += x * x;
    return std::sqrt(s / v.size());
  };
  auto max_of = [](const std::vector<float>& v) {
    return *std::max_element(v.begin(), v.end());
  };
  std::printf("over %zu frames:\n", real_frames.size());
  std::printf(
      "  quickik fit residual to target:      rms=%.5f  mean=%.5f  max=%.5f\n",
      rms_of(quickik_rms), mean(quickik_rms), max_of(quickik_max));
  if (!cross_rms.empty()) {
    std::printf(
        "  cross-solver agreement (vs flygym.ik): rms=%.5f  mean=%.5f  "
        "max=%.5f\n\n",
        rms_of(cross_rms), mean(cross_rms), max_of(cross_max));
  } else {
    std::printf(
        "  cross-solver agreement (vs flygym.ik): n/a (no reference in "
        "fixtures)\n\n");
  }
}

// =============================================================================
//  Performance (mirrors bench.py's run_performance)
// =============================================================================

// Prints the usual latency/throughput summary and returns the mean in
// microseconds, for callers that also want the number for
// ../plot/results/quickik-cpp-<body>.json.
double summarize(const std::string& label, std::vector<double> samples_us) {
  std::sort(samples_us.begin(), samples_us.end());
  size_t n = samples_us.size();
  double mean = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / n;
  auto pct = [&](double p) {
    return samples_us[static_cast<size_t>(std::round((n - 1) * p))];
  };
  std::printf(
      "%-42s n=%-7zu mean=%9.3fus  median=%9.3fus  p95=%9.3fus  p99=%9.3fus  "
      "min=%9.3fus  "
      "max=%9.3fus  throughput=%10.1f calls/s\n",
      label.c_str(), n, mean, pct(0.50), pct(0.95), pct(0.99),
      samples_us.front(), samples_us.back(), 1e6 / mean);
  return mean;
}

double elapsed_us(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// Single-frame latency: a fresh `State::neutral_pose` solved against a fixed
// real target every call (no warm start).
std::vector<double> bench_single_frame_latency(
    const quickik::KinematicTree& tree,
    const quickik::KeypointObservationList& observations, int n_calls,
    bool early_stop) {
  auto solver = quickik::unwrap(quickik::SingleFrameSolver::create(
      tree, *solver_config(kNeutralWeight, early_stop)));
  for (int i = 0; i < 500; i++) {
    auto state = quickik::State::neutral_pose(tree);
    quickik::unwrap(solver->solve(*state, observations, false, false));
  }
  std::vector<double> samples;
  samples.reserve(n_calls);
  for (int i = 0; i < n_calls; i++) {
    auto state = quickik::State::neutral_pose(tree);
    auto t0 = Clock::now();
    auto result = solver->solve(*state, observations, false, false);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// Single-thread sequence throughput: one `SequentialSolver::solve` call over a
// tiled native-rate sequence. A second, fresh `SequentialSolver` is used for
// the timed pass after warming up once, so the sequence's own frame-to-frame
// warm-starting (not cross-call warm-starting from the warm-up pass) is what's
// measured. Returns the elapsed time in milliseconds.
double bench_solve_sequence(const quickik::KinematicTree& tree,
                            const quickik::ObservationSequence& sequence) {
  auto warmup_solver = quickik::unwrap(
      quickik::SequentialSolver::create(tree, *solver_config(), nullptr));
  quickik::unwrap(warmup_solver->solve(sequence, false, false));

  auto timed_solver = quickik::unwrap(
      quickik::SequentialSolver::create(tree, *solver_config(), nullptr));
  auto t0 = Clock::now();
  auto results = timed_solver->solve(sequence, false, false);
  return elapsed_us(t0) / 1e3;
}

// Frames per segment/worker, matching perf.rs exactly (same stride, so a
// `n_segments`-worker run gets exactly one segment per worker).
constexpr size_t kSegmentLen = 200;
// Worker count for the main "multi-thread sequence throughput" metric, passed
// explicitly to `solve_parallel`'s `n_workers`, fixed rather than detected,
// matching perf.rs (see its comment).
constexpr size_t kMultithreadNThreads = 8;
// Frame count for the single-thread sequence-throughput metric, tiled from
// the 300-frame native-rate fixture: larger than the multi-thread metric's
// per-worker segment since this one has no worker count to divide by.
constexpr size_t kSingleThreadNFrames = 1000;

// `solve_parallel` never reads or writes its `SequentialSolver`'s own running
// state, so (unlike `bench_solve_sequence`) the same instance is reused for
// both the warm-up and timed calls without biasing the result. Segments don't
// overlap (`n_overlap = 0`), matching the Rust benchmark. Returns the elapsed
// time in milliseconds.
double bench_multithread_sequence_throughput(
    const quickik::KinematicTree& tree,
    const quickik::ObservationSequence& sequence, intptr_t n_workers) {
  auto solver = quickik::unwrap(
      quickik::SequentialSolver::create(tree, *solver_config(), nullptr));
  // warm up
  quickik::unwrap(solver->solve_parallel(sequence, n_workers, 0, false, false));
  auto t0 = Clock::now();
  auto results = solver->solve_parallel(sequence, n_workers, 0, false, false);
  return elapsed_us(t0) / 1e3;
}

// Writes ../plot/results/quickik-cpp-<body>.json for ../plot/plot_comparison.py
// to pick up. Hand-written (not using json.hpp, which is read-only) since
// this is the only place this binary needs to produce JSON.
void write_results_json(const std::string& body, double single_frame_latency_us,
                        double single_frame_latency_max_us,
                        double single_thread_throughput_fps,
                        double multi_thread_throughput_fps) {
  std::filesystem::path out_dir =
      std::filesystem::path(__FILE__).parent_path() / "../plot/results";
  std::filesystem::create_directories(out_dir);
  std::ofstream out(out_dir / ("quickik-cpp-" + body + ".json"));
  out << "{\n"
      << "  \"name\": \"quickik-cpp\",\n"
      << "  \"body\": \"" << body << "\",\n"
      << "  \"language\": \"cpp\",\n"
      << "  \"formulation\": \"whole-tree\",\n"
      << "  \"single_frame_latency_us\": " << single_frame_latency_us << ",\n"
      << "  \"single_frame_latency_max_us\": " << single_frame_latency_max_us
      << ",\n"
      << "  \"single_thread_throughput_fps\": " << single_thread_throughput_fps
      << ",\n"
      << "  \"multi_thread_throughput_fps\": " << multi_thread_throughput_fps
      << ",\n"
      << "  \"multi_thread_n_threads\": " << kMultithreadNThreads << ",\n"
      << "  \"notes\": null\n"
      << "}\n";
}

void run_performance(const std::string& body,
                     const quickik::KinematicTree& tree, const Json& fixtures) {
  std::printf("quickik C++-bindings benchmark (state_dim=%zu)\n\n",
              tree.state_dim());

  // Same fixture-derived target used by the Rust and Python benchmarks, so
  // this number is directly comparable across all three.
  auto observations = build_observations(
      to_vec3s(fixtures["synthetic_frames"][0]["target_ego"]));
  std::printf(
      "-- single-frame time (latency), default config (adaptive early stop) "
      "--\n");
  double single_frame_latency_us = summarize(
      "solve()", bench_single_frame_latency(tree, *observations, 10000, true));

  // Early stop disabled (tolerances = 0), so every call runs the full
  // n_iterations, the worst case if a frame never converges early.
  std::printf(
      "\n-- single-frame time (latency), early stop disabled (%zu iterations) "
      "--\n",
      quickik::SolverConfig::create()->n_iterations());
  double single_frame_latency_max_us =
      summarize("solve() (forced max iterations)",
                bench_single_frame_latency(tree, *observations, 10000, false));

  std::printf(
      "\n-- single-thread sequence throughput (native-rate frames, adaptive "
      "early stop) --\n");
  auto& native_rate_frames = fixtures["native_rate_frames"].as_array();
  auto sequence = tiled_sequence(native_rate_frames, kSingleThreadNFrames);
  double elapsed_ms = bench_solve_sequence(tree, *sequence);
  double single_thread_fps = sequence->len() / (elapsed_ms / 1e3);
  std::printf(
      "SequentialSolver::solve             n_frames=%-6zu elapsed=%9.3fms  "
      "throughput=%10.1f frames/s\n",
      sequence->len(), elapsed_ms, single_thread_fps);

  std::printf(
      "\n-- multi-thread sequence throughput (segmented parallel, adaptive "
      "early stop, %zu threads) --\n",
      kMultithreadNThreads);
  sequence =
      tiled_sequence(native_rate_frames, kSegmentLen * kMultithreadNThreads);
  elapsed_ms = bench_multithread_sequence_throughput(tree, *sequence,
                                                     kMultithreadNThreads);
  double multithread_fps = sequence->len() / (elapsed_ms / 1e3);
  std::printf(
      "solve_parallel                      n_frames=%-6zu elapsed=%9.3fms  "
      "throughput=%10.1f frames/s\n",
      sequence->len(), elapsed_ms, multithread_fps);

  write_results_json(body, single_frame_latency_us, single_frame_latency_max_us,
                     single_thread_fps, multithread_fps);
}

}  // namespace

int main() {
#ifndef QUICKIK_ASSETS_DIR
#error "QUICKIK_ASSETS_DIR must be defined at compile time"
#endif
  const std::string assets_dir = QUICKIK_ASSETS_DIR;

  const std::vector<BodySpec> bodies = {
      {"neuromechfly",
       assets_dir + "/../../examples/assets/neuromechfly/body_plan.json",
       assets_dir + "/fixtures.json"},
      {"g1", assets_dir + "/../../examples/assets/g1/body_plan.json",
       assets_dir + "/fixtures_g1.json"},
  };

  for (auto& body : bodies) {
    std::printf("===== body: %s =====\n\n", body.name.c_str());

    auto tree = quickik::unwrap(
        quickik::KinematicTree::from_json_file(body.body_plan_path));
    BodyPlan plan = load_body_plan(body.body_plan_path);
    Json fixtures = parse_json_file(body.fixtures_path);

    run_correctness(plan, fixtures, *tree);
    run_performance(body.name, *tree, fixtures);
    std::printf("\n");
  }
  return 0;
}
