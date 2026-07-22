// Performance benchmark for TRAC-IK against fastik's NeuroMechFly body plan
// (see ../../assets/neuromechfly_ypr_legs.json), mirroring
// ../../fastik_cpp/bench_cpp.cpp's methodology as closely as TRAC-IK's API
// allows. See README.md in this directory for the modeling compromises this
// requires and the exact build command.
//
// TRAC-IK only solves ONE kinematic chain against ONE end-effector Cartesian
// target per call (see test_lf_leg.cpp for the proof-of-concept and the API
// notes in trac_ik.hpp) -- there is no floating base and no intermediate
// waypoint fitting. So, unlike fastik (which solves the whole floating-base,
// six-leg tree jointly against all 30 leg-joint keypoints at once), this
// benchmark:
//   - treats "thorax" as a FIXED base (no floating-base DOFs at all);
//   - builds 6 independent KDL::Chain objects, thorax -> each leg's claw;
//   - fits ONLY each leg's claw (tip) 3D position, ignoring the coxa/femur/
//     tibia intermediate keypoints entirely (TRAC-IK has no way to fit them);
//   - solves the 6 legs via 6 independent, sequential CartToJnt() calls.
// "One frame" = all 6 of those calls. See the "notes" field of the emitted
// JSON, and README.md, for why this makes the comparison asymmetric vs.
// fastik's joint whole-tree solve.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "kdl/chain.hpp"
#include "kdl/frames.hpp"
#include "kdl/jntarray.hpp"
#include "trac_ik/trac_ik.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kNumLegs = 6;
// File order doesn't matter here since legs are looked up by prefix below,
// but this fixes the leg <-> array-slot mapping used throughout.
const std::array<std::string, kNumLegs> kLegPrefixes = {"lf", "lm", "lh", "rf", "rm", "rh"};

// ============================================================================
//  Body plan loading (../../assets/neuromechfly_ypr_legs.json) -> KDL::Chain
// ============================================================================

struct DofSpec {
  KDL::Vector axis;
  double neutral_angle;
};

struct JointSpec {
  std::string name;
  std::string parent;
  KDL::Vector offset_pos;
  std::vector<DofSpec> dofs;
};

std::unordered_map<std::string, JointSpec> load_joint_specs(const Json &body_plan) {
  std::unordered_map<std::string, JointSpec> joints;
  for (auto &j : body_plan["joints"].as_array()) {
    JointSpec spec;
    spec.name = j["name"].as_string();
    spec.parent = j["parent"].is_null() ? "" : j["parent"].as_string();
    auto &pos = j["offset_pos"].as_array();
    spec.offset_pos = KDL::Vector(pos[0].as_number(), pos[1].as_number(), pos[2].as_number());
    for (auto &d : j["dofs"].as_array()) {
      auto &axis = d["axis"].as_array();
      DofSpec dof;
      dof.axis = KDL::Vector(axis[0].as_number(), axis[1].as_number(), axis[2].as_number());
      dof.neutral_angle = d["neutral_angle"].as_number();
      spec.dofs.push_back(dof);
    }
    joints.emplace(spec.name, std::move(spec));
  }
  return joints;
}

// Walks parent links from each leg's *_thorax_coxa joint (found by prefix +
// parent=="thorax") down to its claw, rather than assuming file order --
// robust to the JSON's joints being reordered.
std::vector<JointSpec> leg_joint_chain(const std::unordered_map<std::string, JointSpec> &joints,
                                        const std::string &leg_prefix) {
  std::vector<JointSpec> chain;
  const JointSpec *current = nullptr;
  for (auto &[name, spec] : joints) {
    if (spec.parent == "thorax" && name.rfind(leg_prefix + "_", 0) == 0) {
      current = &spec;
      break;
    }
  }
  if (current == nullptr) {
    throw std::runtime_error("no root joint found for leg prefix: " + leg_prefix);
  }
  while (current != nullptr) {
    chain.push_back(*current);
    const JointSpec *next = nullptr;
    for (auto &[name, spec] : joints) {
      if (spec.parent == current->name) {
        next = &spec;
        break;
      }
    }
    current = next;
  }
  return chain;
}

KDL::Segment fixed_translate(const KDL::Vector &offset_pos) {
  return KDL::Segment(KDL::Joint(KDL::Joint::None), KDL::Frame(offset_pos));
}

KDL::Segment rot_dof(const KDL::Vector &axis, double neutral_angle) {
  return KDL::Segment(
      KDL::Joint(KDL::Vector(0, 0, 0), axis, KDL::Joint::RotAxis, /*scale=*/1.0, neutral_angle),
      KDL::Frame::Identity());
}

// Mirrors test_lf_leg.cpp's chain-building pattern: one fixed segment per
// joint (its offset_pos translation) followed by one RotAxis segment per
// DOF, with the body plan's neutral_angle encoded as the KDL Joint offset.
KDL::Chain build_leg_chain(const std::vector<JointSpec> &joint_chain) {
  KDL::Chain chain;
  for (auto &joint : joint_chain) {
    chain.addSegment(fixed_translate(joint.offset_pos));
    for (auto &dof : joint.dofs) {
      chain.addSegment(rot_dof(dof.axis, dof.neutral_angle));
    }
  }
  return chain;
}

// ============================================================================
//  Per-leg solver bundle
// ============================================================================

struct LegSolver {
  KDL::Chain chain;
  unsigned int n_dof = 0;
  std::unique_ptr<TRAC_IK::TRAC_IK> solver;
};

std::vector<LegSolver> build_leg_solvers(const std::unordered_map<std::string, JointSpec> &joints) {
  std::vector<LegSolver> legs(kNumLegs);
  for (int i = 0; i < kNumLegs; i++) {
    auto joint_chain = leg_joint_chain(joints, kLegPrefixes[i]);
    legs[i].chain = build_leg_chain(joint_chain);
    legs[i].n_dof = legs[i].chain.getNrOfJoints();
    KDL::JntArray q_min(legs[i].n_dof), q_max(legs[i].n_dof);
    for (unsigned int k = 0; k < legs[i].n_dof; k++) {
      q_min(k) = -M_PI;
      q_max(k) = M_PI;
    }
    // maxtime/eps/SolveType match test_lf_leg.cpp's verified-working values.
    legs[i].solver = std::make_unique<TRAC_IK::TRAC_IK>(legs[i].chain, q_min, q_max, /*maxtime=*/0.01,
                                                          /*eps=*/1e-5, TRAC_IK::Speed);
  }
  return legs;
}

KDL::JntArray zero_jnt_array(unsigned int n) {
  KDL::JntArray q(n);
  for (unsigned int i = 0; i < n; i++) q(i) = 0.0;
  return q;
}

// Position-only fit: huge rotational tolerance, zero positional tolerance --
// matches test_lf_leg.cpp (only the claw's 3D position is a tracked
// keypoint in the body plan; orientation is unconstrained).
const KDL::Twist kPositionOnlyTolerance(KDL::Vector(0, 0, 0), KDL::Vector(1e6, 1e6, 1e6));

// ============================================================================
//  Fixtures loading (../../assets/fixtures.json)
// ============================================================================

using Targets = std::array<KDL::Vector, kNumLegs>;

// fixtures.json's leg_joint_names lists 5 joints per leg (thorax_coxa,
// coxa_trochanterfemur, trochanterfemur_tibia, tibia_tarsus, claw); the claw
// is always the 5th, i.e. index 4 within each leg's block of 5.
Targets claw_targets_from_target_ego(const Json &target_ego) {
  Targets targets;
  auto &arr = target_ego.as_array();
  for (int leg = 0; leg < kNumLegs; leg++) {
    auto &p = arr[leg * 5 + 4].as_array();
    targets[leg] = KDL::Vector(p[0].as_number(), p[1].as_number(), p[2].as_number());
  }
  return targets;
}

std::vector<Targets> load_native_rate_targets(const Json &fixtures) {
  std::vector<Targets> out;
  for (auto &frame : fixtures["native_rate_frames"].as_array()) {
    out.push_back(claw_targets_from_target_ego(frame["target_ego"]));
  }
  return out;
}

// ============================================================================
//  Solve helpers
// ============================================================================

// Solves all 6 legs once (one "frame"), updating q_state in place (warm
// start for the next call). Returns nothing; timing is done by the caller.
void solve_frame(std::vector<LegSolver> &legs, std::vector<KDL::JntArray> &q_state, const Targets &targets) {
  for (int i = 0; i < kNumLegs; i++) {
    KDL::JntArray q_out(legs[i].n_dof);
    legs[i].solver->CartToJnt(q_state[i], KDL::Frame(targets[i]), q_out, kPositionOnlyTolerance);
    q_state[i] = q_out;
  }
}

std::vector<KDL::JntArray> neutral_state(const std::vector<LegSolver> &legs) {
  std::vector<KDL::JntArray> q_state;
  for (auto &leg : legs) q_state.push_back(zero_jnt_array(leg.n_dof));
  return q_state;
}

double elapsed_us(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// Prints the usual latency/throughput summary and returns the mean in
// microseconds (mirrors bench_cpp.cpp's summarize()).
double summarize(const std::string &label, std::vector<double> samples_us) {
  std::sort(samples_us.begin(), samples_us.end());
  size_t n = samples_us.size();
  double mean = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / n;
  auto pct = [&](double p) { return samples_us[static_cast<size_t>(std::round((n - 1) * p))]; };
  std::printf(
      "%-42s n=%-7zu mean=%9.3fus  median=%9.3fus  p95=%9.3fus  p99=%9.3fus  min=%9.3fus  "
      "max=%9.3fus  throughput=%10.1f frames/s\n",
      label.c_str(), n, mean, pct(0.50), pct(0.95), pct(0.99), samples_us.front(), samples_us.back(), 1e6 / mean);
  return mean;
}

// ============================================================================
//  Metric 1: single_frame_latency_us
// ============================================================================

double bench_single_frame_latency(const std::unordered_map<std::string, JointSpec> &joints, const Targets &target,
                                   int n_calls) {
  auto legs = build_leg_solvers(joints);
  for (int i = 0; i < 1000; i++) {
    auto q_state = neutral_state(legs);
    solve_frame(legs, q_state, target);
  }
  std::vector<double> samples;
  samples.reserve(n_calls);
  for (int i = 0; i < n_calls; i++) {
    auto q_state = neutral_state(legs);
    auto t0 = Clock::now();
    solve_frame(legs, q_state, target);
    samples.push_back(elapsed_us(t0));
  }
  return summarize("CartToJnt x6 (fresh, fixed target)", std::move(samples));
}

// ============================================================================
//  Metric 2: single_thread_throughput_fps
// ============================================================================

double bench_single_thread_sequence(const std::unordered_map<std::string, JointSpec> &joints,
                                     const std::vector<Targets> &native_rate_targets) {
  // Untimed warm-up pass (JIT/cache warm-up; own solvers + state, discarded).
  {
    auto legs = build_leg_solvers(joints);
    auto q_state = neutral_state(legs);
    for (auto &targets : native_rate_targets) solve_frame(legs, q_state, targets);
  }
  // Timed pass: fresh solvers + state, so the sequence's own frame-to-frame
  // warm-starting (not leftover state from the warm-up pass) is measured.
  auto legs = build_leg_solvers(joints);
  auto q_state = neutral_state(legs);
  std::vector<double> samples;
  samples.reserve(native_rate_targets.size());
  for (auto &targets : native_rate_targets) {
    auto t0 = Clock::now();
    solve_frame(legs, q_state, targets);
    samples.push_back(elapsed_us(t0));
  }
  double mean_us = summarize("solve_frame (warm-started, native-rate)", samples);
  return 1e6 / mean_us;
}

// ============================================================================
//  Metric 3: multi_thread_throughput_fps
// ============================================================================

// Fixed at 8 threads (not "whatever's available"), matching the fastik
// benchmarks' MULTITHREAD_N_THREADS. TRAC-IK has no built-in parallel/
// segmented solve path, so this is our own simplified scheme: a longer tiled
// sequence split into kNumThreads *contiguous, non-overlapping* chunks, each
// solved independently (own solvers, cold/neutral start at the chunk's
// start, warm-started within it) on its own std::thread. Unlike fastik's
// solve_sequence_segmented_parallel, there is no overlap/stitching between
// chunks -- see README.md / the results JSON "notes" field.
constexpr size_t kMultithreadNThreads = 8;
constexpr size_t kChunkLen = 200;
constexpr size_t kTotalMultithreadFrames = kMultithreadNThreads * kChunkLen;

std::vector<Targets> tile_targets(const std::vector<Targets> &base, size_t length) {
  std::vector<Targets> out;
  out.reserve(length);
  for (size_t i = 0; i < length; i++) out.push_back(base[i % base.size()]);
  return out;
}

void solve_chunk(const std::unordered_map<std::string, JointSpec> *joints, const std::vector<Targets> *chunk) {
  auto legs = build_leg_solvers(*joints);
  auto q_state = neutral_state(legs);
  for (auto &targets : *chunk) solve_frame(legs, q_state, targets);
}

double bench_multithread_sequence(const std::unordered_map<std::string, JointSpec> &joints,
                                   const std::vector<Targets> &tiled_sequence) {
  // Slice once (shared, read-only across threads).
  std::vector<std::vector<Targets>> chunks;
  for (size_t t = 0; t < kMultithreadNThreads; t++) {
    chunks.emplace_back(tiled_sequence.begin() + t * kChunkLen, tiled_sequence.begin() + (t + 1) * kChunkLen);
  }

  auto run = [&] {
    std::vector<std::thread> workers;
    workers.reserve(kMultithreadNThreads);
    for (size_t t = 0; t < kMultithreadNThreads; t++) {
      workers.emplace_back(solve_chunk, &joints, &chunks[t]);
    }
    for (auto &w : workers) w.join();
  };

  run();  // warm-up
  auto t0 = Clock::now();
  run();
  double elapsed_s = elapsed_us(t0) / 1e6;
  return kTotalMultithreadFrames / elapsed_s;
}

// ============================================================================
//  Results JSON (../../plot/results/trac-ik.json, ../../plot/RESULTS_SCHEMA.md)
// ============================================================================

void write_results_json(double single_frame_latency_us, double single_thread_throughput_fps,
                         double multi_thread_throughput_fps) {
  std::filesystem::path out_dir = std::filesystem::path(__FILE__).parent_path() / "../../plot/results";
  std::filesystem::create_directories(out_dir);
  std::ofstream out(out_dir / "trac-ik.json");
  const char *notes =
      "TRAC-IK solves one KDL::Chain against one Cartesian end-effector target per "
      "CartToJnt() call, with no floating base and no intermediate-waypoint fitting. "
      "Unlike fastik's joint whole-tree solve (floating thorax root + all 6 legs + every "
      "coxa/femur/tibia/claw keypoint, solved together each frame), this benchmark treats "
      "thorax as a FIXED base and runs 6 independent CartToJnt calls per frame, one per leg, "
      "each fitting ONLY that leg's claw (tip) 3D position -- intermediate joint keypoints "
      "are not fit at all, since TRAC-IK has no mechanism to do so. 'One frame' = all 6 "
      "sequential per-leg solves. multi_thread_throughput_fps uses a simplified scheme (no "
      "counterpart to TRAC-IK): a tiled native-rate sequence split into "
      "multi_thread_n_threads equal, CONTIGUOUS, non-overlapping chunks, each solved on its "
      "own std::thread with its own solver instances (cold/neutral start per chunk, "
      "warm-started within it) -- unlike fastik's overlap-stitched segmented parallel solve.";
  out << "{\n"
      << "  \"name\": \"trac-ik\",\n"
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
  const std::filesystem::path assets_dir = std::filesystem::path(__FILE__).parent_path() / "../../assets";

  Json body_plan = parse_json_file((assets_dir / "neuromechfly_ypr_legs.json").string());
  Json fixtures = parse_json_file((assets_dir / "fixtures.json").string());

  auto joints = load_joint_specs(body_plan);
  {
    auto legs = build_leg_solvers(joints);
    unsigned int total_dof = 0;
    for (auto &leg : legs) total_dof += leg.n_dof;
    std::printf("Loaded 6 leg chains (fixed thorax base), total actuated DOFs = %u\n\n", total_dof);
  }

  Targets synthetic_target = claw_targets_from_target_ego(fixtures["synthetic_frames"][0]["target_ego"]);
  auto native_rate_targets = load_native_rate_targets(fixtures);

  std::printf("-- single-frame latency (fresh/neutral start, fixed synthetic_frames[0] target) --\n");
  double single_frame_latency_us = bench_single_frame_latency(joints, synthetic_target, /*n_calls=*/3000);

  std::printf("\n-- single-thread sequence throughput (native-rate frames, warm-started) --\n");
  double single_thread_fps = bench_single_thread_sequence(joints, native_rate_targets);

  std::printf("\n-- multi-thread sequence throughput (%zu threads, %zu contiguous frames each) --\n",
              kMultithreadNThreads, kChunkLen);
  auto tiled_sequence = tile_targets(native_rate_targets, kTotalMultithreadFrames);
  double multi_thread_fps = bench_multithread_sequence(joints, tiled_sequence);
  std::printf("total_frames=%zu  throughput=%10.1f frames/s\n", kTotalMultithreadFrames, multi_thread_fps);

  write_results_json(single_frame_latency_us, single_thread_fps, multi_thread_fps);
  std::printf("\nWrote ../../plot/results/trac-ik.json\n");

  return 0;
}
