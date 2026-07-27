// Throughput/latency benchmark for Orocos KDL, mirroring
// ../../quickik_rust/src/perf.rs and ../../quickik_cpp/bench_cpp.cpp so all
// three (plus quickik-python) are directly comparable. See ../../quickik_cpp/
// for the shared, dependency-free json.hpp / forward_kinematics.hpp copied
// into this directory (per that file's own header comment on why an
// independent FK/JSON reader is used at all).
//
// -----------------------------------------------------------------------
// Modeling compromises (see README.md for the long version):
//
// 1. Floating base: KDL has no native 6-DOF floating joint, so QuickIK's
//    free-floating "thorax" root (parametrized as a translation + a
//    singularity-free unit quaternion) is represented here as 6 scalar
//    joints in series -- TransX, TransY, TransZ, RotZ, RotY, RotX -- i.e. a
//    sequential Euler-angle-like parametrization. This is functionally
//    workable (same reachable pose space, modulo gimbal lock) but not
//    mathematically identical to QuickIK's quaternion root.
//
// 2. Position-only IK with a full-SE(3) solver: `TreeIkSolverPos_NR_JL`
//    (built on `TreeIkSolverVel_wdls`) solves for full 6D pose (position +
//    orientation) per endpoint, but QuickIK only fits 3D keypoint positions.
//    We zero out the 3 rotational rows of the task-space weighting matrix
//    (`TreeIkSolverVel_wdls::setWeightTS`) for every endpoint, so orientation
//    error never drives the solve and the target orientation we supply is a
//    don't-care (we pass identity). This is the standard way to do
//    position-only IK with a full-pose solver.
//
// 3. Every joint in the JSON body plan is added as a named tree segment at
//    its *own* local origin (before its own dof rotations are applied,
//    exactly matching QuickIK's convention that a joint's own rotation only
//    re-orients its children, never displaces itself -- see
//    kdl_leg_fk_test.cpp's header comment) so every one of the 30 non-root
//    joints can be used as both an IK target and an FK query point.
//
// 4. Solve loop: NOT `TreeIkSolverPos_NR_JL` (KDL's own convenience wrapper
//    around `TreeIkSolverVel_wdls`). Its only early-stop check is
//    `residual_norm < eps` (treeiksolverpos_nr_jl.cpp) -- the combined L2
//    norm of ALL 30 endpoints' position error at once. On real (imperfectly
//    fittable) mocap data that residual has a floor around 0.08-0.13, well
//    above any `eps` tight enough to be meaningful, so the check never
//    triggers and every solve silently burns the full `kMaxIter`. `solve()`
//    below reimplements the same per-iteration math (call the velocity
//    solver, apply the update) directly against `TreeIkSolverVel_wdls`, and
//    adds the step-size check `TreeIkSolverPos_NR_JL` lacks: max abs
//    component of the per-iteration joint delta, matching QuickIK's own
//    `position_tolerance`/`angle_tolerance` semantics exactly.

#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/segment.hpp>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treeiksolverpos_nr_jl.hpp>
#include <kdl/treeiksolvervel_wdls.hpp>
#include <kdl/tree.hpp>

#include "forward_kinematics.hpp"
#include "json.hpp"

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

namespace {

using Clock = std::chrono::steady_clock;
using KDL::Frame;
using KDL::Frames;
using KDL::Joint;
using KDL::JntArray;
using KDL::Rotation;
using KDL::Segment;
using KDL::Tree;
using KDL::TreeFkSolverPos_recursive;
using KDL::TreeIkSolverVel_wdls;
using KDL::Twist;
using KDL::Twists;
using KDL::Vector;
using KDL::diff;

constexpr size_t kFloatingBaseDofs = 6;
constexpr unsigned int kMaxIter = 100;
// Lower than QuickIK/RBDL/Pinocchio's shared 10000, since KDL's own per-call
// latency is 1-2 orders of magnitude higher, making its single-frame-latency
// sweeps alone take much longer at that count. Not directly comparable
// sample-count-wise to the other libraries' numbers, but 1000 still gives a
// stable mean/median/p95 (50 samples in the p95 tail), just a noisier p99
// (10 samples in the tail vs 100).
constexpr size_t kLatencyNCalls = 1000;
// Frame count for the single-thread sequence-throughput metric, tiled from
// the 300-frame native-rate fixture. Lower than the other libraries' 1000
// for the same per-call-latency reason as kLatencyNCalls above -- a
// throughput metric already averages over many frames, so it needs fewer of
// them than an isolated per-call latency measurement to be stable.
constexpr size_t kSingleThreadNFrames = 100;
// Matches QuickIK's own default `position_tolerance`/`angle_tolerance` (see
// ../../src/solver.rs's `SolverConfig::default()`) -- see file header note 4
// for why this gates our own step-size check, not `TreeIkSolverPos_NR_JL`'s
// (non-functional, for this data) residual check.
constexpr double kStepTol = 1e-3;
constexpr double kLambda = 0.05;
// All JSON dof limits are null (unbounded), so no joint-limit clamping is
// needed in solve() -- unlike TreeIkSolverPos_NR_JL, which required
// q_min/q_max even when unused.

// A built KDL::Tree for the neuromechfly_ypr_legs body plan, plus the
// bookkeeping needed to (a) address each non-root joint as an IK
// target/FK query segment and (b) lay out q vectors the same way QuickIK's
// flat dof_angles array does (see build_model()'s comment).
struct KdlModel {
  Tree tree;
  // keypoint_segment[i] is the tree segment name representing joint i's
  // *own* position (pre-own-rotation), for i in [1, joints.size()) --
  // index 0 (the thorax root) is unused since it has no keypoint target.
  std::vector<std::string> keypoint_segment;
  std::vector<std::string> endpoints;  // keypoint_segment[1..], the IK/FK target list
  size_t n_dofs = 0;                   // real (non-floating-base) dofs == plan dof count
};

// Builds the tree: 6 floating-base scalar joints in series, then every real
// joint as (Fixed offset segment) + (one RotAxis segment per dof), hooked
// onto its parent's *last* segment (which carries the parent's full
// rotation, needed to place this joint's own offset correctly) -- see
// kdl_leg_fk_test.cpp for why this exactly reproduces QuickIK's FK.
//
// Because Tree::addSegment only assigns a q-vector slot to non-Fixed
// joints, and we add segments in exactly JSON order (floating base first,
// then each joint's dofs in order), the resulting JntArray layout is
// [6 floating-base dofs][42 real dofs in QuickIK's own dof_offset order] --
// i.e. real dof `d` (0-indexed, QuickIK's flat numbering) lives at KDL q
// index `kFloatingBaseDofs + d`.
KdlModel build_model(const BodyPlan &plan) {
  KdlModel m;
  m.tree = Tree("world");
  m.tree.addSegment(Segment("fb_transx", Joint(Joint::TransX)), "world");
  m.tree.addSegment(Segment("fb_transy", Joint(Joint::TransY)), "fb_transx");
  m.tree.addSegment(Segment("fb_transz", Joint(Joint::TransZ)), "fb_transy");
  m.tree.addSegment(Segment("fb_rotz", Joint(Joint::RotZ)), "fb_transz");
  m.tree.addSegment(Segment("fb_roty", Joint(Joint::RotY)), "fb_rotz");
  m.tree.addSegment(Segment("fb_rotx", Joint(Joint::RotX)), "fb_roty");
  m.tree.addSegment(Segment("thorax", Joint(Joint::Fixed)), "fb_rotx");

  m.keypoint_segment.resize(plan.joints.size());
  m.keypoint_segment[0] = "thorax";
  std::vector<std::string> tip_segment(plan.joints.size());
  tip_segment[0] = "thorax";

  size_t n_dofs = 0;
  for (size_t i = 1; i < plan.joints.size(); i++) {
    const auto &j = plan.joints[i];
    const std::string &hook = tip_segment[j.parent];
    Vector offset(j.offset_pos.x, j.offset_pos.y, j.offset_pos.z);
    if (j.dofs.empty()) {
      m.tree.addSegment(Segment(j.name, Joint(Joint::Fixed), Frame(offset)), hook);
      m.keypoint_segment[i] = j.name;
      tip_segment[i] = j.name;
    } else {
      std::string offset_name = j.name + "_offset";
      m.tree.addSegment(Segment(offset_name, Joint(Joint::Fixed), Frame(offset)), hook);
      m.keypoint_segment[i] = offset_name;
      std::string cur = offset_name;
      for (size_t d = 0; d < j.dofs.size(); d++) {
        std::string dof_name = j.name + "_dof" + std::to_string(d);
        Vector axis(j.dofs[d].axis.x, j.dofs[d].axis.y, j.dofs[d].axis.z);
        m.tree.addSegment(Segment(dof_name, Joint(dof_name, Vector::Zero(), axis, Joint::RotAxis)), cur);
        cur = dof_name;
        n_dofs++;
      }
      tip_segment[i] = cur;
    }
  }
  m.n_dofs = n_dofs;
  for (size_t i = 1; i < m.keypoint_segment.size(); i++) m.endpoints.push_back(m.keypoint_segment[i]);
  return m;
}

// Per-thread (and per-main-solve) solver bundle: KDL's tree IK/FK solvers
// hold mutable Eigen scratch buffers as member state, so each concurrent
// solve needs its own instances -- built fresh from the (read-only,
// freely-shareable) BodyPlan. No joint-limit clamping needed: all JSON dof
// limits are null, and `q_min`/`q_max` are only used by
// `TreeIkSolverPos_NR_JL`, which this file doesn't use (see file header
// note 4).
struct SolverBundle {
  KdlModel model;
  TreeFkSolverPos_recursive fk;
  TreeIkSolverVel_wdls vel;

  explicit SolverBundle(const BodyPlan &plan)
      : model(build_model(plan)), fk(model.tree), vel(model.tree, model.endpoints) {
    vel.setLambda(kLambda);
    // Position-only IK: zero the 3 rotational rows of every endpoint's 6-row
    // task-space block (see the file header's compromise #2).
    size_t rows = 6 * model.endpoints.size();
    Eigen::MatrixXd Wy = Eigen::MatrixXd::Zero(rows, rows);
    for (size_t k = 0; k < model.endpoints.size(); k++)
      for (int r = 0; r < 3; r++) Wy(6 * k + r, 6 * k + r) = 1.0;
    vel.setWeightTS(Wy);
  }

  JntArray neutral(const std::vector<double> &neutral_angles) const {
    JntArray q(model.tree.getNrOfJoints());
    for (unsigned int i = 0; i < kFloatingBaseDofs; i++) q(i) = 0.0;
    for (size_t d = 0; d < neutral_angles.size(); d++) q(kFloatingBaseDofs + d) = neutral_angles[d];
    return q;
  }
};

// Reimplements TreeIkSolverPos_NR_JL::CartToJnt's own per-iteration math
// (FK each endpoint, diff to target, one TreeIkSolverVel_wdls step, apply
// the update) but with a working step-size early-stop instead of its
// residual-only check -- see file header note 4. Returns the solved q;
// `q_out` may alias `q_init`. `step_tol=0` disables early stopping, forcing
// every solve to run the full `max_iter`.
JntArray solve(SolverBundle &sb, const JntArray &q_init, const Frames &target, unsigned int max_iter = kMaxIter,
                double step_tol = kStepTol) {
  JntArray q = q_init;
  JntArray delta_q(q.rows());
  Frame cur_frame;
  Twists delta_twists;
  for (auto &kv : target) delta_twists[kv.first] = Twist::Zero();

  for (unsigned int iter = 0; iter < max_iter; iter++) {
    for (auto &kv : target) {
      sb.fk.JntToCart(q, cur_frame, kv.first);
      delta_twists[kv.first] = diff(cur_frame, kv.second);
    }
    sb.vel.CartToJnt(q, delta_twists, delta_q);

    double max_delta = 0.0;
    for (unsigned int i = 0; i < delta_q.rows(); i++) max_delta = std::max(max_delta, std::abs(delta_q(i)));
    for (unsigned int i = 0; i < q.rows(); i++) q(i) += delta_q(i);
    if (max_delta < step_tol) break;
  }
  return q;
}

// Flat, in-quickik-dof-order neutral angles (BodyPlan itself doesn't carry
// them -- only dof axes -- so read them directly from the JSON here).
std::vector<double> load_neutral_angles(const std::string &path) {
  Json root = parse_json_file(path);
  std::vector<double> out;
  for (auto &j : root["joints"].as_array())
    for (auto &d : j["dofs"].as_array()) out.push_back(d["neutral"].as_number());
  return out;
}

std::vector<Vec3> to_vec3s(const Json &target_ego) {
  std::vector<Vec3> out;
  for (auto &p : target_ego.as_array())
    out.push_back({static_cast<float>(p[0].as_number()), static_cast<float>(p[1].as_number()),
                    static_cast<float>(p[2].as_number())});
  return out;
}

// Builds the IK target map: keypoint_segment[i+1] -> Frame(target position,
// don't-care orientation) for each of the 30 non-root joints, matching
// QuickIK's convention of a `Missing` root observation (see file header
// comment #2 for why orientation is a don't-care here).
Frames build_target_frames(const KdlModel &model, const std::vector<Vec3> &target_ego) {
  Frames out;
  for (size_t k = 0; k < target_ego.size(); k++) {
    const Vec3 &p = target_ego[k];
    out[model.keypoint_segment[k + 1]] = Frame(Rotation::Identity(), Vector(p.x, p.y, p.z));
  }
  return out;
}

double elapsed_us(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

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

// -----------------------------------------------------------------------
// Metric 1: single_frame_latency_us -- fresh neutral-pose solve against the
// fixed synthetic_frames[0] target every call, no warm start.
std::vector<double> bench_single_frame_latency(SolverBundle &sb, const JntArray &q_neutral, const Frames &target,
                                                int n_calls, int n_warmup, unsigned int max_iter = kMaxIter,
                                                double step_tol = kStepTol) {
  for (int i = 0; i < n_warmup; i++) solve(sb, q_neutral, target, max_iter, step_tol);

  std::vector<double> samples;
  samples.reserve(n_calls);
  for (int i = 0; i < n_calls; i++) {
    auto t0 = Clock::now();
    solve(sb, q_neutral, target, max_iter, step_tol);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// -----------------------------------------------------------------------
// Metric 2: single_thread_throughput_fps -- warm-started sequential solve
// over the 300-frame native-rate fixture (frame i seeded from frame i-1's
// solution).
std::vector<double> bench_sequence(SolverBundle &sb, const JntArray &q_neutral, const std::vector<Frames> &frames) {
  // max_iter=10 (not this file's own kMaxIter=100) to match quickik's/RBDL's
  // hard iteration cap for the equivalent metric -- see run_body's own
  // single_frame_latency calls and README.md's methodology note. Without
  // this, KDL's warm-started throughput solves can silently run up to 10x
  // more iterations per frame than the other two libraries' for the same
  // metric, biasing the comparison.
  constexpr unsigned int kThroughputMaxIter = 10;
  JntArray q(q_neutral);
  for (auto &target : frames) q = solve(sb, q, target, kThroughputMaxIter);  // untimed warmup pass

  q = q_neutral;
  std::vector<double> samples;
  samples.reserve(frames.size());
  for (auto &target : frames) {
    auto t0 = Clock::now();
    q = solve(sb, q, target, kThroughputMaxIter);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// -----------------------------------------------------------------------
// Metric 3: multi_thread_throughput_fps -- a longer tiled sequence split
// into kNThreads contiguous, roughly-equal chunks, each solved on its own
// std::thread: warm-started within the chunk, cold (neutral pose) at the
// chunk's start. Simplified vs. QuickIK's overlap-stitched segmented solve
// (see README.md's "notes" caveat) -- just plain contiguous chunking, since
// KDL has no built-in parallel solve to mirror.
constexpr size_t kNThreads = 8;
constexpr size_t kSegmentLen = 200;  // matches perf.rs's/bench_cpp.cpp's per-segment frame count
constexpr size_t kTiledLen = kSegmentLen * kNThreads;

std::vector<Frames> tiled_sequence(const KdlModel &model, const Json &fixtures, size_t length) {
  std::vector<Frames> base;
  for (auto &f : fixtures["native_rate_frames"].as_array())
    base.push_back(build_target_frames(model, to_vec3s(f["target_ego"])));
  std::vector<Frames> out;
  out.reserve(length);
  for (size_t i = 0; i < length; i++) out.push_back(base[i % base.size()]);
  return out;
}

double run_multithread_once(const BodyPlan &plan, const std::vector<double> &neutral_angles,
                             const std::vector<Frames> &sequence) {
  size_t total = sequence.size();
  size_t base_chunk = total / kNThreads;
  size_t rem = total % kNThreads;

  std::vector<std::thread> threads;
  auto t0 = Clock::now();
  size_t start = 0;
  for (size_t t = 0; t < kNThreads; t++) {
    size_t len = base_chunk + (t < rem ? 1 : 0);
    size_t end = start + len;
    threads.emplace_back([&plan, &neutral_angles, &sequence, start, end] {
      SolverBundle sb(plan);
      JntArray q = sb.neutral(neutral_angles);
      // max_iter=10: see bench_sequence's own comment on kThroughputMaxIter.
      for (size_t i = start; i < end; i++) q = solve(sb, q, sequence[i], 10);
    });
    start = end;
  }
  for (auto &th : threads) th.join();
  return elapsed_us(t0) / 1e6;  // seconds
}

void write_results_json(const std::string &body, double single_frame_latency_us, double single_frame_latency_max_us,
                         double single_thread_throughput_fps, double multi_thread_throughput_fps) {
  std::filesystem::path out_dir = std::filesystem::path(__FILE__).parent_path() / "../../plot/results";
  std::filesystem::create_directories(out_dir);
  std::ofstream out(out_dir / ("kdl-" + body + ".json"));
  std::string notes =
      "Orocos KDL has no native floating-base joint or "
      "position-only-IK task; the free-floating root is modeled as 6 "
      "scalar joints in series (TransX/Y/Z, RotZ/Y/X -- a sequential "
      "Euler-like parametrization, not QuickIK's singularity-free "
      "quaternion), and position-only fitting is emulated by zeroing the "
      "3 rotational rows of TreeIkSolverVel_wdls's task-space weight "
      "matrix per endpoint. Uses a hand-written outer loop (not KDL's own "
      "TreeIkSolverPos_NR_JL, whose only early-stop -- a combined "
      "residual norm over all endpoints -- never triggers on real, "
      "imperfectly-fittable mocap data, so it silently always burns the "
      "full iteration cap) adding a quickik-tolerance-matched step-size "
      "early-stop instead, cutting mean iteration count from the full cap "
      "to ~7 on the fly body with no change in residual accuracy. KDL is "
      "still far slower than quickik/RBDL on this workload, though -- its "
      "per-iteration cost (a dense SVD over the task-space Jacobian) is "
      "inherently much higher than RBDL's/quickik's normal-equations solve, "
      "a genuine algorithmic difference, not a tuning artifact. "
      "multi_thread_throughput_fps "
      "uses simple contiguous chunking (8 independent, "
      "internally-warm-started, externally-cold-started chunks), not "
      "QuickIK's overlap-stitched segmented solve, since KDL has no "
      "parallel solve path to mirror. Both single_frame_latency_us and "
      "single_frame_latency_max_us force max_iter=10 (not this file's own "
      "kMaxIter=100 ceiling, which only bounds the warm-started throughput "
      "sequence below) so both are comparable to quickik/RBDL/Pinocchio's, "
      "which also cap at 10; only step_tol differs (kStepTol vs. 0, i.e. "
      "early stopping allowed vs. forced to the full 10 iterations).";
  out << "{\n"
      << "  \"name\": \"kdl\",\n"
      << "  \"body\": \"" << body << "\",\n"
      << "  \"language\": \"cpp\",\n"
      << "  \"formulation\": \"whole-tree\",\n"
      << "  \"single_frame_latency_us\": " << single_frame_latency_us << ",\n"
      << "  \"single_frame_latency_max_us\": " << single_frame_latency_max_us << ",\n"
      << "  \"single_thread_throughput_fps\": " << single_thread_throughput_fps << ",\n"
      << "  \"multi_thread_throughput_fps\": " << multi_thread_throughput_fps << ",\n"
      << "  \"multi_thread_n_threads\": " << kNThreads << ",\n"
      << "  \"notes\": \"" << notes << "\"\n"
      << "}\n";
}

// One body to benchmark: its name (used for the output filename and JSON
// "body" field) and its matching body-plan/fixtures files under assets_dir.
struct BodyConfig {
  const char *name;
  const char *body_plan;
  const char *fixtures;
};

constexpr BodyConfig kBodies[] = {
    {"neuromechfly", "neuromechfly_ypr_legs.json", "fixtures.json"},
    {"g1", "g1_body_plan.json", "fixtures_g1.json"},
};

void run_body(const BodyConfig &body, const std::filesystem::path &assets_dir) {
  std::printf("\n########## body: %s ##########\n\n", body.name);

  BodyPlan plan = load_body_plan((assets_dir / body.body_plan).string());
  std::vector<double> neutral_angles = load_neutral_angles((assets_dir / body.body_plan).string());
  Json fixtures = parse_json_file((assets_dir / body.fixtures).string());

  SolverBundle sb(plan);
  std::printf("KDL tree: %u segments, %u dofs (%zu floating-base + %zu real)\n\n", sb.model.tree.getNrOfSegments(),
              sb.model.tree.getNrOfJoints(), kFloatingBaseDofs, sb.model.n_dofs);

  JntArray q_neutral = sb.neutral(neutral_angles);

  // Same fixture-derived target used by the Rust/Python/C++ benchmarks.
  auto target = to_vec3s(fixtures["synthetic_frames"][0]["target_ego"]);
  Frames target_frames = build_target_frames(sb.model, target);

  // max_iter is forced to 10 here (not this file's own kMaxIter=100) to
  // match QuickIK/RBDL/Pinocchio's shared iteration cap -- early stopping
  // (step_tol=kStepTol, the default) still applies within that budget.
  std::printf("-- single-frame time (latency) --\n");
  double single_frame_latency_us =
      summarize("CartToJnt() (cold)",
                 bench_single_frame_latency(sb, q_neutral, target_frames, kLatencyNCalls, 500, /*max_iter=*/10));

  // step_tol=0 additionally disables early stopping, forcing every solve to
  // run the full 10 iterations -- the worst case if a frame never converges
  // early.
  std::printf("\n-- single-frame time (latency), early stop disabled (10 iterations) --\n");
  double single_frame_latency_max_us =
      summarize("CartToJnt() (forced max iterations)",
                 bench_single_frame_latency(sb, q_neutral, target_frames, kLatencyNCalls, 500, /*max_iter=*/10,
                                             /*step_tol=*/0.0));

  std::printf("\n-- single-thread sequence throughput (native-rate frames, warm-started) --\n");
  std::vector<Frames> native_frames = tiled_sequence(sb.model, fixtures, kSingleThreadNFrames);
  double single_thread_mean_us = summarize("CartToJnt() (warm)", bench_sequence(sb, q_neutral, native_frames));

  std::printf("\n-- multi-thread sequence throughput (%zu contiguous chunks, %zu threads) --\n", kNThreads,
              kNThreads);
  std::vector<Frames> sequence = tiled_sequence(sb.model, fixtures, kTiledLen);
  run_multithread_once(plan, neutral_angles, sequence);  // warmup
  double elapsed_s = run_multithread_once(plan, neutral_angles, sequence);
  double multithread_fps = sequence.size() / elapsed_s;
  std::printf("n_frames=%-6zu elapsed=%9.3fs  throughput=%10.1f frames/s\n", sequence.size(), elapsed_s,
              multithread_fps);

  write_results_json(body.name, single_frame_latency_us, single_frame_latency_max_us, 1e6 / single_thread_mean_us,
                      multithread_fps);
  std::printf("\nWrote ../../plot/results/kdl-%s.json\n", body.name);
}

}  // namespace

int main() {
  const std::filesystem::path assets_dir = std::filesystem::path(__FILE__).parent_path() / "../../assets";
  for (const auto &body : kBodies) run_body(body, assets_dir);
  return 0;
}
