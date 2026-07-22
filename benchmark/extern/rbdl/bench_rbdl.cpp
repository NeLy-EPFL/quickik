// Throughput/latency benchmark for RBDL (Rigid Body Dynamics Library),
// mirroring ../../fastik_rust/src/perf.rs, ../../fastik_cpp/bench_cpp.cpp,
// and ../kdl/bench_kdl.cpp so all are directly comparable. See ../../fastik_cpp/
// for the shared, dependency-free json.hpp / forward_kinematics.hpp copied
// into this directory.
//
// -----------------------------------------------------------------------
// Modeling notes and a real RBDL limitation found along the way:
//
// 1. Whole-tree, whole-keypoint-set solve: unlike KDL/TRAC-IK/FABRIK (which
//    can only fit one chain endpoint per solve call, or need explicit
//    per-endpoint task-space hacks), RBDL's `InverseKinematicsConstraintSet`
//    natively takes an arbitrary list of point constraints solved jointly in
//    one linear system, so -- like fastik -- this benchmark fits all 30 leg
//    keypoints (every coxa/femur/tibia/claw, not just the 6 claws)
//    simultaneously against the floating thorax root, in a single Model.
//
// 2. Floating base: the investigation notes suggested RBDL's native
//    `JointTypeFloatingBase` (translation + quaternion) "should just work"
//    for the free-floating thorax root. It does not, for this solver: RBDL
//    stores a spherical/floating-base joint's quaternion with its w
//    component *appended* at the very end of the Q vector (see
//    rbdl/Model.h's `multdof3_w_index`), so for such a model
//    `model.q_size != model.qdot_size` (49 vs 48 here). But
//    `InverseKinematicsConstraintSet`'s internal Newton step
//    (src/Kinematics.cc, the `Wn`/`delta_theta` block) sizes its damping
//    matrix and `Qres += delta_theta` update using `Qres.size()` (== q_size)
//    while the Jacobian-derived `delta_theta` is sized `qdot_size` --
//    confirmed by a minimal repro that segfaults immediately on the very
//    first `InverseKinematics()` call whenever a `JointTypeFloatingBase` (or
//    bare `JointTypeSpherical`) joint is in the model. This looks like a
//    genuine upstream bug in RBDL's IK code for models with quaternion
//    joints, not a modeling mistake on our part.
//
//    Workaround (used here, and functionally equivalent to what the KDL
//    benchmark already does for the same reason): the thorax root is
//    `JointTypeTranslationXYZ` + `JointTypeEulerZYX` in series -- a
//    non-quaternion 6-DOF floating base where q_size == qdot_size
//    everywhere, which sidesteps the bug entirely. Same reachable pose
//    space as a true floating base, modulo gimbal lock; not fastik's
//    singularity-free quaternion root.
//
// 3. Every joint in the JSON body plan is a keypoint. Multi-dof joints are
//    expanded into chains of 1-dof `JointTypeRevolute` bodies exactly as in
//    ../leg_poc.cpp: the *first* dof in the chain carries the joint's
//    offset translation, later dofs use a zero-offset frame. A joint's own
//    keypoint is the local origin (0,0,0) of the *first* body in its chain
//    -- invariant to that joint's own rotation, matching fastik's
//    convention that a joint's own dofs re-orient only its children. Leaf
//    (0-dof) joints (the claws) are not separate RBDL bodies at all: they
//    are a body-point offset off their parent's last chain body.
//
// 4. Solver algorithm: `InverseKinematicsConstraintSet` is NOT the
//    transpose/damped-least-squares method the free `InverseKinematics()`
//    function uses (whose docstring warns accuracy may only reach ~1e-2) --
//    reading src/Kinematics.cc shows it instead solves the *joint-space*
//    Levenberg-Marquardt normal equations `(J^T J + Wn) delta_q = J^T e`,
//    much closer in spirit to fastik's own Gauss-Newton solve. `step_tol`
//    gates two early-stop checks: `||e||_2 < step_tol` (L2 norm of the
//    stacked position-residual vector, checked before the step) and
//    `||delta_theta||_2 < step_tol` (L2 norm of the joint-space update,
//    checked after) -- both whole-vector L2 norms, unlike fastik's
//    per-component max-abs-value check, but comparable in spirit (both are
//    "stop once the update step is negligible" criteria).
//
//    Tuning: `lambda=1e-6` (unaffected by the below -- see README.md's
//    sweep), `max_steps=10`, `step_tol=1e-3` -- literally fastik's own
//    `n_iterations`/`position_tolerance`/`angle_tolerance` defaults, not
//    independently chosen. An earlier version of this benchmark used
//    `max_steps=300, step_tol=1e-10` (RBDL's own tighter-than-necessary
//    defaults): a step_tol/max_steps sweep (see README.md) found this
//    produced IDENTICAL residual accuracy (mean rms 0.0759 either way) while
//    being ~5.4x slower (single-thread throughput 535 fps vs. 2,396 fps) --
//    the gap was RBDL grinding out iterations chasing precision the real,
//    imperfectly-fittable mocap data can never actually reach, not real
//    solver work. Matching fastik's own tolerance/cap avoids that waste.

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

#include <rbdl/rbdl.h>

#include "forward_kinematics.hpp"
#include "json.hpp"

namespace rbdl = RigidBodyDynamics;
namespace rmath = RigidBodyDynamics::Math;

namespace {

using Clock = std::chrono::steady_clock;

// Damped Levenberg-Marquardt tuning for InverseKinematicsConstraintSet --
// literally fastik's own SolverConfig::default() values (n_iterations,
// position_tolerance/angle_tolerance); see the file header comment and
// README.md for why, and for the sweep showing this loses zero accuracy
// vs. RBDL's own (much slower) defaults.
constexpr double kLambda = 1e-6;
constexpr unsigned int kMaxSteps = 10;
constexpr double kStepTol = 1e-3;

// A built RBDL Model for the neuromechfly_ypr_legs body plan, plus the
// bookkeeping needed to (a) address each non-root joint as an IK target/FK
// query point and (b) map fastik's flat dof_offset numbering to RBDL q
// indices.
struct RbdlModel {
  rbdl::Model model;
  // keypoint_body[i]/keypoint_point[i]: body id + local point representing
  // joint i's own (pre-own-rotation) world position, for i in
  // [1, joints.size()) -- index 0 (thorax root) is unused (no keypoint).
  std::vector<unsigned int> keypoint_body;
  std::vector<rmath::Vector3d> keypoint_point;
  // dof_q_index[d]: the RBDL q (== qdot, see file header note 2) index for
  // fastik's flat dof `d`, in JSON dof order.
  std::vector<unsigned int> dof_q_index;
};

RbdlModel build_model(const BodyPlan &plan) {
  RbdlModel m;
  m.model.gravity = rmath::Vector3d(0., 0., 0.);
  rbdl::Body null_body(0., rmath::Vector3d(0., 0., 0.), rmath::Vector3d(0., 0., 0.));

  // Floating thorax root: TranslationXYZ + EulerZYX in series (see file
  // header note 2 for why not JointTypeFloatingBase).
  unsigned int trans_id =
      m.model.AddBody(0, rmath::SpatialTransform(rmath::Matrix3d::Identity(), rmath::Vector3d(0., 0., 0.)),
                       rbdl::Joint(rbdl::JointTypeTranslationXYZ), null_body);
  unsigned int thorax_id =
      m.model.AddBody(trans_id, rmath::SpatialTransform(), rbdl::Joint(rbdl::JointTypeEulerZYX), null_body);

  m.keypoint_body.resize(plan.joints.size());
  m.keypoint_point.resize(plan.joints.size());
  m.keypoint_body[0] = thorax_id;
  m.keypoint_point[0] = rmath::Vector3d(0., 0., 0.);

  // tip_body[i]: the RBDL body id to hook joint i's children onto (its own
  // final chain body, i.e. after its own dofs -- see file header note 3).
  std::vector<unsigned int> tip_body(plan.joints.size());
  tip_body[0] = thorax_id;

  for (size_t i = 1; i < plan.joints.size(); i++) {
    const auto &j = plan.joints[i];
    unsigned int hook = tip_body[j.parent];
    rmath::Vector3d offset(j.offset_pos.x, j.offset_pos.y, j.offset_pos.z);

    if (j.dofs.empty()) {
      // Leaf keypoint (a claw): no RBDL body of its own.
      m.keypoint_body[i] = hook;
      m.keypoint_point[i] = offset;
      tip_body[i] = hook;
      continue;
    }

    unsigned int b = hook;
    unsigned int first_body = 0;
    bool first = true;
    for (auto &d : j.dofs) {
      rmath::Vector3d off = first ? offset : rmath::Vector3d(0., 0., 0.);
      rmath::Vector3d axis(d.axis.x, d.axis.y, d.axis.z);
      rbdl::Body body(0., rmath::Vector3d(0., 0., 0.), rmath::Vector3d(0., 0., 0.));
      rmath::SpatialTransform frame(rmath::Matrix3d::Identity(), off);
      b = m.model.AddBody(b, frame, rbdl::Joint(rbdl::JointTypeRevolute, axis), body);
      if (first) first_body = b;
      first = false;
      m.dof_q_index.push_back(m.model.mJoints[b].q_index);
    }
    m.keypoint_body[i] = first_body;
    m.keypoint_point[i] = rmath::Vector3d(0., 0., 0.);
    tip_body[i] = b;
  }
  return m;
}

// Flat, in-fastik-dof-order neutral angles (BodyPlan itself doesn't carry
// them -- only dof axes -- so read them directly from the JSON here).
std::vector<double> load_neutral_angles(const std::string &path) {
  Json root = parse_json_file(path);
  std::vector<double> out;
  for (auto &j : root["joints"].as_array())
    for (auto &d : j["dofs"].as_array()) out.push_back(d["neutral_angle"].as_number());
  return out;
}

rmath::VectorNd neutral_q(const RbdlModel &m, const std::vector<double> &neutral_angles) {
  rmath::VectorNd q = rmath::VectorNd::Zero(m.model.q_size);
  for (size_t d = 0; d < neutral_angles.size(); d++) q[m.dof_q_index[d]] = neutral_angles[d];
  return q;
}

std::vector<Vec3> to_vec3s(const Json &target_ego) {
  std::vector<Vec3> out;
  for (auto &p : target_ego.as_array())
    out.push_back({static_cast<float>(p[0].as_number()), static_cast<float>(p[1].as_number()),
                    static_cast<float>(p[2].as_number())});
  return out;
}

// Builds an IK constraint set for one target frame: one point constraint per
// non-root joint, in joint order (matching target_ego's order 1:1), fastik's
// convention of a `Missing` root observation. `step_tol=0` disables early
// stopping, forcing every solve to run the full `max_steps`.
rbdl::InverseKinematicsConstraintSet build_cs(const RbdlModel &m, const std::vector<Vec3> &target_ego,
                                               double step_tol = kStepTol) {
  rbdl::InverseKinematicsConstraintSet cs;
  cs.lambda = kLambda;
  cs.max_steps = kMaxSteps;
  cs.step_tol = step_tol;
  cs.constraint_tol = step_tol;
  for (size_t k = 0; k < target_ego.size(); k++)
    cs.AddPointConstraint(m.keypoint_body[k + 1], m.keypoint_point[k + 1],
                           rmath::Vector3d(target_ego[k].x, target_ego[k].y, target_ego[k].z));
  return cs;
}

// Mutates an existing constraint set's targets in place (constraint
// body/point list is unchanged frame to frame) -- avoids rebuilding the
// vectors every frame in the sequence benchmarks.
void set_targets(rbdl::InverseKinematicsConstraintSet &cs, const std::vector<Vec3> &target_ego) {
  for (size_t k = 0; k < target_ego.size(); k++)
    cs.target_positions[k] = rmath::Vector3d(target_ego[k].x, target_ego[k].y, target_ego[k].z);
}

// 3D distance rms/max between a solved q's keypoints and their targets, via
// CalcBodyToBaseCoordinates -- an independent check of `cs.error_norm`.
std::pair<double, double> residual_stats(RbdlModel &m, const rmath::VectorNd &q, const std::vector<Vec3> &target) {
  double sum_sq = 0.0, max_d = 0.0;
  for (size_t k = 0; k < target.size(); k++) {
    rmath::Vector3d achieved =
        rbdl::CalcBodyToBaseCoordinates(m.model, q, m.keypoint_body[k + 1], m.keypoint_point[k + 1], false);
    rmath::Vector3d tgt(target[k].x, target[k].y, target[k].z);
    double d = (achieved - tgt).norm();
    sum_sq += d * d;
    max_d = std::max(max_d, d);
  }
  return {std::sqrt(sum_sq / target.size()), max_d};
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
std::vector<double> bench_single_frame_latency(RbdlModel &m, const rmath::VectorNd &q_neutral,
                                                const std::vector<Vec3> &target, int n_calls, int n_warmup,
                                                double step_tol = kStepTol) {
  auto cs = build_cs(m, target, step_tol);
  rmath::VectorNd q_out;
  for (int i = 0; i < n_warmup; i++) rbdl::InverseKinematics(m.model, q_neutral, cs, q_out);

  std::vector<double> samples;
  samples.reserve(n_calls);
  for (int i = 0; i < n_calls; i++) {
    auto t0 = Clock::now();
    rbdl::InverseKinematics(m.model, q_neutral, cs, q_out);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// -----------------------------------------------------------------------
// Metric 2: single_thread_throughput_fps -- warm-started sequential solve
// over the 300-frame native-rate fixture (frame i seeded from frame i-1's
// solution).
std::vector<double> bench_sequence(RbdlModel &m, const rmath::VectorNd &q_neutral,
                                    const std::vector<std::vector<Vec3>> &frames) {
  auto cs = build_cs(m, frames[0]);

  rmath::VectorNd q = q_neutral;
  rmath::VectorNd q_out;
  for (auto &target : frames) {  // untimed warmup pass
    set_targets(cs, target);
    rbdl::InverseKinematics(m.model, q, cs, q_out);
    q = q_out;
  }

  q = q_neutral;
  std::vector<double> samples;
  samples.reserve(frames.size());
  for (auto &target : frames) {
    set_targets(cs, target);
    auto t0 = Clock::now();
    rbdl::InverseKinematics(m.model, q, cs, q_out);
    samples.push_back(elapsed_us(t0));
    q = q_out;
  }
  return samples;
}

// -----------------------------------------------------------------------
// Metric 3: multi_thread_throughput_fps -- a longer tiled sequence split
// into kNThreads contiguous, roughly-equal chunks, each solved on its own
// std::thread (its own RbdlModel/Model instance, since RBDL's kinematics
// caches inside Model are not thread-safe to share): warm-started within
// the chunk, cold (neutral pose) at the chunk's start. Simplified vs.
// fastik's overlap-stitched segmented solve (see README.md's "notes"
// caveat) -- plain contiguous chunking, since RBDL has no parallel solve
// path to mirror.
constexpr size_t kNThreads = 8;
constexpr size_t kSegmentLen = 200;  // matches perf.rs's/bench_cpp.cpp's per-segment frame count
constexpr size_t kTiledLen = kSegmentLen * kNThreads;

std::vector<std::vector<Vec3>> tiled_sequence(const Json &fixtures, size_t length) {
  std::vector<std::vector<Vec3>> base;
  for (auto &f : fixtures["native_rate_frames"].as_array()) base.push_back(to_vec3s(f["target_ego"]));
  std::vector<std::vector<Vec3>> out;
  out.reserve(length);
  for (size_t i = 0; i < length; i++) out.push_back(base[i % base.size()]);
  return out;
}

double run_multithread_once(const BodyPlan &plan, const std::vector<double> &neutral_angles,
                             const std::vector<std::vector<Vec3>> &sequence) {
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
      RbdlModel m = build_model(plan);
      rmath::VectorNd q = neutral_q(m, neutral_angles);
      auto cs = build_cs(m, sequence[start]);
      rmath::VectorNd q_out;
      for (size_t i = start; i < end; i++) {
        set_targets(cs, sequence[i]);
        rbdl::InverseKinematics(m.model, q, cs, q_out);
        q = q_out;
      }
    });
    start = end;
  }
  for (auto &th : threads) th.join();
  return elapsed_us(t0) / 1e6;  // seconds
}

// -----------------------------------------------------------------------
// Correctness sanity check (quick, not the main deliverable): residual
// rms/max over all 30 keypoints, per synthetic_frames entry (cold from
// neutral) and summarized over native_rate_frames (warm-started).
void run_correctness(RbdlModel &m, const rmath::VectorNd &q_neutral, const Json &fixtures) {
  std::printf("== Synthetic exact-fit frames (cold from neutral) ==\n");
  std::printf("%6s %10s %14s %14s\n", "frame", "steps", "rms", "max");
  for (auto &frame : fixtures["synthetic_frames"].as_array()) {
    auto target = to_vec3s(frame["target_ego"]);
    auto cs = build_cs(m, target);
    rmath::VectorNd q_res;
    rbdl::InverseKinematics(m.model, q_neutral, cs, q_res);
    auto [rms, max_d] = residual_stats(m, q_res, target);
    std::printf("%6lld %10u %14.3e %14.3e\n", (long long)frame["frame"].as_number(), cs.num_steps, rms, max_d);
  }

  std::printf("\n== Real native-rate frames (warm-started, 300 frames) ==\n");
  auto cs = build_cs(m, to_vec3s(fixtures["native_rate_frames"][0]["target_ego"]));
  rmath::VectorNd q = q_neutral;
  std::vector<double> rms_all, max_all;
  for (auto &frame : fixtures["native_rate_frames"].as_array()) {
    auto target = to_vec3s(frame["target_ego"]);
    set_targets(cs, target);
    rmath::VectorNd q_res;
    rbdl::InverseKinematics(m.model, q, cs, q_res);
    auto [rms, max_d] = residual_stats(m, q_res, target);
    rms_all.push_back(rms);
    max_all.push_back(max_d);
    q = q_res;
  }
  auto mean = [](const std::vector<double> &v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
  std::printf("residual to target (model units): mean_rms=%.4e  max_rms=%.4e  max=%.4e\n", mean(rms_all),
              *std::max_element(rms_all.begin(), rms_all.end()), *std::max_element(max_all.begin(), max_all.end()));
  std::printf(
      "(real mocap frames don't perfectly satisfy this exact rigid rotation-axis model -- see "
      "README.md.)\n\n");
}

void write_results_json(double single_frame_latency_us, double single_frame_latency_max_us,
                         double single_thread_throughput_fps, double multi_thread_throughput_fps) {
  std::filesystem::path out_dir = std::filesystem::path(__FILE__).parent_path() / "../../plot/results";
  std::filesystem::create_directories(out_dir);
  std::ofstream out(out_dir / "rbdl.json");
  out << "{\n"
      << "  \"name\": \"rbdl\",\n"
      << "  \"language\": \"cpp\",\n"
      << "  \"formulation\": \"whole-tree\",\n"
      << "  \"single_frame_latency_us\": " << single_frame_latency_us << ",\n"
      << "  \"single_frame_latency_max_us\": " << single_frame_latency_max_us << ",\n"
      << "  \"single_thread_throughput_fps\": " << single_thread_throughput_fps << ",\n"
      << "  \"multi_thread_throughput_fps\": " << multi_thread_throughput_fps << ",\n"
      << "  \"multi_thread_n_threads\": " << kNThreads << ",\n"
      << "  \"notes\": \"RBDL's InverseKinematicsConstraintSet solves the joint-space damped "
         "Levenberg-Marquardt normal equations (J^T J + Wn) delta_q = J^T e, closer to fastik's own "
         "Gauss-Newton than to RBDL's simple transpose/DLS InverseKinematics() free function. Tuning "
         "(max_steps=10, step_tol=1e-3) is literally fastik's own n_iterations/tolerance defaults, "
         "chosen after a sweep found RBDL's own tighter defaults (max_steps=300, step_tol=1e-10) "
         "burned ~5.4x more time for byte-identical residual accuracy on this workload -- real mocap "
         "data never gets close enough to trigger the tighter tolerance, so it was pure wasted "
         "iteration, not extra precision. The free-floating thorax root uses TranslationXYZ + "
         "EulerZYX in series, not RBDL's native JointTypeFloatingBase: that quaternion joint type "
         "crashes InverseKinematicsConstraintSet in this RBDL version (its Newton step mixes up "
         "q_size vs qdot_size once q_size > qdot_size), confirmed with a minimal repro -- this looks "
         "like a genuine upstream bug, not a modeling error. All 30 leg keypoints (not just the 6 "
         "claws) are fit jointly in one solve, same as fastik's whole-tree solve. Accuracy: synthetic "
         "(exact-fit) frames converge to residual rms ~1e-4-1e-5 model units in 4-5 steps; real "
         "native_rate_frames converge to a real, tuning-independent residual floor of rms ~0.076 "
         "model units (identical to 4 significant figures across the whole step_tol sweep, 1e-10 to "
         "1e-2), since real mocap data doesn't exactly satisfy this rigid rotation-axis model. "
         "multi_thread_throughput_fps uses simple "
         "contiguous chunking (8 independent, internally-warm-started, externally-cold-started "
         "chunks, each with its own Model instance), not fastik's overlap-stitched segmented solve, "
         "since RBDL has no parallel solve path to mirror.\"\n"
      << "}\n";
}

}  // namespace

int main() {
  const std::filesystem::path assets_dir = std::filesystem::path(__FILE__).parent_path() / "../../assets";

  BodyPlan plan = load_body_plan((assets_dir / "neuromechfly_ypr_legs.json").string());
  std::vector<double> neutral_angles = load_neutral_angles((assets_dir / "neuromechfly_ypr_legs.json").string());
  Json fixtures = parse_json_file((assets_dir / "fixtures.json").string());

  RbdlModel m = build_model(plan);
  std::printf("RBDL model: %u bodies, q_size=%u dofs (%u floating-base + %zu real)\n\n",
              static_cast<unsigned int>(m.model.mBodies.size()) - 1, m.model.q_size, 6u, m.dof_q_index.size());

  rmath::VectorNd q_neutral = neutral_q(m, neutral_angles);

  run_correctness(m, q_neutral, fixtures);

  // Same fixture-derived target used by the Rust/Python/C++ benchmarks.
  auto target = to_vec3s(fixtures["synthetic_frames"][0]["target_ego"]);

  std::printf("-- single-frame time (latency) --\n");
  double single_frame_latency_us =
      summarize("InverseKinematics() (cold)", bench_single_frame_latency(m, q_neutral, target, 20000, 1000));

  // step_tol=0 disables early stopping, forcing every solve to run the full
  // max_steps -- the worst case if a frame never converges early.
  std::printf("\n-- single-frame time (latency), early stop disabled (%u steps) --\n", kMaxSteps);
  double single_frame_latency_max_us = summarize(
      "InverseKinematics() (forced max steps)", bench_single_frame_latency(m, q_neutral, target, 20000, 1000, 0.0));

  std::printf("\n-- single-thread sequence throughput (native-rate frames, warm-started) --\n");
  std::vector<std::vector<Vec3>> native_frames;
  for (auto &f : fixtures["native_rate_frames"].as_array()) native_frames.push_back(to_vec3s(f["target_ego"]));
  double single_thread_mean_us =
      summarize("InverseKinematics() (warm)", bench_sequence(m, q_neutral, native_frames));

  std::printf("\n-- multi-thread sequence throughput (%zu contiguous chunks, %zu threads) --\n", kNThreads,
              kNThreads);
  std::vector<std::vector<Vec3>> sequence = tiled_sequence(fixtures, kTiledLen);
  run_multithread_once(plan, neutral_angles, sequence);  // warmup
  double elapsed_s = run_multithread_once(plan, neutral_angles, sequence);
  double multithread_fps = sequence.size() / elapsed_s;
  std::printf("n_frames=%-6zu elapsed=%9.3fs  throughput=%10.1f frames/s\n", sequence.size(), elapsed_s,
              multithread_fps);

  write_results_json(single_frame_latency_us, single_frame_latency_max_us, 1e6 / single_thread_mean_us, multithread_fps);
  std::printf("\nWrote ../../plot/results/rbdl.json\n");
  return 0;
}
