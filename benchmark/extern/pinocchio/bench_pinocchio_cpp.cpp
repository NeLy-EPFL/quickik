// Native C++ port of bench_pinocchio.py, mirroring ../rbdl/bench_rbdl.cpp's
// exact methodology so the two are directly comparable: same Model/Frame
// construction and Gauss-Newton/LM math, but with the outer loop implemented
// in Eigen instead of numpy, so Pinocchio's own C++ speed on this workload
// can be measured without Python/numpy overhead. See
// ../../plot/results/pinocchio.json / bench_pinocchio.py's README.md section
// for the Python numbers and this file's README.md section for the C++ ones.
//
// Modeling notes (identical to bench_pinocchio.py -- see its README.md
// section for the full write-up):
//   - Thorax root: `pinocchio::JointModelFreeFlyer` (translation + unit
//     quaternion, nq=7/nv=6) -- unlike RBDL, Pinocchio's floating-base joint
//     works fine with its own IK-style Newton step (there's no built-in IK
//     solver to trip over the RBDL bug this benchmark hit).
//   - Each named JSON joint's N scalar DOFs become N chained single-DOF
//     `JointModelRX/RY/RZ` joints; mirrored-leg axes (e.g. [-1, 0, 0]) are
//     reproduced by negating that DOF's driven angle rather than baking a
//     rotation into the joint placement (R(-n, -t) = R(n, t)).
//   - All 30 leg keypoints (every coxa/femur/tibia/claw) are tracked via
//     `OP_FRAME` operational frames and fit jointly against the free-
//     floating root, in one Gauss-Newton solve, matching fastik/RBDL's
//     whole-tree formulation.
//   - Solver: position-only 3-row residuals per keypoint accumulated into
//     an nv x nv normal-equations matrix/nv-vector, a neutral-pose Tikhonov
//     prior on the 42 leg DOFs, LM diagonal damping, `colPivHouseholderQr`
//     solve, then `pinocchio::integrate` (respects the free-flyer's
//     quaternion manifold). Same fastik SolverConfig::default() tuning as
//     every other benchmark in this repo (n_iterations=10, damping=1e-6,
//     neutral_pose_weight=1e-3, position/angle_tolerance=1e-3).

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

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody.hpp"

#include "json.hpp"

namespace pin = pinocchio;

namespace {

using Clock = std::chrono::steady_clock;

// Gauss-Newton/LM config -- literally fastik's own SolverConfig::default()
// values (src/solver.rs), matching every other benchmark in this repo.
constexpr int kNIterations = 10;
constexpr double kDamping = 1e-6;
constexpr double kNeutralPoseWeight = 1e-3;
constexpr double kPositionTolerance = 1e-3;
constexpr double kAngleTolerance = 1e-3;

// -----------------------------------------------------------------------
// JSON body plan -> Pinocchio Model, generalizing poc_one_leg.py /
// bench_pinocchio.py's `build_full_model` to C++. Parses the JSON directly
// (rather than reusing ../rbdl/forward_kinematics.hpp's float-based
// BodyPlan) to keep full double precision throughout, matching Python's
// float64 numpy arrays exactly.
struct DofSpec {
  Eigen::Vector3d axis;
  double neutral_angle;
};

struct JointNode {
  std::string name;
  int parent;  // -1 for the root (thorax)
  Eigen::Vector3d offset_pos;
  Eigen::Quaterniond offset_quat;
  std::vector<DofSpec> dofs;
};

std::vector<JointNode> load_joint_nodes(const std::string &path) {
  Json root = parse_json_file(path);
  std::vector<JointNode> nodes;
  for (auto &j : root["joints"].as_array()) {
    JointNode node;
    node.name = j["name"].as_string();
    node.parent = -1;
    if (!j["parent"].is_null()) {
      const std::string &parent_name = j["parent"].as_string();
      for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].name == parent_name) {
          node.parent = static_cast<int>(i);
          break;
        }
      }
    }
    auto &op = j["offset_pos"].as_array();
    node.offset_pos = Eigen::Vector3d(op[0].as_number(), op[1].as_number(), op[2].as_number());
    auto &oq = j["offset_quat"].as_array();  // (w, x, y, z)
    node.offset_quat = Eigen::Quaterniond(oq[0].as_number(), oq[1].as_number(), oq[2].as_number(), oq[3].as_number());
    for (auto &d : j["dofs"].as_array()) {
      DofSpec dof;
      auto &axis = d["axis"].as_array();
      dof.axis = Eigen::Vector3d(axis[0].as_number(), axis[1].as_number(), axis[2].as_number());
      dof.neutral_angle = d["neutral_angle"].as_number();
      node.dofs.push_back(dof);
    }
    nodes.push_back(std::move(node));
  }
  return nodes;
}

// Maps a (possibly negative) unit axis like [-1, 0, 0] to a Pinocchio
// RX/RY/RZ joint (always defined about the *positive* axis) plus a sign,
// exactly mirroring bench_pinocchio.py's `_axis_to_joint`: rotating by
// `sign * theta` about the positive axis equals rotating by `theta` about
// `sign * (positive axis)`, since R(-n, -t) = R(n, t).
pin::JointModel axis_to_joint_model(const Eigen::Vector3d &axis, double &sign_out) {
  int idx = 0;
  double best = std::abs(axis[0]);
  for (int i = 1; i < 3; i++) {
    if (std::abs(axis[i]) > best) {
      best = std::abs(axis[i]);
      idx = i;
    }
  }
  sign_out = axis[idx] >= 0.0 ? 1.0 : -1.0;
  switch (idx) {
    case 0:
      return pin::JointModel(pin::JointModelRX());
    case 1:
      return pin::JointModel(pin::JointModelRY());
    default:
      return pin::JointModel(pin::JointModelRZ());
  }
}

struct FullModel {
  pin::Model model;
  std::vector<pin::FrameIndex> keypoint_frame_ids;  // 30 leg keypoints, thorax excluded
  Eigen::VectorXd q_neutral;                        // size model.nq
  std::vector<double> dof_signs;                    // size 42, JSON dof-flatten order
};

FullModel build_full_model(const std::vector<JointNode> &nodes) {
  FullModel fm;
  pin::Model &model = fm.model;

  pin::JointIndex root_id = model.addJoint(0, pin::JointModelFreeFlyer(), pin::SE3::Identity(), nodes[0].name);
  model.appendBodyToJoint(root_id, pin::Inertia::Zero(), pin::SE3::Identity());

  std::vector<pin::JointIndex> parent_joint_id(nodes.size());
  parent_joint_id[0] = root_id;  // nodes[0] is the root itself

  std::vector<double> dof_neutral_json;

  for (size_t i = 1; i < nodes.size(); i++) {
    const JointNode &node = nodes[i];
    pin::JointIndex parent_id = parent_joint_id[static_cast<size_t>(node.parent)];
    pin::SE3 offset(node.offset_quat.toRotationMatrix(), node.offset_pos);

    if (node.dofs.empty()) {
      // Leaf keypoint with no DOFs (claw tip): fixed operational frame.
      pin::Frame frame(node.name, parent_id, 0, offset, pin::OP_FRAME);
      fm.keypoint_frame_ids.push_back(model.addFrame(frame));
      parent_joint_id[i] = parent_id;  // unused (leaves have no children)
      continue;
    }

    // One single-DOF revolute joint per scalar DOF; only the first carries
    // the translational offset from the parent keypoint, the rest are
    // collocated (identity placement) -- same convention as bench_rbdl.cpp.
    pin::JointIndex current_parent = parent_id;
    pin::SE3 placement = offset;
    pin::JointIndex last_joint_id = parent_id;
    for (size_t d = 0; d < node.dofs.size(); d++) {
      double sign;
      pin::JointModel joint_model = axis_to_joint_model(node.dofs[d].axis, sign);
      std::string dof_name = node.name + "_dof" + std::to_string(d);
      pin::JointIndex joint_id = model.addJoint(current_parent, joint_model, placement, dof_name);
      model.appendBodyToJoint(joint_id, pin::Inertia::Zero(), pin::SE3::Identity());
      fm.dof_signs.push_back(sign);
      dof_neutral_json.push_back(node.dofs[d].neutral_angle);
      current_parent = joint_id;
      placement = pin::SE3::Identity();
      last_joint_id = joint_id;
    }
    parent_joint_id[i] = last_joint_id;

    // Operational frame at this node's own keypoint (tip of its DOF chain).
    pin::Frame frame(node.name, last_joint_id, 0, pin::SE3::Identity(), pin::OP_FRAME);
    fm.keypoint_frame_ids.push_back(model.addFrame(frame));
  }

  fm.q_neutral = pin::neutral(model);
  for (size_t k = 0; k < fm.dof_signs.size(); k++) {
    fm.q_neutral[7 + static_cast<long>(k)] = fm.dof_signs[k] * dof_neutral_json[k];
  }
  return fm;
}

// -----------------------------------------------------------------------
// Fixtures (assets/fixtures.json): target_ego arrays, one per synthetic /
// native-rate frame, 30 keypoints each -- same order as keypoint_frame_ids
// (both derived from the same JSON joint order).
std::vector<Eigen::Vector3d> to_targets(const Json &target_ego) {
  std::vector<Eigen::Vector3d> out;
  for (auto &p : target_ego.as_array()) out.emplace_back(p[0].as_number(), p[1].as_number(), p[2].as_number());
  return out;
}

// -----------------------------------------------------------------------
// Gauss-Newton/LM inverse kinematics, matching bench_pinocchio.py's
// `solve_ik` line for line (position-only residuals, neutral-pose prior on
// leg DOFs, LM diagonal damping, pinocchio::integrate for the update).
// `Scratch` holds reusable Eigen buffers so a solve does no heap allocation
// beyond what Pinocchio's own algorithms need -- this is what a real C++
// implementation would do, unlike the Python version's per-iteration
// np.zeros() allocations.
struct Scratch {
  Eigen::MatrixXd jtj;
  Eigen::VectorXd jtr;
  Eigen::MatrixXd J;  // 6 x nv

  explicit Scratch(int nv) : jtj(nv, nv), jtr(nv), J(6, nv) {}
};

Eigen::VectorXd solve_ik(const pin::Model &model, pin::Data &data, const std::vector<pin::FrameIndex> &keypoint_frame_ids,
                          const std::vector<Eigen::Vector3d> &target, const Eigen::VectorXd &q0,
                          const Eigen::VectorXd &neutral_q, Scratch &s, bool disable_early_stop = false) {
  Eigen::VectorXd q = q0;
  const int nv = model.nv;

  for (int iter = 0; iter < kNIterations; iter++) {
    pin::computeJointJacobians(model, data, q);
    pin::updateFramePlacements(model, data);

    s.jtj.setZero();
    s.jtr.setZero();
    for (size_t k = 0; k < keypoint_frame_ids.size(); k++) {
      pin::FrameIndex fid = keypoint_frame_ids[k];
      Eigen::Vector3d residual = target[k] - data.oMf[fid].translation();
      s.J.setZero();
      // This per-frame extraction dominates the loop's cost (see README.md's
      // "Native C++ benchmark" section) -- more than the rest of the
      // iteration (Eigen bookkeeping + QR solve) combined.
      pin::getFrameJacobian(model, data, fid, pin::LOCAL_WORLD_ALIGNED, s.J);
      auto jac_pos = s.J.topRows<3>();
      s.jtj.noalias() += jac_pos.transpose() * jac_pos;
      s.jtr.noalias() += jac_pos.transpose() * residual;
    }

    // Neutral-pose Tikhonov prior on the leg DOFs only (v-indices 6..nv-1,
    // aligned to q-indices 7..nq-1 since every leg DOF is a 1-dof revolute
    // joint chained directly after the 6-dof free-flyer root).
    for (int i = 6; i < nv; i++) {
      s.jtj(i, i) += kNeutralPoseWeight;
      s.jtr(i) += kNeutralPoseWeight * (neutral_q[i + 1] - q[i + 1]);
    }

    // Levenberg-Marquardt relative damping on the full diagonal.
    for (int i = 0; i < nv; i++) s.jtj(i, i) += kDamping * std::max(s.jtj(i, i), 1.0);

    Eigen::VectorXd delta = s.jtj.colPivHouseholderQr().solve(s.jtr);
    q = pin::integrate(model, q, delta);

    if (disable_early_stop) continue;
    double max_pos = delta.head<3>().cwiseAbs().maxCoeff();
    double max_ang = delta.tail(nv - 3).cwiseAbs().maxCoeff();
    if (max_pos <= kPositionTolerance && max_ang <= kAngleTolerance) break;
  }
  return q;
}

// -----------------------------------------------------------------------
// Correctness sanity check (quick, not the main deliverable): residual
// rms/max over all 30 keypoints for each synthetic_frames entry, solved
// cold from neutral -- mirrors bench_rbdl.cpp's run_correctness.
void run_correctness(const pin::Model &model, pin::Data &data, const std::vector<pin::FrameIndex> &keypoint_frame_ids,
                      const Eigen::VectorXd &q_neutral, Scratch &scratch, const Json &fixtures) {
  std::printf("== Synthetic exact-fit frames (cold from neutral) ==\n");
  std::printf("%6s %14s %14s\n", "frame", "rms", "max");
  for (auto &frame : fixtures["synthetic_frames"].as_array()) {
    auto target = to_targets(frame["target_ego"]);
    Eigen::VectorXd q = solve_ik(model, data, keypoint_frame_ids, target, q_neutral, q_neutral, scratch);
    pin::forwardKinematics(model, data, q);
    pin::updateFramePlacements(model, data);

    double sum_sq = 0.0, max_d = 0.0;
    for (size_t k = 0; k < keypoint_frame_ids.size(); k++) {
      double d = (data.oMf[keypoint_frame_ids[k]].translation() - target[k]).norm();
      sum_sq += d * d;
      max_d = std::max(max_d, d);
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(keypoint_frame_ids.size()));
    std::printf("%6lld %14.3e %14.3e\n", (long long)frame["frame"].as_number(), rms, max_d);
  }
  std::printf("(rms/max: 3D distance to target, model units.)\n\n");
}

// -----------------------------------------------------------------------
double elapsed_us(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }

double summarize(const std::string &label, std::vector<double> samples_us) {
  std::sort(samples_us.begin(), samples_us.end());
  size_t n = samples_us.size();
  double mean = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / static_cast<double>(n);
  auto pct = [&](double p) { return samples_us[static_cast<size_t>(std::round((static_cast<double>(n) - 1) * p))]; };
  std::printf(
      "%-42s n=%-7zu mean=%9.3fus  median=%9.3fus  p95=%9.3fus  p99=%9.3fus  min=%9.3fus  "
      "max=%9.3fus  throughput=%10.1f calls/s\n",
      label.c_str(), n, mean, pct(0.50), pct(0.95), pct(0.99), samples_us.front(), samples_us.back(), 1e6 / mean);
  return mean;
}

// -----------------------------------------------------------------------
// Metric 1: single_frame_latency_us -- fresh neutral-pose solve against the
// fixed synthetic_frames[0] target every call, no warm start.
std::vector<double> bench_single_frame_latency(const pin::Model &model, pin::Data &data,
                                                const std::vector<pin::FrameIndex> &keypoint_frame_ids,
                                                const Eigen::VectorXd &q_neutral,
                                                const std::vector<Eigen::Vector3d> &target, Scratch &scratch,
                                                int n_calls, int n_warmup, bool disable_early_stop = false) {
  for (int i = 0; i < n_warmup; i++)
    solve_ik(model, data, keypoint_frame_ids, target, q_neutral, q_neutral, scratch, disable_early_stop);

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(n_calls));
  for (int i = 0; i < n_calls; i++) {
    auto t0 = Clock::now();
    solve_ik(model, data, keypoint_frame_ids, target, q_neutral, q_neutral, scratch, disable_early_stop);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// -----------------------------------------------------------------------
// Metric 2: single_thread_throughput_fps -- warm-started sequential solve
// over the 300-frame native-rate fixture (frame i seeded from frame i-1's
// solution).
std::vector<double> bench_sequence(const pin::Model &model, pin::Data &data,
                                    const std::vector<pin::FrameIndex> &keypoint_frame_ids,
                                    const Eigen::VectorXd &q_neutral,
                                    const std::vector<std::vector<Eigen::Vector3d>> &frames, Scratch &scratch) {
  Eigen::VectorXd q = q_neutral;
  for (auto &target : frames) q = solve_ik(model, data, keypoint_frame_ids, target, q, q_neutral, scratch);  // warmup

  q = q_neutral;
  std::vector<double> samples;
  samples.reserve(frames.size());
  for (auto &target : frames) {
    auto t0 = Clock::now();
    q = solve_ik(model, data, keypoint_frame_ids, target, q, q_neutral, scratch);
    samples.push_back(elapsed_us(t0));
  }
  return samples;
}

// -----------------------------------------------------------------------
// Metric 3: multi_thread_throughput_fps -- a longer tiled sequence split
// into kNThreads contiguous, roughly-equal chunks, each solved on its own
// std::thread: warm-started within the chunk, cold (neutral pose) at the
// chunk's start. The Model is built once and shared read-only across
// threads; each thread gets its own Data (mutable scratch state) and its
// own Scratch buffers. Mirrors bench_rbdl.cpp's contiguous-chunking scheme.
constexpr size_t kNThreads = 8;
constexpr size_t kSegmentLen = 200;  // matches bench_rbdl.cpp's per-segment frame count
constexpr size_t kTiledLen = kSegmentLen * kNThreads;

std::vector<std::vector<Eigen::Vector3d>> tiled_sequence(const Json &fixtures, size_t length) {
  std::vector<std::vector<Eigen::Vector3d>> base;
  for (auto &f : fixtures["native_rate_frames"].as_array()) base.push_back(to_targets(f["target_ego"]));
  std::vector<std::vector<Eigen::Vector3d>> out;
  out.reserve(length);
  for (size_t i = 0; i < length; i++) out.push_back(base[i % base.size()]);
  return out;
}

double run_multithread_once(const pin::Model &model, const std::vector<pin::FrameIndex> &keypoint_frame_ids,
                             const Eigen::VectorXd &q_neutral,
                             const std::vector<std::vector<Eigen::Vector3d>> &sequence) {
  size_t total = sequence.size();
  size_t base_chunk = total / kNThreads;
  size_t rem = total % kNThreads;

  std::vector<std::thread> threads;
  auto t0 = Clock::now();
  size_t start = 0;
  for (size_t t = 0; t < kNThreads; t++) {
    size_t len = base_chunk + (t < rem ? 1 : 0);
    size_t end = start + len;
    threads.emplace_back([&model, &keypoint_frame_ids, &q_neutral, &sequence, start, end] {
      pin::Data data(model);
      Scratch scratch(model.nv);
      Eigen::VectorXd q = q_neutral;
      for (size_t i = start; i < end; i++) q = solve_ik(model, data, keypoint_frame_ids, sequence[i], q, q_neutral, scratch);
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
  std::ofstream out(out_dir / ("pinocchio-cpp-" + body + ".json"));
  out << "{\n"
      << "  \"name\": \"pinocchio-cpp\",\n"
      << "  \"body\": \"" << body << "\",\n"
      << "  \"language\": \"cpp\",\n"
      << "  \"formulation\": \"whole-tree\",\n"
      << "  \"single_frame_latency_us\": " << single_frame_latency_us << ",\n"
      << "  \"single_frame_latency_max_us\": " << single_frame_latency_max_us << ",\n"
      << "  \"single_thread_throughput_fps\": " << single_thread_throughput_fps << ",\n"
      << "  \"multi_thread_throughput_fps\": " << multi_thread_throughput_fps << ",\n"
      << "  \"multi_thread_n_threads\": " << kNThreads << ",\n"
      << "  \"notes\": \"Native C++ re-implementation of pinocchio.json's benchmark: same model "
         "(thorax JointModelFreeFlyer + 6 legs x 7 DOFs, all 30 leg keypoints fit jointly via "
         "OP_FRAME operational frames) and same fastik SolverConfig::default() tuning as "
         "pinocchio.json and rbdl.json, but with the Gauss-Newton outer loop (building the "
         "normal-equations matrix, the linear solve, LM damping, the neutral-pose prior, "
         "pin.integrate) done in Eigen instead of Python/numpy, to measure Pinocchio's own C++ speed "
         "without Python overhead. Still slower than RBDL's native C++ numbers on this workload -- "
         "most of the gap is Pinocchio's per-frame getFrameJacobian API cost (30 separate extractions "
         "per iteration, one per tracked keypoint), not a Python-vs-C++ artifact; see README.md's "
         "Native C++ benchmark section. multi_thread_throughput_fps uses the same simple "
         "contiguous-chunking scheme as bench_rbdl.cpp (8 std::thread workers, each with its own "
         "Data, warm-started within its chunk and cold at the chunk's start, sharing one read-only "
         "Model), not an in-process thread pool -- comparable in spirit to rbdl.json's multi-thread "
         "metric, unlike pinocchio.json's multiprocessing-based one.\"\n"
      << "}\n";
}

// One body to benchmark: its body plan and matching fixtures file.
struct BodyConfig {
  const char *name;
  const char *body_plan;
  const char *fixtures;
};

constexpr BodyConfig kBodies[] = {
    {"neuromechfly", "neuromechfly_ypr_legs.json", "fixtures.json"},
    {"g1", "g1_body_plan.json", "fixtures_g1.json"},
};

}  // namespace

int main() {
  const std::filesystem::path assets_dir = std::filesystem::path(__FILE__).parent_path() / "../../assets";

  for (const auto &body : kBodies) {
    std::printf("\n########## body: %s ##########\n\n", body.name);

    std::vector<JointNode> nodes = load_joint_nodes((assets_dir / body.body_plan).string());
    Json fixtures = parse_json_file((assets_dir / body.fixtures).string());

    FullModel fm = build_full_model(nodes);
    pin::Data data(fm.model);
    Scratch scratch(fm.model.nv);

    std::printf("Pinocchio C++ benchmark (nq=%d, nv=%d)\n\n", fm.model.nq, fm.model.nv);

    run_correctness(fm.model, data, fm.keypoint_frame_ids, fm.q_neutral, scratch, fixtures);

    // Same fixture-derived target used by the Rust/Python/C++ fastik benchmarks and RBDL.
    auto target = to_targets(fixtures["synthetic_frames"][0]["target_ego"]);

    std::printf("-- single-frame time (latency), no warm start --\n");
    double single_frame_latency_us = summarize(
        "solve_ik() (cold)", bench_single_frame_latency(fm.model, data, fm.keypoint_frame_ids, fm.q_neutral, target,
                                                          scratch, 20000, 1000));

    // Early stop disabled, so every call runs the full kNIterations -- the
    // worst case if a frame never converges early.
    std::printf("\n-- single-frame time (latency), early stop disabled (%d iterations) --\n", kNIterations);
    double single_frame_latency_max_us = summarize(
        "solve_ik() (forced max iterations)",
        bench_single_frame_latency(fm.model, data, fm.keypoint_frame_ids, fm.q_neutral, target, scratch, 20000, 1000,
                                    /*disable_early_stop=*/true));

    std::printf("\n-- single-thread sequence throughput (native-rate frames, warm-started) --\n");
    std::vector<std::vector<Eigen::Vector3d>> native_frames;
    for (auto &f : fixtures["native_rate_frames"].as_array()) native_frames.push_back(to_targets(f["target_ego"]));
    double single_thread_mean_us =
        summarize("solve_ik() (warm)", bench_sequence(fm.model, data, fm.keypoint_frame_ids, fm.q_neutral, native_frames, scratch));

    std::printf("\n-- multi-thread sequence throughput (%zu contiguous chunks, %zu threads) --\n", kNThreads, kNThreads);
    std::vector<std::vector<Eigen::Vector3d>> sequence = tiled_sequence(fixtures, kTiledLen);
    run_multithread_once(fm.model, fm.keypoint_frame_ids, fm.q_neutral, sequence);  // warmup
    double elapsed_s = run_multithread_once(fm.model, fm.keypoint_frame_ids, fm.q_neutral, sequence);
    double multithread_fps = static_cast<double>(sequence.size()) / elapsed_s;
    std::printf("n_frames=%-6zu elapsed=%9.3fs  throughput=%10.1f frames/s\n", sequence.size(), elapsed_s, multithread_fps);

    write_results_json(body.name, single_frame_latency_us, single_frame_latency_max_us, 1e6 / single_thread_mean_us,
                        multithread_fps);
    std::printf("\nWrote ../../plot/results/pinocchio-cpp-%s.json\n", body.name);
  }
  return 0;
}
