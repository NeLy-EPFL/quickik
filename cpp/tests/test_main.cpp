// Tests for the QuickIK C++ bindings, mirroring tests/solver_test.rs,
// tests/sequential_solver_test.rs, and tests/batched_solver_test.rs. Uses the
// same "two-joint chain" fixture as those (see tests/common/mod.rs): a root,
// joint1 and joint2 (each with one Z-axis DOF, joint2 limited to
// [-0.5, 0.5]), and a trailing fixed tip.
//
// Forward kinematics isn't exposed to C++ (same as Python; see
// benchmark/scripts/bench_python.py's own from-scratch FK replica), so
// `two_link_positions` below computes the four keypoints' world positions
// directly from the chain's known geometry, in `[root, joint1, joint2, tip]`
// order, to build observations for a target pose.
//
// No external test framework: each `test_*` function returns true on success,
// printing a message and returning false on the first failed CHECK.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "quickik.h"

namespace {

// Defaults matching the old `SolverConfig::default()`.
constexpr size_t kNIterations = 10;
constexpr float kNeutralWeight = 1e-3f;
constexpr float kPositionTolerance = 1e-3f;
constexpr float kAngleTolerance = 1e-3f;
constexpr float kDamping = 1e-6f;

const char *kTwoJointChainJson = R"JSON(
{
    "joints": [
        {"name": "root", "parent": null, "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []},
        {"name": "joint1", "parent": "root", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": null}]},
        {"name": "joint2", "parent": "joint1", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": [-0.5, 0.5]}]},
        {"name": "tip", "parent": "joint2", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []}
    ]
}
)JSON";

const char *kFixedBaseTwoJointChainJson = R"JSON(
{
    "fixed_base": true,
    "joints": [
        {"name": "root", "parent": null, "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []},
        {"name": "joint1", "parent": "root", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": null}]},
        {"name": "joint2", "parent": "joint1", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": [-0.5, 0.5]}]},
        {"name": "tip", "parent": "joint2", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []}
    ]
}
)JSON";

rust::Box<quickik::KinematicTree> two_joint_chain() {
  return quickik::kinematic_tree_from_json_str(kTwoJointChainJson);
}

rust::Box<quickik::KinematicTree> fixed_base_two_joint_chain() {
  return quickik::kinematic_tree_from_json_str(kFixedBaseTwoJointChainJson);
}

// Positions of [root, joint1, joint2, tip] when joint1/joint2 are at angles
// (a1, a2) about the shared Z axis. See tests/common/mod.rs's doc comment
// for why joint1's own keypoint never moves with a1.
std::array<std::array<float, 3>, 4> two_link_positions(float a1, float a2) {
  return {{
      {0.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {1.0f + std::cos(a1), std::sin(a1), 0.0f},
      {1.0f + std::cos(a1) + std::cos(a1 + a2), std::sin(a1) + std::sin(a1 + a2), 0.0f},
  }};
}

std::vector<quickik::KeypointObservation> observations_for(float a1, float a2) {
  std::vector<quickik::KeypointObservation> obs;
  for (auto &pos : two_link_positions(a1, a2)) {
    obs.push_back(quickik::keypoint_position_3d(pos, 1.0f));
  }
  return obs;
}

rust::Slice<const quickik::KeypointObservation> slice_of(const std::vector<quickik::KeypointObservation> &v) {
  return rust::Slice<const quickik::KeypointObservation>(v.data(), v.size());
}

rust::Vec<rust::String> joint_names(std::initializer_list<const char *> names) {
  rust::Vec<rust::String> out;
  for (auto *name : names) out.push_back(rust::String(name));
  return out;
}

bool test_malformed_json_throws() {
  bool ok = true;
  try {
    quickik::kinematic_tree_from_json_str("not valid json");
    std::fprintf(stderr, "  FAILED: expected an exception for malformed JSON\n");
    ok = false;
  } catch (const std::exception &) {
    // expected
  }
  return ok;
}

bool test_recovers_pose_from_3d_observations() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto observations = observations_for(0.4f, 0.3f);

  auto state = quickik::state_neutral_pose(*tree);
  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(std::abs(result.dof_angles[0] - 0.4f) < 1e-3f);
  CHECK(std::abs(result.dof_angles[1] - 0.3f) < 1e-3f);
#undef CHECK
  return ok;
}

bool test_solve_with_fk_reports_keypoint_positions_matching_recovered_pose() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto observations = observations_for(0.4f, 0.3f);

  auto state = quickik::state_neutral_pose(*tree);
  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), false, true);

  auto expected = two_link_positions(0.4f, 0.3f);
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(result.has_keypoint_pos);
  CHECK(result.keypoint_pos.size() == tree->n_joints() * 3);
  for (size_t k = 0; k < tree->n_joints(); k++) {
    for (size_t d = 0; d < 3; d++) {
      CHECK(std::abs(result.keypoint_pos[k * 3 + d] - expected[k][d]) < 1e-2f);
    }
  }
#undef CHECK
  return ok;
}

bool test_solve_without_with_fk_or_with_grad_leaves_optional_fields_empty() {
  bool ok = true;
  auto tree = two_joint_chain();
  std::vector<quickik::KeypointObservation> observations;
  for (size_t i = 0; i < tree->n_joints(); i++) observations.push_back(quickik::keypoint_missing());

  auto state = quickik::state_neutral_pose(*tree);
  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(!result.has_keypoint_pos && result.keypoint_pos.empty());
  CHECK(!result.has_jacobian && result.jacobian.empty());
  CHECK(!result.has_cholesky_l && result.cholesky_l.empty());
#undef CHECK
  return ok;
}

bool test_solve_with_grad_reports_jacobian_and_cholesky_l() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto observations = observations_for(0.4f, 0.3f);

  auto state = quickik::state_neutral_pose(*tree);
  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), true, false);

  size_t state_dim = tree->state_dim();
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(result.has_jacobian);
  CHECK(result.jacobian.size() == 3 * tree->n_joints() * state_dim);
  CHECK(result.has_cholesky_l);
  CHECK(result.cholesky_l.size() == state_dim * state_dim);
#undef CHECK
  return ok;
}

bool test_position2d_observation_on_mapperless_solver_throws() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto state = quickik::state_neutral_pose(*tree);
  std::vector<quickik::KeypointObservation> observations;
  for (size_t i = 0; i < tree->n_joints(); i++) observations.push_back(quickik::keypoint_missing());
  observations[1] = quickik::keypoint_position_2d({1.0f, 0.0f}, 1.0f);

  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  try {
    solver->solve(*state, slice_of(observations), false, false);
    std::fprintf(stderr, "  FAILED: expected an exception for a Position2D observation with no mapper set\n");
    ok = false;
  } catch (const std::exception &) {
    // expected: the underlying panic is caught and rethrown as a C++
    // exception rather than aborting the process.
  }
  return ok;
}

bool test_recovers_pose_from_xyview_observations() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto positions = two_link_positions(0.35f, -0.25f);

  std::vector<quickik::KeypointObservation> observations;
  for (auto &pos : positions) {
    observations.push_back(quickik::keypoint_position_2d({pos[0], pos[1]}, 1.0f));
  }

  auto state = quickik::state_neutral_pose(*tree);
  auto solver = quickik::new_solver(*tree, quickik::xyview_mapper(), kNIterations, 0.0f, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(std::abs(result.dof_angles[0] - 0.35f) < 1e-3f);
  CHECK(std::abs(result.dof_angles[1] - (-0.25f)) < 1e-3f);
#undef CHECK
  return ok;
}

bool test_recovers_pose_from_camera_observations() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto positions = two_link_positions(0.2f, 0.15f);

  quickik::Camera camera{};
  camera.fx = 500.0f;
  camera.fy = 500.0f;
  camera.cx = 320.0f;
  camera.cy = 240.0f;
  camera.world2cam_pos = {0.0f, 0.0f, 5.0f};
  camera.world2cam_rot_mat = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  std::vector<quickik::KeypointObservation> observations;
  for (auto &pos : positions) {
    // Pinhole projection with world2cam_rot_mat = identity: cam == world.
    float cam_z = pos[2] + camera.world2cam_pos[2];
    float u = camera.fx * pos[0] / cam_z + camera.cx;
    float v = camera.fy * pos[1] / cam_z + camera.cy;
    observations.push_back(quickik::keypoint_position_2d({u, v}, 1.0f));
  }

  auto state = quickik::state_neutral_pose(*tree);
  auto solver = quickik::new_solver(*tree, quickik::camera_mapper(camera), kNIterations, 0.0f, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(std::abs(result.dof_angles[0] - 0.2f) < 1e-3f);
  CHECK(std::abs(result.dof_angles[1] - 0.15f) < 1e-3f);
#undef CHECK
  return ok;
}

// Sanity check, not a benchmark (see benchmark/ for real numbers): XYView's
// per-keypoint sparse-accumulation path (solver.rs's Position2D branch)
// shouldn't be dramatically slower than the Position3D path it mirrors. A
// generous factor: this only needs to catch a gross regression (e.g. an
// accidental per-call allocation creeping back in), not assert precise
// parity, since single-frame timing on this tiny fixture is dominated by
// FFI call overhead common to both paths.
double mean_solve_seconds(rust::Box<quickik::KinematicTree> &tree,
                           const std::vector<quickik::KeypointObservation> &observations, quickik::Mapper mapper) {
  auto solver = quickik::new_solver(*tree, mapper, kNIterations, kNeutralWeight, kPositionTolerance, kAngleTolerance,
                                     kDamping);
  auto warm_state = quickik::state_neutral_pose(*tree);
  solver->solve(*warm_state, slice_of(observations), false, false);  // warm up

  constexpr int kNCalls = 2000;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kNCalls; i++) {
    auto state = quickik::state_neutral_pose(*tree);
    solver->solve(*state, slice_of(observations), false, false);
  }
  auto elapsed = std::chrono::steady_clock::now() - t0;
  return std::chrono::duration<double>(elapsed).count() / kNCalls;
}

bool test_xyview_latency_not_much_worse_than_3d() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto positions = two_link_positions(0.4f, 0.3f);

  auto observations_3d = observations_for(0.4f, 0.3f);
  std::vector<quickik::KeypointObservation> observations_2d;
  for (auto &pos : positions) observations_2d.push_back(quickik::keypoint_position_2d({pos[0], pos[1]}, 1.0f));

  double t_3d = mean_solve_seconds(tree, observations_3d, quickik::no_mapper());
  double t_2d = mean_solve_seconds(tree, observations_2d, quickik::xyview_mapper());

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(t_2d < t_3d * 5);
#undef CHECK
  return ok;
}

bool test_missing_observations_leave_state_at_neutral_prior() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto state = quickik::state_neutral_pose(*tree);

  std::vector<quickik::KeypointObservation> observations;
  for (size_t i = 0; i < tree->n_joints(); i++) {
    observations.push_back(quickik::keypoint_missing());
  }

  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), false, false);

  for (float angle : result.dof_angles) {
#define CHECK(cond) \
    if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
    CHECK(std::abs(angle) < 1e-6f);
#undef CHECK
  }
  return ok;
}

bool test_solver_fields_can_be_tuned_between_solve_calls() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto state = quickik::state_neutral_pose(*tree);
  std::vector<quickik::KeypointObservation> observations;
  for (size_t i = 0; i < tree->n_joints(); i++) {
    observations.push_back(quickik::keypoint_missing());
  }

  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  solver->solve(*state, slice_of(observations), false, false);

  solver->set_n_iterations(3);
  solver->solve(*state, slice_of(observations), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(solver->n_iterations() == 3);
#undef CHECK
  return ok;
}

bool test_solve_respects_joint_limits() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto state = quickik::state_neutral_pose(*tree);

  // Same unreachable target as tests/solver_test.rs's
  // solve_respects_joint_limits: joint2 would need ~1.2 rad, past its 0.5 cap.
  std::vector<quickik::KeypointObservation> observations = {
      quickik::keypoint_missing(),
      quickik::keypoint_position_3d({1.0f, 0.0f, 0.0f}, 1.0f),
      quickik::keypoint_position_3d({2.0f, 0.0f, 0.0f}, 1.0f),
      quickik::keypoint_position_3d({2.3624f, 0.9320f, 0.0f}, 1.0f),
  };

  auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                     kAngleTolerance, kDamping);
  auto result = solver->solve(*state, slice_of(observations), false, false);

  float joint2_angle = result.dof_angles[1];
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(joint2_angle >= -0.5f - 1e-6f && joint2_angle <= 0.5f + 1e-6f);
  CHECK(joint2_angle > 0.45f);
#undef CHECK
  return ok;
}

bool test_sequence_solver_warm_starts_across_separate_calls() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto target = observations_for(0.4f, 0.3f);

  auto cold = quickik::new_sequence_solver(*tree, quickik::no_mapper(), 1, 0.0f, kPositionTolerance, kAngleTolerance,
                                            kDamping);
  auto cold_results = cold->solve(slice_of(target), tree->n_joints(), false, false);
  float cold_error = std::abs(cold_results->at(0).dof_angles[0] - 0.4f);

  auto warm = quickik::new_sequence_solver(*tree, quickik::no_mapper(), 1, 0.0f, kPositionTolerance, kAngleTolerance,
                                            kDamping);
  warm->solve(slice_of(target), tree->n_joints(), false, false);
  auto warm_results = warm->solve(slice_of(target), tree->n_joints(), false, false);
  float warm_error = std::abs(warm_results->at(0).dof_angles[0] - 0.4f);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(warm_error < cold_error);
#undef CHECK
  return ok;
}

bool test_sequence_solver_solve_returns_one_result_per_frame() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto solver = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                              kPositionTolerance, kAngleTolerance, kDamping);

  std::vector<quickik::KeypointObservation> flat;
  for (auto [a1, a2] : {std::pair{0.1f, 0.05f}, std::pair{0.2f, 0.1f}, std::pair{0.3f, 0.15f}}) {
    for (auto &obs : observations_for(a1, a2)) flat.push_back(obs);
  }

  auto results = solver->solve(slice_of(flat), tree->n_joints(), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(results->len() == 3);
  auto last = results->at(2);
  CHECK(std::abs(last.dof_angles[0] - 0.3f) < 1e-2f);
  CHECK(std::abs(last.dof_angles[1] - 0.15f) < 1e-2f);
#undef CHECK
  return ok;
}

bool test_sequence_solver_solve_with_fk_matches_recovered_pose() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto solver = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f, kPositionTolerance,
                                              kAngleTolerance, kDamping);
  auto target = observations_for(0.4f, 0.3f);
  auto results = solver->solve(slice_of(target), tree->n_joints(), false, true);
  auto result = results->at(0);

  auto expected = two_link_positions(0.4f, 0.3f);
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(result.has_keypoint_pos);
  CHECK(result.keypoint_pos.size() == tree->n_joints() * 3);
  for (size_t k = 0; k < tree->n_joints(); k++) {
    for (size_t d = 0; d < 3; d++) {
      CHECK(std::abs(result.keypoint_pos[k * 3 + d] - expected[k][d]) < 1e-2f);
    }
  }
#undef CHECK
  return ok;
}

bool test_solver_result_list_at_out_of_range_throws() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto solver = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                              kPositionTolerance, kAngleTolerance, kDamping);

  auto flat = observations_for(0.1f, 0.05f);
  auto results = solver->solve(slice_of(flat), tree->n_joints(), false, false);

  try {
    results->at(results->len());
    std::fprintf(stderr, "  FAILED: expected an exception for an out-of-range SolverResultList index\n");
    ok = false;
  } catch (const std::exception &) {
    // expected: caught and rethrown as a C++ exception rather than aborting
    // the process.
  }
  return ok;
}

bool test_sequence_solver_solve_rejects_length_not_a_multiple_of_n_joints() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto solver = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                              kPositionTolerance, kAngleTolerance, kDamping);

  // One fewer than a whole frame's worth of observations.
  auto flat = observations_for(0.1f, 0.05f);
  flat.pop_back();

  try {
    solver->solve(slice_of(flat), tree->n_joints(), false, false);
    std::fprintf(stderr, "  FAILED: expected an exception for a misaligned observations length\n");
    ok = false;
  } catch (const std::exception &) {
    // expected: caught and rethrown as a C++ exception rather than aborting
    // the process (regression test for the validation running outside the
    // catch_panic boundary).
  }
  return ok;
}

std::pair<std::vector<quickik::KeypointObservation>, std::vector<std::array<float, 2>>> sine_trajectory(
    size_t n_frames) {
  std::vector<std::array<float, 2>> true_angles;
  std::vector<quickik::KeypointObservation> flat;
  for (size_t t = 0; t < n_frames; t++) {
    float a = 0.3f * std::sin(t * 0.15f);
    true_angles.push_back({a, a * 0.5f});
    for (auto &obs : observations_for(a, a * 0.5f)) flat.push_back(obs);
  }
  return {flat, true_angles};
}

bool test_solve_segments_parallel_reconstructs_smooth_trajectory() {
  bool ok = true;
  auto tree = two_joint_chain();
  const size_t n_frames = 40;
  auto [flat, true_angles] = sine_trajectory(n_frames);

  auto solver = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                              kPositionTolerance, kAngleTolerance, kDamping);
  auto results = solver->solve_segments_parallel(slice_of(flat), tree->n_joints(), 4, false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(results->len() == n_frames);
  for (size_t i = 0; i < n_frames; i++) {
    auto result = results->at(i);
    CHECK(std::abs(result.dof_angles[0] - true_angles[i][0]) < 1e-2f);
    CHECK(std::abs(result.dof_angles[1] - true_angles[i][1]) < 1e-2f);
  }
#undef CHECK
  return ok;
}

bool test_solve_segments_parallel_honors_explicit_n_workers() {
  bool ok = true;
  auto tree = two_joint_chain();
  const size_t n_frames = 40;
  auto [flat, true_angles] = sine_trajectory(n_frames);

  // n_workers=1 forces the whole sequence through a single segment,
  // exercising a different code path than the >1 case used above.
  auto solver = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                              kPositionTolerance, kAngleTolerance, kDamping);
  auto results = solver->solve_segments_parallel(slice_of(flat), tree->n_joints(), 1, false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(results->len() == n_frames);
  for (size_t i = 0; i < n_frames; i++) {
    auto result = results->at(i);
    CHECK(std::abs(result.dof_angles[0] - true_angles[i][0]) < 1e-2f);
    CHECK(std::abs(result.dof_angles[1] - true_angles[i][1]) < 1e-2f);
  }
#undef CHECK
  return ok;
}

bool test_solve_segments_parallel_rejects_zero_workers() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto [flat, true_angles] = sine_trajectory(5);
  auto solver = quickik::new_sequence_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                              kPositionTolerance, kAngleTolerance, kDamping);

  try {
    solver->solve_segments_parallel(slice_of(flat), tree->n_joints(), 0, false, false);
    std::fprintf(stderr, "  FAILED: expected an exception for n_workers = 0\n");
    ok = false;
  } catch (const std::exception &) {
    // expected
  }
  return ok;
}

bool test_batched_solver_matches_sequential_solve() {
  bool ok = true;
  auto tree = two_joint_chain();
  // A permutation of the tree's own joint order, so this actually exercises
  // name-based remapping rather than happening to pass only for the
  // identity order.
  auto keypoints_order = joint_names({"tip", "root", "joint2", "joint1"});
  const size_t order_joint_indices[] = {3, 0, 2, 1};
  std::vector<std::array<float, 2>> targets = {{0.4f, 0.3f}, {-0.2f, 0.1f}, {0.3f, -0.4f}, {0.15f, 0.25f}};

  std::vector<std::vector<float>> expected_dof_angles;
  for (auto &angles : targets) {
    auto state = quickik::state_neutral_pose(*tree);
    auto solver = quickik::new_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f, kPositionTolerance,
                                       kAngleTolerance, kDamping);
    auto result = solver->solve(*state, slice_of(observations_for(angles[0], angles[1])), false, false);
    expected_dof_angles.emplace_back(result.dof_angles.begin(), result.dof_angles.end());
  }

  std::vector<quickik::KeypointObservation> flat;
  for (auto &angles : targets) {
    auto internal_order = observations_for(angles[0], angles[1]);
    for (size_t idx : order_joint_indices) flat.push_back(internal_order[idx]);
  }

  auto batched_solver = quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f,
                                                      kPositionTolerance, kAngleTolerance, kDamping,
                                                      keypoints_order, -1);
  auto result = batched_solver->solve(slice_of(flat), tree->n_joints(), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(result.joint_angles.size() == targets.size() * tree->n_dofs());
  for (size_t i = 0; i < targets.size(); i++) {
    for (size_t d = 0; d < tree->n_dofs(); d++) {
      CHECK(std::abs(result.joint_angles[i * tree->n_dofs() + d] - expected_dof_angles[i][d]) < 1e-4f);
    }
  }
#undef CHECK
  return ok;
}

bool test_batched_solver_with_grad_reports_jacobian_and_valid() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto keypoints_order = joint_names({"root", "joint1", "joint2", "tip"});
  auto flat = observations_for(0.4f, 0.3f);

  auto batched_solver = quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, 0.0f,
                                                      kPositionTolerance, kAngleTolerance, kDamping,
                                                      keypoints_order, -1);
  auto result = batched_solver->solve(slice_of(flat), tree->n_joints(), true, false);

  size_t state_dim = tree->state_dim();
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(result.has_jacobian);
  CHECK(result.jacobian.size() == 3 * tree->n_joints() * state_dim);
  CHECK(result.has_cholesky_l);
  CHECK(result.cholesky_l.size() == state_dim * state_dim);
  CHECK(result.valid.size() == 1);
  CHECK(result.valid[0]);
#undef CHECK
  return ok;
}

bool test_batched_solver_without_with_grad_or_with_fk_leaves_optional_fields_empty() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto keypoints_order = joint_names({"root", "joint1", "joint2", "tip"});

  std::vector<quickik::KeypointObservation> flat;
  for (auto &obs : observations_for(0.4f, 0.3f)) flat.push_back(obs);
  for (auto &obs : observations_for(-0.1f, 0.2f)) flat.push_back(obs);

  auto batched_solver = quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                                      kPositionTolerance, kAngleTolerance, kDamping,
                                                      keypoints_order, -1);
  auto result = batched_solver->solve(slice_of(flat), tree->n_joints(), false, false);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(result.joint_angles.size() == 2 * tree->n_dofs());
  CHECK(!result.has_keypoint_pos && result.keypoint_pos.empty());
  CHECK(!result.has_jacobian && result.jacobian.empty());
  CHECK(!result.has_cholesky_l && result.cholesky_l.empty() && result.valid.empty());
#undef CHECK
  return ok;
}

bool test_batched_solver_with_fk_reports_keypoint_positions() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto keypoints_order = joint_names({"root", "joint1", "joint2", "tip"});
  auto flat = observations_for(0.4f, 0.3f);

  auto batched_solver = quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                                      kPositionTolerance, kAngleTolerance, kDamping,
                                                      keypoints_order, -1);
  auto result = batched_solver->solve(slice_of(flat), tree->n_joints(), false, true);

  auto expected = two_link_positions(0.4f, 0.3f);
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(result.has_keypoint_pos);
  CHECK(result.keypoint_pos.size() == tree->n_joints() * 3);
  for (size_t k = 0; k < tree->n_joints(); k++) {
    for (size_t d = 0; d < 3; d++) {
      CHECK(std::abs(result.keypoint_pos[k * 3 + d] - expected[k][d]) < 1e-2f);
    }
  }
#undef CHECK
  return ok;
}

bool test_batched_solver_keypoint_to_joint_idx_matches_keypoints_order() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto keypoints_order = joint_names({"tip", "root", "joint2", "joint1"});
  auto batched_solver = quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight,
                                                      kPositionTolerance, kAngleTolerance, kDamping,
                                                      keypoints_order, -1);
  auto idx = batched_solver->keypoint_to_joint_idx();

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(idx.size() == 4);
  CHECK(idx[0] == 3 && idx[1] == 0 && idx[2] == 2 && idx[3] == 1);
#undef CHECK
  return ok;
}

bool test_batched_solver_rejects_unknown_joint_name() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto keypoints_order = joint_names({"root", "joint1", "joint2", "nonexistent"});
  try {
    quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                 kAngleTolerance, kDamping, keypoints_order, -1);
    std::fprintf(stderr, "  FAILED: expected an exception for an unknown joint name\n");
    ok = false;
  } catch (const std::exception &) {
    // expected
  }
  return ok;
}

bool test_batched_solver_rejects_duplicate_joint_name() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto keypoints_order = joint_names({"root", "joint1", "joint1", "tip"});
  try {
    quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                 kAngleTolerance, kDamping, keypoints_order, -1);
    std::fprintf(stderr, "  FAILED: expected an exception for a duplicate joint name\n");
    ok = false;
  } catch (const std::exception &) {
    // expected
  }
  return ok;
}

bool test_batched_solver_rejects_fixed_base_tree() {
  bool ok = true;
  auto tree = fixed_base_two_joint_chain();
  auto keypoints_order = joint_names({"root", "joint1", "joint2", "tip"});
  try {
    quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                 kAngleTolerance, kDamping, keypoints_order, -1);
    std::fprintf(stderr, "  FAILED: expected an exception for a fixed-base tree\n");
    ok = false;
  } catch (const std::exception &) {
    // expected
  }
  return ok;
}

bool test_batched_solver_rejects_zero_workers() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto keypoints_order = joint_names({"root", "joint1", "joint2", "tip"});
  try {
    quickik::new_batched_solver(*tree, quickik::no_mapper(), kNIterations, kNeutralWeight, kPositionTolerance,
                                 kAngleTolerance, kDamping, keypoints_order, 0);
    std::fprintf(stderr, "  FAILED: expected an exception for n_workers = 0\n");
    ok = false;
  } catch (const std::exception &) {
    // expected
  }
  return ok;
}

struct NamedTest {
  const char *name;
  bool (*run)();
};

}  // namespace

int main() {
  const std::vector<NamedTest> tests = {
      {"malformed_json_throws", test_malformed_json_throws},
      {"recovers_pose_from_3d_observations", test_recovers_pose_from_3d_observations},
      {"solve_with_fk_reports_keypoint_positions_matching_recovered_pose",
       test_solve_with_fk_reports_keypoint_positions_matching_recovered_pose},
      {"solve_without_with_fk_or_with_grad_leaves_optional_fields_empty",
       test_solve_without_with_fk_or_with_grad_leaves_optional_fields_empty},
      {"solve_with_grad_reports_jacobian_and_cholesky_l", test_solve_with_grad_reports_jacobian_and_cholesky_l},
      {"position2d_observation_on_mapperless_solver_throws", test_position2d_observation_on_mapperless_solver_throws},
      {"recovers_pose_from_xyview_observations", test_recovers_pose_from_xyview_observations},
      {"recovers_pose_from_camera_observations", test_recovers_pose_from_camera_observations},
      {"xyview_latency_not_much_worse_than_3d", test_xyview_latency_not_much_worse_than_3d},
      {"missing_observations_leave_state_at_neutral_prior", test_missing_observations_leave_state_at_neutral_prior},
      {"solver_fields_can_be_tuned_between_solve_calls", test_solver_fields_can_be_tuned_between_solve_calls},
      {"solve_respects_joint_limits", test_solve_respects_joint_limits},
      {"sequence_solver_warm_starts_across_separate_calls", test_sequence_solver_warm_starts_across_separate_calls},
      {"sequence_solver_solve_returns_one_result_per_frame", test_sequence_solver_solve_returns_one_result_per_frame},
      {"sequence_solver_solve_with_fk_matches_recovered_pose",
       test_sequence_solver_solve_with_fk_matches_recovered_pose},
      {"solver_result_list_at_out_of_range_throws", test_solver_result_list_at_out_of_range_throws},
      {"sequence_solver_solve_rejects_length_not_a_multiple_of_n_joints",
       test_sequence_solver_solve_rejects_length_not_a_multiple_of_n_joints},
      {"solve_segments_parallel_reconstructs_smooth_trajectory",
       test_solve_segments_parallel_reconstructs_smooth_trajectory},
      {"solve_segments_parallel_honors_explicit_n_workers", test_solve_segments_parallel_honors_explicit_n_workers},
      {"solve_segments_parallel_rejects_zero_workers", test_solve_segments_parallel_rejects_zero_workers},
      {"batched_solver_matches_sequential_solve", test_batched_solver_matches_sequential_solve},
      {"batched_solver_with_grad_reports_jacobian_and_valid", test_batched_solver_with_grad_reports_jacobian_and_valid},
      {"batched_solver_without_with_grad_or_with_fk_leaves_optional_fields_empty",
       test_batched_solver_without_with_grad_or_with_fk_leaves_optional_fields_empty},
      {"batched_solver_with_fk_reports_keypoint_positions", test_batched_solver_with_fk_reports_keypoint_positions},
      {"batched_solver_keypoint_to_joint_idx_matches_keypoints_order",
       test_batched_solver_keypoint_to_joint_idx_matches_keypoints_order},
      {"batched_solver_rejects_unknown_joint_name", test_batched_solver_rejects_unknown_joint_name},
      {"batched_solver_rejects_duplicate_joint_name", test_batched_solver_rejects_duplicate_joint_name},
      {"batched_solver_rejects_fixed_base_tree", test_batched_solver_rejects_fixed_base_tree},
      {"batched_solver_rejects_zero_workers", test_batched_solver_rejects_zero_workers},
  };

  int n_failed = 0;
  for (const auto &t : tests) {
    bool passed = t.run();
    std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", t.name);
    if (!passed) n_failed++;
  }
  std::printf("%zu/%zu tests passed\n", tests.size() - n_failed, tests.size());
  return n_failed == 0 ? 0 : 1;
}
