// Tests for the QuickIK C++ bindings, mirroring tests/solver_test.rs and
// tests/high_level_test.rs. Uses the same "two-joint chain" fixture as those
// (see tests/common/mod.rs): a root, joint1 and joint2 (each with one Z-axis
// DOF, joint2 limited to [-0.5, 0.5]), and a trailing fixed tip.
//
// Forward kinematics isn't exposed to C++ (same as Python -- see
// benchmark/scripts/bench_python.py's own from-scratch FK replica), so
// `two_link_positions` below computes the four keypoints' world positions
// directly from the chain's known geometry, in `[root, joint1, joint2, tip]`
// order, to build observations for a target pose.
//
// No external test framework: each `test_*` function returns true on success,
// printing a message and returning false on the first failed CHECK.

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "quickik.h"

namespace {

const char *kTwoJointChainJson = R"JSON(
{
    "joints": [
        {"name": "root", "parent": null, "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []},
        {"name": "joint1", "parent": "root", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "neutral_angle": 0.0, "limits": null}]},
        {"name": "joint2", "parent": "joint1", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "neutral_angle": 0.0, "limits": [-0.5, 0.5]}]},
        {"name": "tip", "parent": "joint2", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []}
    ]
}
)JSON";

rust::Box<quickik::KinematicTree> two_joint_chain() {
  return quickik::kinematic_tree_from_json_str(kTwoJointChainJson);
}

// Positions of [root, joint1, joint2, tip] when joint1/joint2 are at angles
// (a1, a2) about the shared Z axis -- see tests/common/mod.rs's doc comment
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

quickik::SolverConfig no_prior_config() {
  quickik::SolverConfig config = quickik::default_solver_config();
  config.neutral_pose_weight = 0.0f;
  return config;
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
  auto solver = quickik::new_solver(*tree, no_prior_config(), quickik::no_mapper());
  solver->solve(*state, slice_of(observations));

  auto angles = state->dof_angles();
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(std::abs(angles[0] - 0.4f) < 1e-3f);
  CHECK(std::abs(angles[1] - 0.3f) < 1e-3f);
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

  auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());
  try {
    solver->solve(*state, slice_of(observations));
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
  auto solver = quickik::new_solver(*tree, no_prior_config(), quickik::xyview_mapper());
  solver->solve(*state, slice_of(observations));

  auto angles = state->dof_angles();
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(std::abs(angles[0] - 0.35f) < 1e-3f);
  CHECK(std::abs(angles[1] - (-0.25f)) < 1e-3f);
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
  auto solver = quickik::new_solver(*tree, no_prior_config(), quickik::camera_mapper(camera));
  solver->solve(*state, slice_of(observations));

  auto angles = state->dof_angles();
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(std::abs(angles[0] - 0.2f) < 1e-3f);
  CHECK(std::abs(angles[1] - 0.15f) < 1e-3f);
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

  auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());
  solver->solve(*state, slice_of(observations));

  for (float angle : state->dof_angles()) {
#define CHECK(cond) \
    if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
    CHECK(std::abs(angle) < 1e-6f);
#undef CHECK
  }
  return ok;
}

bool test_config_can_be_tuned_between_solve_calls() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto state = quickik::state_neutral_pose(*tree);
  std::vector<quickik::KeypointObservation> observations;
  for (size_t i = 0; i < tree->n_joints(); i++) {
    observations.push_back(quickik::keypoint_missing());
  }

  auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());
  solver->solve(*state, slice_of(observations));

  quickik::SolverConfig config = solver->config();
  config.n_iterations = 3;
  solver->set_config(config);
  solver->solve(*state, slice_of(observations));

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(solver->config().n_iterations == 3);
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

  auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());
  solver->solve(*state, slice_of(observations));

  float joint2_angle = state->dof_angles()[1];
#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(joint2_angle >= -0.5f - 1e-6f && joint2_angle <= 0.5f + 1e-6f);
  CHECK(joint2_angle > 0.45f);
#undef CHECK
  return ok;
}

bool test_sequence_solver_warm_start_converges_faster() {
  bool ok = true;
  auto tree = two_joint_chain();
  quickik::SolverConfig config = no_prior_config();
  config.n_iterations = 1;
  auto target = observations_for(0.4f, 0.3f);

  auto cold = quickik::new_sequence_solver(*tree, config, quickik::no_mapper());
  cold->solve_frame(slice_of(target));
  float cold_error = std::abs(cold->state()->dof_angles()[0] - 0.4f);

  auto warm = quickik::new_sequence_solver(*tree, config, quickik::no_mapper());
  warm->solve_frame(slice_of(target));
  warm->solve_frame(slice_of(target));
  float warm_error = std::abs(warm->state()->dof_angles()[0] - 0.4f);

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(warm_error < cold_error);
#undef CHECK
  return ok;
}

bool test_solve_sequence_returns_one_state_per_frame() {
  bool ok = true;
  auto tree = two_joint_chain();
  auto solver = quickik::new_sequence_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());

  std::vector<quickik::KeypointObservation> flat;
  for (auto [a1, a2] : {std::pair{0.1f, 0.05f}, std::pair{0.2f, 0.1f}, std::pair{0.3f, 0.15f}}) {
    for (auto &obs : observations_for(a1, a2)) flat.push_back(obs);
  }

  auto states = solver->solve_sequence(slice_of(flat), tree->n_joints());

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(states->len() == 3);
  auto last = states->at(2);
  CHECK(std::abs(last->dof_angles()[0] - 0.3f) < 1e-2f);
  CHECK(std::abs(last->dof_angles()[1] - 0.15f) < 1e-2f);
#undef CHECK
  return ok;
}

bool test_solve_sequence_segmented_parallel_reconstructs_smooth_trajectory() {
  bool ok = true;
  auto tree = two_joint_chain();
  const size_t n_frames = 40;
  std::vector<std::array<float, 2>> true_angles;
  std::vector<quickik::KeypointObservation> flat;
  for (size_t t = 0; t < n_frames; t++) {
    float a = 0.3f * std::sin(t * 0.15f);
    true_angles.push_back({a, a * 0.5f});
    for (auto &obs : observations_for(a, a * 0.5f)) flat.push_back(obs);
  }

  quickik::SegmentedSolveConfig segmented_config{10, 3, 0.05f};
  auto states = quickik::solve_sequence_segmented_parallel(
      *tree, quickik::default_solver_config(), slice_of(flat), tree->n_joints(), segmented_config, quickik::no_mapper());

#define CHECK(cond) \
  if (!(cond)) { std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); ok = false; }
  CHECK(states->len() == n_frames);
  for (size_t i = 0; i < n_frames; i++) {
    auto state = states->at(i);
    auto angles = state->dof_angles();
    CHECK(std::abs(angles[0] - true_angles[i][0]) < 1e-2f);
    CHECK(std::abs(angles[1] - true_angles[i][1]) < 1e-2f);
  }
#undef CHECK
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
      {"position2d_observation_on_mapperless_solver_throws", test_position2d_observation_on_mapperless_solver_throws},
      {"recovers_pose_from_3d_observations", test_recovers_pose_from_3d_observations},
      {"recovers_pose_from_xyview_observations", test_recovers_pose_from_xyview_observations},
      {"recovers_pose_from_camera_observations", test_recovers_pose_from_camera_observations},
      {"missing_observations_leave_state_at_neutral_prior", test_missing_observations_leave_state_at_neutral_prior},
      {"config_can_be_tuned_between_solve_calls", test_config_can_be_tuned_between_solve_calls},
      {"solve_respects_joint_limits", test_solve_respects_joint_limits},
      {"sequence_solver_warm_start_converges_faster", test_sequence_solver_warm_start_converges_faster},
      {"solve_sequence_returns_one_state_per_frame", test_solve_sequence_returns_one_state_per_frame},
      {"solve_sequence_segmented_parallel_reconstructs_smooth_trajectory",
       test_solve_sequence_segmented_parallel_reconstructs_smooth_trajectory},
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
