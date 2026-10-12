// Tests for QuickIK's C++ bindings. The solver itself is tested in Rust
// (tests/), so these focus on what the bindings add: conversions, array
// layouts, error translation, and list types. Most use a "two-joint chain": a
// root, joint1 and joint2 (each with one Z-axis hinge, joint2's limited to
// [-0.5, 0.5]) and a tip, each 1 unit along X from its parent.
//
// No test framework: each test is a function that CHECKs conditions, and
// main() runs them all and reports failures.

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "quickik.hpp"

namespace {

int n_failures = 0;

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "  FAILED: %s (line %d)\n", #cond, __LINE__); \
      n_failures++;                                                      \
    }                                                                    \
  } while (0)

#define CHECK_THROWS(expr, expected_kind)                                  \
  do {                                                                     \
    try {                                                                  \
      expr;                                                                \
      std::fprintf(stderr, "  FAILED: %s didn't throw (line %d)\n", #expr, \
                   __LINE__);                                              \
      n_failures++;                                                        \
    } catch (const quickik::Exception& e) {                                \
      CHECK(e.kind() == quickik::ErrorKind::expected_kind);                \
    }                                                                      \
  } while (0)

bool near(float a, float b, float tol = 1e-3f) {
  return std::abs(a - b) <= tol;
}

std::string two_joint_chain_json(bool fixed_base = false) {
  const char* joints = R"JSON([
    {"name": "root", "parent": null, "offset_pos": [0, 0, 0], "offset_quat": [1, 0, 0, 0], "dofs": []},
    {"name": "joint1", "parent": "root", "offset_pos": [1, 0, 0], "offset_quat": [1, 0, 0, 0],
     "dofs": [{"axis": [0, 0, 1], "type": "hinge", "neutral": 0, "limits": null}]},
    {"name": "joint2", "parent": "joint1", "offset_pos": [1, 0, 0], "offset_quat": [1, 0, 0, 0],
     "dofs": [{"axis": [0, 0, 1], "type": "hinge", "neutral": 0, "limits": [-0.5, 0.5]}]},
    {"name": "tip", "parent": "joint2", "offset_pos": [1, 0, 0], "offset_quat": [1, 0, 0, 0], "dofs": []}
  ])JSON";
  return std::string("{\"fixed_base\": ") + (fixed_base ? "true" : "false") +
         ", \"joints\": " + joints + "}";
}

std::unique_ptr<quickik::KinematicTree> two_joint_chain(
    bool fixed_base = false) {
  return quickik::unwrap(
      quickik::KinematicTree::from_json_str(two_joint_chain_json(fixed_base)));
}

// Keypoint positions of the chain with DOF values (a1, a2). A joint's own DOFs
// only move its descendants, so joint1 stays put.
std::vector<quickik::Vec3> chain_positions(float a1, float a2) {
  return {
      {0, 0, 0},
      {1, 0, 0},
      {1 + std::cos(a1), std::sin(a1), 0},
      {1 + std::cos(a1) + std::cos(a1 + a2), std::sin(a1) + std::sin(a1 + a2),
       0},
  };
}

std::unique_ptr<quickik::KeypointObservationList> observations_3d(float a1,
                                                                  float a2) {
  auto observations = quickik::KeypointObservationList::create();
  auto positions = chain_positions(a1, a2);
  for (size_t joint = 0; joint < positions.size(); joint++) {
    observations->add_position_3d(joint, positions[joint], 1.0f);
  }
  return observations;
}

// No priors and no early stopping, so solves converge to an exact fit.
std::unique_ptr<quickik::SolverConfig> exact_config() {
  return quickik::SolverConfig::create(20, 0, 0);
}

// A pinhole camera 5 units above the chain, looking down.
std::unique_ptr<quickik::Projection> camera(float x_offset) {
  quickik::Mat3 rot{{1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
  return quickik::Projection::from_pinhole_camera(
      {500, 500, 320, 240, {-x_offset, 0, 5}, rot});
}

// DOF values (a1, a2) of frame t of a smooth trajectory.
std::pair<float, float> sine_dof_values(size_t t) {
  float s = std::sin(0.15f * static_cast<float>(t));
  return {0.3f * s, 0.15f * s};
}

// A sequence of `n_frames` frames of the sine trajectory, observed in 3D.
std::unique_ptr<quickik::ObservationSequence> sine_sequence(size_t n_frames) {
  auto observations = quickik::KeypointObservationList::create();
  std::vector<size_t> frame_lengths;
  for (size_t t = 0; t < n_frames; t++) {
    auto [a1, a2] = sine_dof_values(t);
    auto positions = chain_positions(a1, a2);
    for (size_t joint = 0; joint < positions.size(); joint++) {
      observations->add_position_3d(joint, positions[joint], 1.0f);
    }
    frame_lengths.push_back(positions.size());
  }
  return quickik::unwrap(
      quickik::ObservationSequence::from_flat(*observations, frame_lengths));
}

// =============================================================================
//  Body plan, state and forward kinematics
// =============================================================================

void test_tree_queries() {
  auto tree = two_joint_chain();
  CHECK(tree->n_joints() == 4);
  CHECK(tree->n_dofs() == 2);
  CHECK(tree->state_dim() == 8);
  CHECK(tree->joint_index("tip") == 3u);
  CHECK(!tree->joint_index("nonexistent").has_value());
  CHECK(two_joint_chain(true)->state_dim() == 2);
}

void test_errors_become_exceptions() {
  CHECK_THROWS(
      quickik::unwrap(quickik::KinematicTree::from_json_str("not valid json")),
      Json);
  CHECK_THROWS(quickik::unwrap(
                   quickik::KinematicTree::from_json_file("/nonexistent.json")),
               Io);

  // The message is the core error's.
  try {
    auto config = quickik::SolverConfig::create();
    quickik::unwrap(config->add_dof_neutral_prior(1.0f));
    quickik::unwrap(config->add_dof_neutral_prior(1.0f));
    CHECK(false);
  } catch (const quickik::Exception& e) {
    CHECK(std::string(e.what()).find("already added") != std::string::npos);
  }
}

void test_state_accessors() {
  auto tree = two_joint_chain();
  auto state = quickik::State::neutral_pose(*tree);
  state->set_root_pos({1, 2, 3});
  CHECK(state->root_pos().z == 3);
  // Normalized: a 90 degree rotation about Z.
  state->set_root_quat({2, 0, 0, 2});
  CHECK(near(state->root_quat().w, std::sqrt(0.5f), 1e-6f));
  CHECK(near(state->root_quat().z, std::sqrt(0.5f), 1e-6f));

  std::vector<float> dof_values{0.4f, 0.3f};
  quickik::unwrap(state->set_dof_values(dof_values));
  CHECK(state->dof_values()[1] == 0.3f);
  CHECK_THROWS(quickik::unwrap(state->set_dof_values(std::vector<float>{1.0f})),
               InvalidInput);

  // Clones are independent.
  auto copy = state->clone();
  state->set_root_pos({0, 0, 0});
  CHECK(copy->root_pos().z == 3);
}

void test_forward_kinematics() {
  auto tree = two_joint_chain();
  auto state = quickik::State::neutral_pose(*tree);
  quickik::unwrap(state->set_dof_values(std::vector<float>{0.4f, 0.3f}));
  auto workspace = quickik::ForwardKinematicsWorkspace::create(*tree);
  quickik::unwrap(quickik::forward_kinematics(*workspace, *state));

  auto positions = workspace->keypoint_positions();
  auto expected = chain_positions(0.4f, 0.3f);
  CHECK(positions.size() == 12);
  for (size_t joint = 0; joint < expected.size(); joint++) {
    CHECK(near(positions[3 * joint], expected[joint].x, 1e-5f));
    CHECK(near(positions[3 * joint + 1], expected[joint].y, 1e-5f));
  }

  // A state of a tree with a different number of DOFs.
  auto other_tree =
      quickik::unwrap(quickik::KinematicTree::from_json_str(R"JSON({"joints": [
    {"name": "root", "parent": null, "offset_pos": [0, 0, 0], "offset_quat": [1, 0, 0, 0], "dofs": []}
  ]})JSON"));
  auto other_state = quickik::State::neutral_pose(*other_tree);
  CHECK_THROWS(
      quickik::unwrap(quickik::forward_kinematics(*workspace, *other_state)),
      InvalidInput);
}

// =============================================================================
//  Observations
// =============================================================================

void test_projections() {
  auto ortho = quickik::Projection::from_orthographic_view_xy({2, {10, 20}});
  auto uv = ortho->project({1, 2, 3});
  CHECK(uv.x == 12 && uv.y == 24);
  CHECK(!ortho->as_pinhole_camera().has_value());
  CHECK(ortho->as_orthographic_view_xy()->offset.y == 20);

  // A camera at the origin looking along +Z, rotated 90 degrees about Z, so a
  // wrong row/column order of the rotation matrix would flip the result.
  quickik::Mat3 rot{{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}};
  auto cam = quickik::Projection::from_pinhole_camera(
      {100, 100, 0, 0, {0, 0, 0}, rot});
  uv = cam->project({1, 0, 1});
  CHECK(near(uv.x, 0) && near(uv.y, -100));
  auto params = cam->as_pinhole_camera();
  CHECK(params.has_value() && params->world2cam_rot_mat.row1.x == -1);
  CHECK(!cam->as_orthographic_view_xy().has_value());
}

void test_sequence_construction_checks_lengths() {
  auto observations = observations_3d(0, 0);
  auto sequence = quickik::unwrap(quickik::ObservationSequence::from_flat(
      *observations, std::vector<size_t>{1, 3}));
  CHECK(sequence->len() == 2);
  CHECK_THROWS(quickik::unwrap(quickik::ObservationSequence::from_flat(
                   *observations, std::vector<size_t>{1, 2})),
               InvalidInput);

  std::vector<float> pos_3d(5 * 4 * 3), weights_3d(5 * 4, 1.0f);
  std::vector<const quickik::Projection*> no_projections;
  sequence = quickik::unwrap(quickik::ObservationSequence::from_arrays(
      5, 4, pos_3d, weights_3d, no_projections, std::nullopt, std::nullopt));
  CHECK(sequence->len() == 5);
  weights_3d.pop_back();
  CHECK_THROWS(quickik::unwrap(quickik::ObservationSequence::from_arrays(
                   5, 4, pos_3d, weights_3d, no_projections, std::nullopt,
                   std::nullopt)),
               InvalidInput);
}

// =============================================================================
//  Solvers
// =============================================================================

void test_single_frame_solver_from_3d() {
  auto tree = two_joint_chain();
  auto solver = quickik::unwrap(
      quickik::SingleFrameSolver::create(*tree, *exact_config()));
  auto state = quickik::State::neutral_pose(*tree);
  auto result = quickik::unwrap(
      solver->solve(*state, *observations_3d(0.4f, 0.3f), false, true));

  auto dof_values = result->state().dof_values();
  CHECK(near(dof_values[0], 0.4f) && near(dof_values[1], 0.3f));
  // The state is updated in place.
  CHECK(near(state->dof_values()[0], 0.4f));
  auto keypoint_pos = result->keypoint_pos();
  CHECK(keypoint_pos.has_value() && keypoint_pos->size() == 12);
  CHECK(near((*keypoint_pos)[3 * 3], chain_positions(0.4f, 0.3f)[3].x));
  CHECK(!result->jacobian().has_value() && !result->cholesky_l().has_value());
}

void test_single_frame_solver_gradients_are_row_major() {
  auto tree = two_joint_chain();
  auto solver = quickik::unwrap(
      quickik::SingleFrameSolver::create(*tree, *exact_config()));
  auto state = quickik::State::neutral_pose(*tree);
  auto result = quickik::unwrap(
      solver->solve(*state, *observations_3d(0.4f, 0.3f), true, false));
  const size_t n_rows = 12, dim = tree->state_dim();
  auto jacobian = *result->jacobian();
  auto l = *result->cholesky_l();
  CHECK(jacobian.size() == n_rows * dim && l.size() == dim * dim);

  // The root keypoint's rows: its position moves 1:1 with the root position
  // (columns 0 to 2) and not with the root rotation (columns 3 to 5).
  CHECK(near(jacobian[0 * dim + 0], 1) && near(jacobian[1 * dim + 1], 1));
  CHECK(near(jacobian[0 * dim + 1], 0) && near(jacobian[1 * dim + 3], 0));
  // L is lower-triangular, and L L^T = J^T J (no priors or weights, up to the
  // tiny damping).
  CHECK(l[0 * dim + 1] == 0);
  float max_error = 0;
  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      float llt = 0, jtj = 0;
      for (size_t k = 0; k < dim; k++) llt += l[i * dim + k] * l[j * dim + k];
      for (size_t r = 0; r < n_rows; r++)
        jtj += jacobian[r * dim + i] * jacobian[r * dim + j];
      max_error = std::max(max_error, std::abs(llt - jtj));
    }
  }
  CHECK(max_error < 1e-2f);
}

void test_single_frame_solver_from_two_cameras() {
  // 2D observations from two projections mixed in one frame.
  auto tree = two_joint_chain();
  std::vector<std::unique_ptr<quickik::Projection>> cameras;
  cameras.push_back(camera(-1));
  cameras.push_back(camera(1));
  auto observations = quickik::KeypointObservationList::create();
  auto positions = chain_positions(0.2f, -0.3f);
  for (const auto& cam : cameras) {
    for (size_t joint = 0; joint < positions.size(); joint++) {
      observations->add_position_2d(joint, cam->project(positions[joint]), *cam,
                                    1.0f);
    }
  }
  CHECK(observations->len() == 8);

  auto solver = quickik::unwrap(
      quickik::SingleFrameSolver::create(*tree, *exact_config()));
  auto state = quickik::State::neutral_pose(*tree);
  auto result =
      quickik::unwrap(solver->solve(*state, *observations, true, false));
  CHECK(near(state->dof_values()[0], 0.2f) &&
        near(state->dof_values()[1], -0.3f));
  // 2 rows per 2D observation.
  CHECK(result->jacobian()->size() == 16 * tree->state_dim());
}

void test_solver_creation_errors() {
  auto tree = two_joint_chain();
  auto config = quickik::SolverConfig::create();
  quickik::unwrap(config->add_dof_smoothness_prior(1.0f));
  CHECK_THROWS(
      quickik::unwrap(quickik::SingleFrameSolver::create(*tree, *config)),
      InvalidInput);

  auto root_prior_config = quickik::SolverConfig::create();
  quickik::unwrap(root_prior_config->add_root_smoothness_prior(1.0f, 1.0f));
  CHECK_THROWS(quickik::unwrap(quickik::SequentialSolver::create(
                   *two_joint_chain(true), *root_prior_config, nullptr)),
               InvalidInput);
}

void test_sequential_solver_step_solve_and_solve_parallel_agree() {
  const size_t n_frames = 40;
  auto tree = two_joint_chain();
  auto config = quickik::SolverConfig::create();
  quickik::unwrap(config->add_dof_smoothness_prior(1e-3f));
  auto sequence = sine_sequence(n_frames);

  auto solver = quickik::unwrap(
      quickik::SequentialSolver::create(*tree, *config, nullptr));
  auto solved = quickik::unwrap(solver->solve(*sequence, false, false));
  CHECK(solved->len() == n_frames);
  // The running state ends at the last frame.
  CHECK(solver->state()->dof_values()[0] ==
        (*solved)[n_frames - 1]->state().dof_values()[0]);
  CHECK((*solved)[n_frames] == nullptr);

  // The same frames from arrays solve identically.
  std::vector<float> pos_3d, weights_3d(n_frames * 4, 1.0f);
  for (size_t t = 0; t < n_frames; t++) {
    auto [a1, a2] = sine_dof_values(t);
    for (const auto& p : chain_positions(a1, a2)) {
      pos_3d.insert(pos_3d.end(), {p.x, p.y, p.z});
    }
  }
  std::vector<const quickik::Projection*> no_projections;
  auto from_arrays = quickik::unwrap(quickik::ObservationSequence::from_arrays(
      n_frames, 4, pos_3d, weights_3d, no_projections, std::nullopt,
      std::nullopt));
  quickik::unwrap(solver->reset(nullptr));
  auto solved_from_arrays =
      quickik::unwrap(solver->solve(*from_arrays, false, false));
  CHECK((*solved_from_arrays)[n_frames - 1]->state().dof_values()[1] ==
        (*solved)[n_frames - 1]->state().dof_values()[1]);

  quickik::unwrap(solver->reset(nullptr));
  std::vector<float> stepped;
  for (size_t t = 0; t < n_frames; t++) {
    auto [a1, a2] = sine_dof_values(t);
    auto result =
        quickik::unwrap(solver->step(*observations_3d(a1, a2), false, false));
    stepped.push_back(result->state().dof_values()[0]);
  }

  auto parallel =
      quickik::unwrap(solver->solve_parallel(*sequence, 4, 5, false, false));
  size_t t = 0;
  for (const auto& result : *parallel) {
    float expected = sine_dof_values(t).first;
    CHECK(near(result.state().dof_values()[0], expected, 1e-2f));
    CHECK(near(stepped[t], (*solved)[t]->state().dof_values()[0], 1e-6f));
    t++;
  }
  CHECK(t == n_frames);
  CHECK_THROWS(
      quickik::unwrap(solver->solve_parallel(*sequence, 0, 0, false, false)),
      InvalidInput);
}

void test_sequential_solver_initial_state() {
  auto tree = two_joint_chain();
  auto initial_state = quickik::State::neutral_pose(*tree);
  initial_state->set_root_pos({0, 0, 7});
  auto config = quickik::SolverConfig::create();
  auto solver = quickik::unwrap(
      quickik::SequentialSolver::create(*tree, *config, initial_state.get()));
  CHECK(solver->state()->root_pos().z == 7);
  // A copy: changing the original afterwards doesn't affect the solver.
  initial_state->set_root_pos({0, 0, 0});
  CHECK(solver->state()->root_pos().z == 7);
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"tree_queries", test_tree_queries},
      {"errors_become_exceptions", test_errors_become_exceptions},
      {"state_accessors", test_state_accessors},
      {"forward_kinematics", test_forward_kinematics},
      {"projections", test_projections},
      {"sequence_construction_checks_lengths",
       test_sequence_construction_checks_lengths},
      {"single_frame_solver_from_3d", test_single_frame_solver_from_3d},
      {"single_frame_solver_gradients_are_row_major",
       test_single_frame_solver_gradients_are_row_major},
      {"single_frame_solver_from_two_cameras",
       test_single_frame_solver_from_two_cameras},
      {"solver_creation_errors", test_solver_creation_errors},
      {"sequential_solver_step_solve_and_solve_parallel_agree",
       test_sequential_solver_step_solve_and_solve_parallel_agree},
      {"sequential_solver_initial_state", test_sequential_solver_initial_state},
  };
  int n_failed_tests = 0;
  for (const auto& [name, test] : tests) {
    int failures_before = n_failures;
    try {
      test();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  FAILED: unexpected exception: %s\n", e.what());
      n_failures++;
    }
    bool passed = n_failures == failures_before;
    n_failed_tests += !passed;
    std::printf("%s %s\n", passed ? "PASS" : "FAIL", name);
  }
  std::printf("\n%zu tests, %d failed\n", tests.size(), n_failed_tests);
  return n_failed_tests == 0 ? 0 : 1;
}
