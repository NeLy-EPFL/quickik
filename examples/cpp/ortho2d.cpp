// Example fitting DOF angles from observations in 2D orthographic view.
//
// Bodies:
//   neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
//   g1: Unitree G1 robot (https://www.unitree.com/g1/)
//
// Input data:
//   Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
//   Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
//   (only XY positions used)
//
// To run this example (from the repository root):
//   cmake -S examples/cpp -B examples/cpp/build -DCMAKE_BUILD_TYPE=Release
//   cmake --build examples/cpp/build -j
//   ./examples/cpp/build/ortho2d <body>
// where <body> is one of "neuromechfly" or "g1". Results will be saved to
// examples/cpp/output/ortho2d_<body>.json
//
// To plot the results, run
//   uv run examples/plot_results.py examples/cpp/output/ortho2d_<body>.json
// and a video will be saved under the same directory. Run the `from_3d` example
// first to also show its fit for reference.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>

#include "quickik.hpp"

// Data utilities for the specific demos
#include "util.hpp"

const std::array<float, 2> kImageSize = {640.0f, 480.0f};

CameraView emulate_bottom_view(const Trajectory& trajectory, float scale);
std::unique_ptr<quickik::State> neutral_pose_on_ground(
    const quickik::KinematicTree& tree);

IkOutput run_ik_from_ortho2d_obs(const std::filesystem::path& body_plan_path,
                                 const std::filesystem::path& trajectory_path,
                                 float scale, float prior_weight,
                                 bool on_ground) {
  // ===== Step 1: Load the body plan ========================================
  // Fallible calls return a result; quickik::unwrap gets its value or throws.
  auto tree = quickik::unwrap(
      quickik::KinematicTree::from_json_file(body_plan_path.string()));

  // ===== Step 2: Load the motion trajectory ================================
  // This is the part that you would replace with your own data. For these
  // examples, utilities to load the trajectory, quantify errors, and save
  // results are provided in util.hpp.
  auto trajectory = Trajectory::load(trajectory_path);
  auto joint_indices = trajectory.joint_indices(*tree);
  // We emulate 2D observations by dropping the Z coordinate (orthographic
  // projection). In real applications, these would come directly from a 2D pose
  // model.
  auto view = emulate_bottom_view(trajectory, scale);
  // Make one list of KeypointObservations for all frames, and record how many
  // of them belong to each frame. Each observation carries the projection it
  // was observed in. Pixel errors are `scale` times larger than position
  // errors, so the weight is scaled down to match from_3d.
  float weight = 1.0f / (scale * scale);
  const auto& camera = *view.projection;
  auto observations = quickik::KeypointObservationList::create();
  std::vector<size_t> frame_lengths;
  for (const auto& uvs : view.observed) {
    for (size_t i = 0; i < joint_indices.size(); i++) {
      observations->add_position_2d(joint_indices[i], uvs[i], camera, weight);
    }
    frame_lengths.push_back(joint_indices.size());
  }

  // ===== Step 3: Configure the solver ======================================
  // With height unobserved, the neutral pose is the main thing keeping each
  // keypoint at a plausible height, so the neutral prior is stronger than in
  // from_3d.
  auto config = quickik::SolverConfig::create();
  // favor natural poses
  quickik::unwrap(config->add_dof_neutral_prior(2 * prior_weight));
  // favor smooth motion
  quickik::unwrap(config->add_dof_smoothness_prior(prior_weight));
  // The body's absolute height can't be recovered, so it stays at the initial
  // height
  auto initial_state = on_ground ? neutral_pose_on_ground(*tree)
                                 : quickik::State::neutral_pose(*tree);

  // ===== Step 4: Solve =====================================================
  auto solver = quickik::unwrap(
      quickik::SequentialSolver::create(*tree, *config, initial_state.get()));
  auto sequence = quickik::unwrap(
      quickik::ObservationSequence::from_flat(*observations, frame_lengths));
  auto start = std::chrono::steady_clock::now();
  // Set with_grad to false to skip gradient tracking
  // Set with_fk to true to have forward kinematics (fitted keypoint positions)
  // in the results
  auto results = quickik::unwrap(solver->solve(*sequence, false, true));
  double solve_time = seconds_since(start);

  std::vector<CameraView> views;
  views.push_back(std::move(view));
  return {std::move(tree),
          std::move(trajectory),
          std::move(joint_indices),
          std::move(results),
          solve_time,
          std::move(views)};
}

// ===== Main and helper functions below (unrelated to core QuickIK logic) =====

// Emulates a camera below the body with its image axes along world X and Y, at
// `scale` pixels per unit length (`uv = scale * (x, y) + offset`), centered on
// the trajectory's mean position.
CameraView emulate_bottom_view(const Trajectory& trajectory, float scale) {
  auto center = trajectory.mean_position();
  quickik::Vec2 offset{kImageSize[0] / 2 - scale * center.x,
                       kImageSize[1] / 2 - scale * center.y};
  auto projection =
      quickik::Projection::from_orthographic_view_xy({scale, offset});
  std::vector<std::vector<quickik::Vec2>> observed;
  for (const auto& positions : trajectory.positions) {
    auto& uvs = observed.emplace_back();
    for (const auto& p : positions) uvs.push_back(projection->project(p));
  }
  return {"bottom view", kImageSize, std::move(projection),
          std::move(observed)};
}

// The neutral pose, lifted so that its lowest keypoint is on the ground (z =
// 0).
std::unique_ptr<quickik::State> neutral_pose_on_ground(
    const quickik::KinematicTree& tree) {
  auto state = quickik::State::neutral_pose(tree);
  auto workspace = quickik::ForwardKinematicsWorkspace::create(tree);
  quickik::unwrap(quickik::forward_kinematics(*workspace, *state));
  auto positions = workspace->keypoint_positions();
  float min_z = std::numeric_limits<float>::infinity();
  for (size_t i = 2; i < positions.size(); i += 3)
    min_z = std::min(min_z, positions[i]);
  state->set_root_pos({0, 0, -min_z});
  return state;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: ortho2d <neuromechfly|g1>\n");
    return 1;
  }
  std::string body = argv[1];
  // Body-specific settings: image scale (pixels per unit length), prior weight
  // (depends on the units), and whether the recording has the body on the
  // ground at z = 0 (G1) rather than centered on its root (the fly's thorax).
  float scale, prior_weight;
  bool on_ground;
  if (body == "g1") {
    scale = 130.0f;  // 130 px/m
    prior_weight = 1e-2f;
    on_ground = true;
  } else if (body == "neuromechfly") {
    scale = 100.0f;  // 100 px/mm
    prior_weight = 1e-3f;
    on_ground = false;
  } else {
    std::fprintf(stderr, "Unknown body: %s\n", body.c_str());
    return 1;
  }
  auto assets = kExamplesDir / "assets" / body;

  auto output = run_ik_from_ortho2d_obs(assets / "body_plan.json",
                                        assets / "example_trajectory.json",
                                        scale, prior_weight, on_ground);

  // Compare with the recorded 3D positions, which the solver never saw.
  float error_xy = 0, error_z = 0;
  auto errors = output.errors();
  for (const auto& e : errors) {
    error_xy += std::hypot(e.x, e.y);
    error_z += std::abs(e.z);
  }
  const char* units = output.trajectory.units.c_str();
  std::printf(
      "Fitted %zu frames in %.1f ms, mean keypoint error vs. 3D recording: "
      "%.4f %s in XY, %.4f %s in Z\n",
      output.results->len(), output.solve_time * 1e3, error_xy / errors.size(),
      units, error_z / errors.size(), units);
  save_output("ortho2d", body, output);
}
