// Example fitting DOF angles from 2D keypoint observations from two cameras.
//
// Bodies:
//   neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
//   g1: Unitree G1 robot (https://www.unitree.com/g1/)
//
// Input data:
//   Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
//   Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
//   (2D observations are emulated by projecting the 3D positions into two
//   pinhole cameras)
//
// To run this example (from the repository root):
//   cmake -S examples/cpp -B examples/cpp/build -DCMAKE_BUILD_TYPE=Release
//   cmake --build examples/cpp/build -j
//   ./examples/cpp/build/from_cameras <body>
// where <body> is one of "neuromechfly" or "g1". Results will be saved to
// examples/cpp/output/from_cameras_<body>.json
//
// To plot the results, run
//   uv run examples/plot_results.py
//   examples/cpp/output/from_cameras_<body>.json
// and a video will be saved under the same directory. Run the `from_3d` example
// first to also show its fit for reference.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

#include "quickik.hpp"

// Data utilities for the specific demos
#include "util.hpp"

std::vector<CameraView> emulate_two_camera_views(
    const Trajectory& trajectory, float distance, float focal_length,
    std::array<float, 2> image_size);

IkOutput run_ik_from_camera_obs(const std::filesystem::path& body_plan_path,
                                const std::filesystem::path& trajectory_path,
                                float distance, float focal_length,
                                std::array<float, 2> image_size,
                                float prior_weight) {
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
  // We emulate 2D observations by projecting 3D positions into two cameras. In
  // real applications, these would come directly from a 2D pose model.
  auto views =
      emulate_two_camera_views(trajectory, distance, focal_length, image_size);
  // Make one list of KeypointObservations for all frames, and record how many
  // of them belong to each frame. Each frame's observations mix both cameras,
  // each observation carrying the projection (camera) it was observed in. Pixel
  // errors are about focal_length / distance times larger than position errors,
  // so the weight is scaled down to match from_3d.
  float weight = std::pow(distance / focal_length, 2.0f);
  auto observations = quickik::KeypointObservationList::create();
  std::vector<size_t> frame_lengths;
  for (size_t t = 0; t < trajectory.positions.size(); t++) {
    for (const auto& view : views) {
      for (size_t i = 0; i < joint_indices.size(); i++) {
        observations->add_position_2d(joint_indices[i], view.observed[t][i],
                                      *view.projection, weight);
      }
    }
    frame_lengths.push_back(views.size() * joint_indices.size());
  }

  // ===== Step 3: Configure the solver ======================================
  auto config = quickik::SolverConfig::create();
  // favor natural poses
  quickik::unwrap(config->add_dof_neutral_prior(prior_weight));
  // favor smooth motion
  quickik::unwrap(config->add_dof_smoothness_prior(prior_weight));

  // ===== Step 4: Solve =====================================================
  auto solver = quickik::unwrap(
      quickik::SequentialSolver::create(*tree, *config, nullptr));
  auto sequence = quickik::unwrap(
      quickik::ObservationSequence::from_flat(*observations, frame_lengths));
  auto start = std::chrono::steady_clock::now();
  // Set with_grad to false to skip gradient tracking
  // Set with_fk to true to have forward kinematics (fitted keypoint positions)
  // in the results
  auto results = quickik::unwrap(solver->solve(*sequence, false, true));
  double solve_time = seconds_since(start);

  return {std::move(tree),
          std::move(trajectory),
          std::move(joint_indices),
          std::move(results),
          solve_time,
          std::move(views)};
}

// ===== Main and helper functions below (unrelated to core QuickIK logic) =====

// A pinhole camera `distance` away from `target` at the same height, at the
// given azimuth (counterclockwise from +x, seen from above), facing `target`
// with it at the image center.
std::unique_ptr<quickik::Projection> make_pinhole_camera(
    quickik::Vec3 target, float distance, float azimuth_deg, float focal_length,
    std::array<float, 2> image_size) {
  float azimuth = azimuth_deg * std::numbers::pi_v<float> / 180.0f;
  quickik::Vec3 direction{std::cos(azimuth), std::sin(azimuth), 0};
  auto position = target + distance * direction;

  // Camera axes in world coordinates: z forward (toward the target), x to the
  // right in the image, y down in the image.
  auto z = -direction;
  auto x = normalize(cross(z, {0, 0, 1}));
  auto y = cross(z, x);
  quickik::Mat3 world2cam_rot_mat{x, y, z};
  auto world2cam_pos = -(world2cam_rot_mat * position);
  auto [width, height] = image_size;
  float cx = width / 2, cy = height / 2;
  return quickik::Projection::from_pinhole_camera(
      {focal_length, focal_length, cx, cy, world2cam_pos, world2cam_rot_mat});
}

// Emulates two cameras, 45 degrees to the front left and front right of the
// body (which faces +x), at `distance` from the trajectory's average position
// and looking horizontally at it.
std::vector<CameraView> emulate_two_camera_views(
    const Trajectory& trajectory, float distance, float focal_length,
    std::array<float, 2> image_size) {
  auto target = trajectory.mean_position();
  std::vector<CameraView> views;
  for (auto [name, azimuth_deg] :
       {std::pair{"left camera", 45.0f}, std::pair{"right camera", -45.0f}}) {
    auto camera = make_pinhole_camera(target, distance, azimuth_deg,
                                      focal_length, image_size);
    std::vector<std::vector<quickik::Vec2>> observed;
    for (const auto& positions : trajectory.positions) {
      auto& uvs = observed.emplace_back();
      for (const auto& p : positions) uvs.push_back(camera->project(p));
    }
    views.push_back({name, image_size, std::move(camera), std::move(observed)});
  }
  return views;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: from_cameras <neuromechfly|g1>\n");
    return 1;
  }
  std::string body = argv[1];
  // Body-specific settings: camera distance, focal length and image size in
  // pixels (chosen so the whole recording just fits in the image), and prior
  // weight (depends on the units).
  float distance, focal_length, prior_weight;
  std::array<float, 2> image_size;
  if (body == "g1") {  // 8 m away
    distance = 8.0f;
    focal_length = 1600.0f;
    image_size = {960.0f, 320.0f};
    prior_weight = 1e-2f;
  } else if (body == "neuromechfly") {  // 100 mm away
    distance = 100.0f;
    focal_length = 13000.0f;
    image_size = {640.0f, 256.0f};
    prior_weight = 1e-3f;
  } else {
    std::fprintf(stderr, "Unknown body: %s\n", body.c_str());
    return 1;
  }
  auto assets = kExamplesDir / "assets" / body;

  auto output = run_ik_from_camera_obs(
      assets / "body_plan.json", assets / "example_trajectory.json", distance,
      focal_length, image_size, prior_weight);

  // Compare with the recorded 3D positions, which the solver never saw.
  std::printf(
      "Fitted %zu frames in %.1f ms, mean keypoint error vs. 3D recording: "
      "%.4f %s\n",
      output.results->len(), output.solve_time * 1e3, output.mean_error(),
      output.trajectory.units.c_str());
  save_output("from_cameras", body, output);
}
