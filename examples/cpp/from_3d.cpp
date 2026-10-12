// Example fitting DOF angles from keypoint observations in 3D.
//
// Bodies:
//   neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
//   g1: Unitree G1 robot (https://www.unitree.com/g1/)
//
// Input data:
//   Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
//   Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
//
// To run this example (from the repository root):
//   cmake -S examples/cpp -B examples/cpp/build -DCMAKE_BUILD_TYPE=Release
//   cmake --build examples/cpp/build -j
//   ./examples/cpp/build/from_3d <body>
// where <body> is one of "neuromechfly" or "g1". Results will be saved to
// examples/cpp/output/from_3d_<body>.json
//
// To plot the results, run
//   uv run examples/plot_results.py examples/cpp/output/from_3d_<body>.json
// and a video will be saved under the same directory.

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

#include "quickik.hpp"

// Data utilities for the specific demos
#include "util.hpp"

IkOutput run_ik_from_3d_obs(const std::filesystem::path& body_plan_path,
                            const std::filesystem::path& trajectory_path,
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
  // Make one list of KeypointObservations for all frames, and record how many
  // of them belong to each frame. Keypoint positions are given in 3D, so we use
  // add_position_3d.
  auto observations = quickik::KeypointObservationList::create();
  std::vector<size_t> frame_lengths;
  for (const auto& positions : trajectory.positions) {
    for (size_t i = 0; i < joint_indices.size(); i++) {
      observations->add_position_3d(joint_indices[i], positions[i], 1.0f);
    }
    frame_lengths.push_back(joint_indices.size());
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
          {}};
}

// ===== Main and helper functions below (unrelated to core QuickIK logic) =====

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: from_3d <neuromechfly|g1>\n");
    return 1;
  }
  std::string body = argv[1];
  // Due to body size (and unit) differences, we assign different prior weights
  // for the two bodies.
  float prior_weight;
  if (body == "g1") {
    prior_weight = 1e-2f;
  } else if (body == "neuromechfly") {
    prior_weight = 1e-3f;
  } else {
    std::fprintf(stderr, "Unknown body: %s\n", body.c_str());
    return 1;
  }
  auto assets = kExamplesDir / "assets" / body;

  auto output =
      run_ik_from_3d_obs(assets / "body_plan.json",
                         assets / "example_trajectory.json", prior_weight);

  std::printf("Fitted %zu frames in %.1f ms, mean keypoint error %.4f %s\n",
              output.results->len(), output.solve_time * 1e3,
              output.mean_error(), output.trajectory.units.c_str());
  save_output("from_3d", body, output);
}
