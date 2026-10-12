# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "quickik"]
#
# [tool.uv.sources]
# quickik = { path = "../../python" }
# ///
"""Example fitting DOF angles from keypoint observations in 3D.

Bodies:
  neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
  g1: Unitree G1 robot (https://www.unitree.com/g1/)

Input data:
  Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
  Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json

To run this example (from anywhere in the repository):
  uv run examples/python/from_3d.py <body>
where <body> is one of "neuromechfly" or "g1". Results will be saved to
examples/python/output/from_3d_<body>.json

To plot the results, run
  uv run examples/plot_results.py examples/python/output/from_3d_<body>.json
and a video will be saved under the same directory.
"""

import sys
import time
from pathlib import Path

import quickik

# Data utilities for the specific demos
from util import IkOutput, Trajectory, save_output


def run_ik_from_3d_obs(body_plan_path, trajectory_path, prior_weight):
    # ===== Step 1: Load the body plan ========================================
    tree = quickik.KinematicTree.from_json_file(body_plan_path)

    # ===== Step 2: Load the motion trajectory ================================
    # This is the part that you would replace with your own data. For these
    # examples, utilities to load the trajectory, quantify errors, and save results
    # are provided in the util module.
    trajectory = Trajectory.load(trajectory_path)
    joint_indices = trajectory.joint_indices(tree)
    # Make a list of frames, where each frame is a list of KeypointObservations.
    # Keypoint positions are given in 3D, so we use KeypointObservation.position_3d.
    frames = [
        [
            quickik.KeypointObservation.position_3d(joint, pos, weight=1.0)
            for joint, pos in zip(joint_indices, positions, strict=True)
        ]
        for positions in trajectory.positions
    ]

    # ===== Step 3: Configure the solver ======================================
    config = quickik.SolverConfig()
    config.add_dof_neutral_prior(prior_weight)  # favor natural poses
    config.add_dof_smoothness_prior(prior_weight)  # favor smooth motion

    # ===== Step 4: Solve =====================================================
    solver = quickik.SequentialSolver(tree, config)
    sequence = quickik.ObservationSequence.from_lists(frames)
    start = time.perf_counter()
    # Set with_grad=False to skip gradient tracking
    # Set with_fk=True to have forward kinematics (fitted keypoint positions) in results
    results = solver.solve(sequence, with_grad=False, with_fk=True)
    solve_time = time.perf_counter() - start

    return IkOutput(tree, trajectory, joint_indices, results, solve_time)


# ===== Main and helper functions below (unrelated to core QuickIK logic) =====


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: from_3d.py <neuromechfly|g1>")
    body = sys.argv[1]
    # Due to body size (and unit) differences, we assign different prior weights
    # for the two bodies.
    match body:
        case "g1":
            prior_weight = 1e-2
        case "neuromechfly":
            prior_weight = 1e-3
        case _:
            sys.exit(f"Unknown body: {body}")
    assets = Path(__file__).resolve().parents[1] / "assets" / body

    output = run_ik_from_3d_obs(
        assets / "body_plan.json", assets / "example_trajectory.json", prior_weight
    )

    print(
        f"Fitted {len(output.results)} frames in {output.solve_time * 1e3:.1f} ms, "
        f"mean keypoint error {output.mean_error():.4f} {output.trajectory.units}"
    )
    save_output("from_3d", body, output)


if __name__ == "__main__":
    main()
