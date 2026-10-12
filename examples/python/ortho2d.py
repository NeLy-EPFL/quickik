# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "quickik"]
#
# [tool.uv.sources]
# quickik = { path = "../../python" }
# ///
"""Example fitting DOF angles from observations in 2D orthographic view.

Bodies:
  neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
  g1: Unitree G1 robot (https://www.unitree.com/g1/)

Input data:
  Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
  Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
  (only XY positions used)

To run this example (from anywhere in the repository):
  uv run examples/python/ortho2d.py <body>
where <body> is one of "neuromechfly" or "g1". Results will be saved to
examples/python/output/ortho2d_<body>.json

To plot the results, run
  uv run examples/plot_results.py examples/python/output/ortho2d_<body>.json
and a video will be saved under the same directory. Run the `from_3d` example
first to also show its fit for reference.
"""

import sys
import time
from pathlib import Path

import numpy as np
import quickik

# Data utilities for the specific demos
from util import CameraView, IkOutput, Trajectory, save_output

IMAGE_SIZE = (640.0, 480.0)


def run_ik_from_ortho2d_obs(
    body_plan_path, trajectory_path, scale, prior_weight, on_ground
):
    # ===== Step 1: Load the body plan ========================================
    tree = quickik.KinematicTree.from_json_file(body_plan_path)

    # ===== Step 2: Load the motion trajectory ================================
    # This is the part that you would replace with your own data. For these
    # examples, utilities to load the trajectory, quantify errors, and save results
    # are provided in the util module.
    trajectory = Trajectory.load(trajectory_path)
    joint_indices = trajectory.joint_indices(tree)
    # We emulate 2D observations by dropping the Z coordinate (orthographic
    # projection). In real applications, these would come directly from a 2D pose
    # model.
    view = emulate_bottom_view(trajectory, scale)
    # Make a list of frames, where each frame is a list of KeypointObservations,
    # each carrying the projection it was observed in. Pixel errors are `scale`
    # times larger than position errors, so the weight is scaled down to match from_3d.
    weight = 1 / scale**2
    camera = view.projection
    frames = [
        [
            quickik.KeypointObservation.position_2d(joint, uv, camera, weight)
            for joint, uv in zip(joint_indices, uvs, strict=True)
        ]
        for uvs in view.observed
    ]

    # ===== Step 3: Configure the solver ======================================
    # With height unobserved, the neutral pose is the main thing keeping each
    # keypoint at a plausible height, so the neutral prior is stronger than in from_3d.
    config = quickik.SolverConfig()
    config.add_dof_neutral_prior(2 * prior_weight)  # favor natural poses
    config.add_dof_smoothness_prior(prior_weight)  # favor smooth motion
    # The body's absolute height can't be recovered, so it stays at the initial height
    if on_ground:
        initial_state = neutral_pose_on_ground(tree)
    else:
        initial_state = quickik.State.neutral_pose(tree)

    # ===== Step 4: Solve =====================================================
    solver = quickik.SequentialSolver(tree, config, initial_state)
    sequence = quickik.ObservationSequence.from_lists(frames)
    start = time.perf_counter()
    # Set with_grad to False to skip gradient tracking
    # Set with_fk to True to have forward kinematics (fitted keypoint positions) in
    # the results
    results = solver.solve(sequence, with_grad=False, with_fk=True)
    solve_time = time.perf_counter() - start

    return IkOutput(tree, trajectory, joint_indices, results, solve_time, [view])


# ===== Main and helper functions below (unrelated to core QuickIK logic) =====


def emulate_bottom_view(trajectory, scale):
    """Emulates a camera below the body with its image axes along world X and Y,
    at `scale` pixels per unit length (`uv = scale * (x, y) + offset`), centered on
    the trajectory's mean position."""
    center = trajectory.positions.reshape(-1, 3).mean(axis=0)[:2]
    offset = np.array(IMAGE_SIZE) / 2 - scale * center
    projection = quickik.OrthographicViewXY(scale, offset)
    observed = np.array(
        [[projection.project(p)[0] for p in frame] for frame in trajectory.positions]
    )
    return CameraView("bottom view", IMAGE_SIZE, projection, observed)


def neutral_pose_on_ground(tree):
    """The neutral pose, lifted so that its lowest keypoint is on the ground
    (z = 0)."""
    state = quickik.State.neutral_pose(tree)
    workspace = quickik.ForwardKinematicsWorkspace(tree)
    quickik.forward_kinematics(workspace, state)
    state.root_pos = (0, 0, -workspace.keypoint_positions[:, 2].min())
    return state


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: ortho2d.py <neuromechfly|g1>")
    body = sys.argv[1]
    # Body-specific settings: image scale (pixels per unit length), prior weight
    # (depends on the units), and whether the recording has the body on the ground
    # at z = 0 (G1) rather than centered on its root (the fly's thorax).
    match body:
        case "g1":
            scale, prior_weight, on_ground = 130.0, 1e-2, True  # 130 px/m
        case "neuromechfly":
            scale, prior_weight, on_ground = 100.0, 1e-3, False  # 100 px/mm
        case _:
            sys.exit(f"Unknown body: {body}")
    assets = Path(__file__).resolve().parents[1] / "assets" / body

    output = run_ik_from_ortho2d_obs(
        assets / "body_plan.json",
        assets / "example_trajectory.json",
        scale,
        prior_weight,
        on_ground,
    )

    # Compare with the recorded 3D positions, which the solver never saw.
    errors = output.errors()
    error_xy = np.linalg.norm(errors[..., :2], axis=-1).mean()
    error_z = np.abs(errors[..., 2]).mean()
    units = output.trajectory.units
    print(
        f"Fitted {len(output.results)} frames in {output.solve_time * 1e3:.1f} ms, "
        f"mean keypoint error vs. 3D recording: {error_xy:.4f} {units} in XY, "
        f"{error_z:.4f} {units} in Z"
    )
    save_output("ortho2d", body, output)


if __name__ == "__main__":
    main()
