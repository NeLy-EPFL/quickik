# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "quickik"]
#
# [tool.uv.sources]
# quickik = { path = "../../python" }
# ///
"""Example fitting DOF angles from 2D keypoint observations from two cameras.

Bodies:
  neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
  g1: Unitree G1 robot (https://www.unitree.com/g1/)

Input data:
  Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
  Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
  (2D observations are emulated by projecting the 3D positions into two pinhole
  cameras)

To run this example (from anywhere in the repository):
  uv run examples/python/from_cameras.py <body>
where <body> is one of "neuromechfly" or "g1". Results will be saved to
examples/python/output/from_cameras_<body>.json

To plot the results, run
  uv run examples/plot_results.py examples/python/output/from_cameras_<body>.json
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


def run_ik_from_camera_obs(
    body_plan_path, trajectory_path, distance, focal_length, image_size, prior_weight
):
    # ===== Step 1: Load the body plan ========================================
    tree = quickik.KinematicTree.from_json_file(body_plan_path)

    # ===== Step 2: Load the motion trajectory ================================
    # This is the part that you would replace with your own data. For these
    # examples, utilities to load the trajectory, quantify errors, and save results
    # are provided in the util module.
    trajectory = Trajectory.load(trajectory_path)
    joint_indices = trajectory.joint_indices(tree)
    # We emulate 2D observations by projecting 3D positions into two cameras. In
    # real applications, these would come directly from a 2D pose model.
    views = emulate_two_camera_views(trajectory, distance, focal_length, image_size)
    # Make a list of frames, where each frame is a list of KeypointObservations.
    # Each frame's observations are one flat list mixing both cameras, each
    # observation carrying the projection (camera) it was observed in. Pixel errors
    # are about focal_length / distance times larger than position errors, so the
    # weight is scaled down to match from_3d.
    weight = (distance / focal_length) ** 2
    frames = [
        [
            quickik.KeypointObservation.position_2d(joint, uv, view.projection, weight)
            for view in views
            for joint, uv in zip(joint_indices, view.observed[t], strict=True)
        ]
        for t in range(len(trajectory.positions))
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

    return IkOutput(tree, trajectory, joint_indices, results, solve_time, views)


# ===== Main and helper functions below (unrelated to core QuickIK logic) =====


def emulate_two_camera_views(trajectory, distance, focal_length, image_size):
    """Emulates two cameras, 45 degrees to the front left and front right of the
    body (which faces +x), at `distance` from the trajectory's average position and
    looking horizontally at it."""
    target = trajectory.positions.reshape(-1, 3).mean(axis=0)
    views = []
    for name, azimuth_deg in [("left camera", 45.0), ("right camera", -45.0)]:
        camera = make_pinhole_camera(
            target, distance, azimuth_deg, focal_length, image_size
        )
        observed = np.array(
            [[camera.project(p)[0] for p in frame] for frame in trajectory.positions]
        )
        views.append(CameraView(name, image_size, camera, observed))
    return views


def make_pinhole_camera(target, distance, azimuth_deg, focal_length, image_size):
    """A pinhole camera `distance` away from `target` at the same height, at the
    given azimuth (counterclockwise from +x, seen from above), facing `target` with
    it at the image center."""
    azimuth = np.radians(azimuth_deg)
    direction = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    position = target + distance * direction

    # Camera axes in world coordinates: z forward (toward the target), x to the
    # right in the image, y down in the image.
    z = -direction
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    world2cam_rot_mat = np.array([x, y, z])
    world2cam_pos = -(world2cam_rot_mat @ position)
    width, height = image_size
    cx, cy = width / 2, height / 2
    return quickik.PinholeCamera(
        focal_length, focal_length, cx, cy, world2cam_pos, world2cam_rot_mat
    )


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: from_cameras.py <neuromechfly|g1>")
    body = sys.argv[1]
    # Body-specific settings: camera distance, focal length and image size in pixels
    # (chosen so the whole recording just fits in the image), and prior weight
    # (depends on the units).
    match body:
        case "g1":  # 8 m away
            distance, focal_length, image_size = 8.0, 1600.0, (960.0, 320.0)
            prior_weight = 1e-2
        case "neuromechfly":  # 100 mm away
            distance, focal_length, image_size = 100.0, 13000.0, (640.0, 256.0)
            prior_weight = 1e-3
        case _:
            sys.exit(f"Unknown body: {body}")
    assets = Path(__file__).resolve().parents[1] / "assets" / body

    output = run_ik_from_camera_obs(
        assets / "body_plan.json",
        assets / "example_trajectory.json",
        distance,
        focal_length,
        image_size,
        prior_weight,
    )

    # Compare with the recorded 3D positions, which the solver never saw.
    print(
        f"Fitted {len(output.results)} frames in {output.solve_time * 1e3:.1f} ms, "
        f"mean keypoint error vs. 3D recording: {output.mean_error():.4f} "
        f"{output.trajectory.units}"
    )
    save_output("from_cameras", body, output)


if __name__ == "__main__":
    main()
