"""Derives each body's `example_trajectory.json` from the benchmark fixtures.

Sources (see their generators for how they were recorded and processed):

- NeuroMechFly: `native_rate_frames` of `benchmark/assets/fixtures.json`
  (`benchmark/scripts/generate_fixtures.py`), 300 consecutive frames at 330 Hz
  of a real walking fly, in mm.
- G1: `native_rate_frames` of `benchmark/assets/fixtures_g1.json`
  (`benchmark/preprocessing/g1_fixtures.py`), 300 consecutive frames at 30 Hz
  of LAFAN1 human walking motion capture rescaled onto G1's proportions, in m.

Positions are rounded to 3 decimal places. The fly's are relative to its thorax
(the root of its body plan), as in the fixtures. For G1, three further changes
make the data easier to use:

- The wrist keypoints are dropped: the motion capture has a single hand marker,
  which the fixtures copy onto the 3 wrist joints and the hand alike, so only
  the hand's copy is kept.
- The fixtures' frame has a tilted vertical axis (see `render_video.py` in
  `benchmark/plot`), so the trajectory is rotated to make up +z and the overall
  walking direction +x.
- It's shifted so the robot stands on the ground (lowest point at z = 0), with
  the mean hip position at x = y = 0.

Usage: python examples/assets/make_example_trajectories.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_DIR = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_DIR / "benchmark" / "assets"
ASSETS_DIR = Path(__file__).resolve().parent
G1_DROPPED_JOINTS = {
    f"{side}_wrist_{axis}"
    for side in ("left", "right")
    for axis in ("roll", "pitch", "yaw")
}


def load_fixtures(name):
    fixtures = json.loads((FIXTURES_DIR / name).read_text())
    positions = np.array([f["target_ego"] for f in fixtures["native_rate_frames"]])
    return fixtures["leg_joint_names"], positions


def upright_g1(joint_names, positions):
    """Rotates and shifts G1's trajectory as described in the module docstring."""
    idx = {name: i for i, name in enumerate(joint_names)}
    ankles = positions[:, [idx["left_ankle_roll"], idx["right_ankle_roll"]]].mean(1)
    up = (positions[:, idx["head"]] - ankles).mean(0)
    to_upright, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])
    positions = to_upright.apply(positions.reshape(-1, 3)).reshape(positions.shape)

    hips = positions[:, [idx["left_hip_pitch"], idx["right_hip_pitch"]]].mean(1)
    walk = hips[-1] - hips[0]
    heading = Rotation.from_rotvec([0.0, 0.0, -np.arctan2(walk[1], walk[0])])
    positions = heading.apply(positions.reshape(-1, 3)).reshape(positions.shape)

    hips = heading.apply(hips)
    return positions - np.array([*hips.mean(0)[:2], positions[..., 2].min()])


def write(body, description, units, fps, joint_names, positions):
    trajectory = {
        "description": description,
        "units": units,
        "fps": fps,
        "joint_names": joint_names,
        "positions": np.round(positions, 3).tolist(),
    }
    path = ASSETS_DIR / body / "example_trajectory.json"
    path.write_text(json.dumps(trajectory) + "\n")
    print(f"Wrote {positions.shape[0]} frames x {len(joint_names)} keypoints to {path}")


def main():
    joint_names, positions = load_fixtures("fixtures.json")
    write(
        "neuromechfly",
        "Real fly walking, 3D motion capture (recording 20250613-fly1b-012).",
        "mm",
        330.0,
        joint_names,
        positions,
    )

    joint_names, positions = load_fixtures("fixtures_g1.json")
    positions = upright_g1(joint_names, positions)
    keep = [i for i, name in enumerate(joint_names) if name not in G1_DROPPED_JOINTS]
    write(
        "g1",
        "Human walking (LAFAN1 walk1_subject1) rescaled onto G1's proportions.",
        "m",
        30.0,
        [joint_names[i] for i in keep],
        positions[:, keep],
    )


if __name__ == "__main__":
    main()
