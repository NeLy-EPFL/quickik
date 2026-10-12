"""Helpers shared by the Python examples: loading the recorded trajectory, and
summarizing and saving results in the format `examples/plot_results.py` reads.
Nothing here is specific to QuickIK beyond reading its results.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import quickik

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class Trajectory:
    """A recorded keypoint trajectory, as stored in `example_trajectory.json`."""

    units: str
    fps: float
    joint_names: list[str]
    # `(n_frames, n_keypoints, 3)`: `positions[t, i]` is the position of keypoint
    # `joint_names[i]` in frame `t`.
    positions: np.ndarray

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        positions = np.array(data["positions"], dtype=np.float32)
        return cls(data["units"], data["fps"], data["joint_names"], positions)

    def joint_indices(self, tree):
        """The body plan's joint index for each recorded keypoint."""
        return [tree.joint_index(name) for name in self.joint_names]


@dataclass
class CameraView:
    """A camera's projection and what it observed."""

    name: str
    image_size: tuple[float, float]
    projection: quickik.Projection
    # `(n_frames, n_keypoints, 2)`: `observed[t, i]` is the observed pixel position
    # of the trajectory's i-th keypoint in frame `t`.
    observed: np.ndarray


@dataclass
class IkOutput:
    """Everything an example's IK run produces, for reporting and saving."""

    tree: quickik.KinematicTree
    trajectory: Trajectory
    # The body plan's joint index for each recorded keypoint.
    joint_indices: list[int]
    # One result per frame.
    results: list[quickik.SolverResult]
    # Time spent in the solver, in seconds.
    solve_time: float
    # The cameras the observations came from (empty for 3D observations).
    views: list[CameraView] = field(default_factory=list)

    def errors(self):
        """`(n_frames, n_keypoints, 3)`: fitted minus recorded position of every
        recorded keypoint in every frame."""
        fitted = np.array([result.keypoint_pos for result in self.results])
        return fitted[:, self.joint_indices] - self.trajectory.positions

    def mean_error(self):
        """Mean distance between fitted and recorded keypoints, over all frames."""
        return np.linalg.norm(self.errors(), axis=-1).mean()


def save_output(example, body, output):
    """Writes the inputs and results to `examples/python/output/<example>_<body>.json`
    in the format `examples/plot_results.py` reads (shared by all languages), with
    every number rounded to 3 decimal places."""
    states = [result.state for result in output.results]
    data = {
        "example": example,
        "body": body,
        "language": "python",
        "units": output.trajectory.units,
        "fps": output.trajectory.fps,
        "joint_names": output.tree.joint_names,
        "parents": [joint.parent for joint in output.tree.joints],
        "observations": {
            "joint_indices": output.joint_indices,
            "positions": output.trajectory.positions,
        },
        "fit": {
            "root_pos": [state.root_pos for state in states],
            "root_quat_wxyz": [state.root_quat for state in states],
            "dof_values": [state.dof_values for state in states],
            "keypoint_positions": [result.keypoint_pos for result in output.results],
        },
        "views": [view_json(view) for view in output.views],
        "solve_time_s": output.solve_time,
    }
    path = OUTPUT_DIR / f"{example}_{body}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_numbers(data), separators=(",", ":")))
    print(f"Saved {path}")


def view_json(view):
    projection = view.projection
    if isinstance(projection, quickik.PinholeCamera):
        projection_json = {
            "type": "pinhole_camera",
            "fx": projection.fx,
            "fy": projection.fy,
            "cx": projection.cx,
            "cy": projection.cy,
            "world2cam_pos": projection.world2cam_pos,
            "world2cam_rot_mat": projection.world2cam_rot_mat,
        }
    else:
        projection_json = {
            "type": "orthographic_view_xy",
            "scale": projection.scale,
            "offset": projection.offset,
        }
    return {
        "name": view.name,
        "image_size": view.image_size,
        "projection": projection_json,
        "observed": view.observed,
    }


def round_numbers(value):
    """`value` as plain JSON-serializable Python objects, with every non-integer
    number rounded to 3 decimal places."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [round_numbers(item) for item in value]
    if isinstance(value, float | np.floating):
        return round(float(value), 3)
    return value
