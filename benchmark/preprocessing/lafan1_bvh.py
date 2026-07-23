"""A minimal BVH parser and forward-kinematics evaluator for the raw LAFAN1
motion capture skeleton (22 joints: Hips root, leg/spine/head/arm chains --
see `../assets/g1_raw/README.md` for where the file comes from). From-scratch
reimplementation, same role as `g1_kinematics.py`'s own G1 FK.

BVH conventions this assumes (true of every LAFAN1 clip): the root's own 6
channels directly give its per-frame world position/rotation (its static
OFFSET is vestigial and ignored, per standard BVH practice); every other
joint's channels are 3 rotations applied in the file's own listed order as
intrinsic axis rotations (e.g. "Zrotation Yrotation Xrotation" ->
R = Rz(z) @ Ry(y) @ Rx(x)); translation channels (root only) are listed
X, Y, Z regardless of the rotation channels' order.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

_AXIS_LETTERS = {"X": "X", "Y": "Y", "Z": "Z"}


class BvhJoint:
    def __init__(self, name, offset, channels, parent):
        self.name = name
        self.offset = np.array(offset)
        self.channels = channels  # e.g. ["Zrotation", "Yrotation", "Xrotation"]
        self.parent = parent
        self.children = []


def parse_bvh(path):
    """Returns (joints: dict[name, BvhJoint] in depth-first declaration order,
    motion: (n_frames, n_channels) array, frame_time: float)."""
    with open(path) as f:
        lines = [line.strip() for line in f]

    joints = {}
    order = []
    stack = []  # BvhJoint currently being built, one per open "{"
    i = 0
    while lines[i] != "MOTION":
        tokens = lines[i].split()
        if tokens[:1] in (["ROOT"], ["JOINT"]):
            name = tokens[1]
            parent = stack[-1].name if stack else None
            joint = BvhJoint(name, offset=[0, 0, 0], channels=[], parent=parent)
            joints[name] = joint
            order.append(name)
            if parent is not None:
                joints[parent].children.append(name)
            stack.append(joint)
        elif tokens[:2] == ["End", "Site"]:
            stack.append(None)  # placeholder so its matching "}" pops cleanly
        elif tokens[:1] == ["OFFSET"] and stack and stack[-1] is not None:
            stack[-1].offset = np.array([float(v) for v in tokens[1:]])
        elif tokens[:1] == ["CHANNELS"]:
            stack[-1].channels = tokens[2:]
        elif tokens[:1] == ["}"]:
            stack.pop()
        i += 1

    frame_time = float(lines[i + 2].split()[-1])
    motion = np.array([[float(v) for v in line.split()] for line in lines[i + 3 :] if line])
    return {name: joints[name] for name in order}, motion, frame_time


def _local_rotation(channel_names, values_deg):
    """Composes the 3 rotation channels in the order BVH lists them, as
    intrinsic axis rotations (scipy's uppercase-axis convention matches
    this directly)."""
    axes = "".join(_AXIS_LETTERS[c[0]] for c in channel_names if c.endswith("rotation"))
    angles = [v for c, v in zip(channel_names, values_deg, strict=True) if c.endswith("rotation")]
    return R.from_euler(axes, angles, degrees=True)


class Lafan1Skeleton:
    """Precomputes each joint's channel-column offsets into the flat motion
    array, then evaluates world-frame positions/rotations for any frame."""

    def __init__(self, joints):
        self.joints = joints
        self.names = list(joints)
        col = 0
        self.channel_cols = {}
        for name, j in joints.items():
            self.channel_cols[name] = (col, j.channels)
            col += len(j.channels)

    def evaluate(self, motion_row):
        """Returns {joint_name: (world_pos (3,), world_rot: R)}."""
        out = {}
        for name in self.names:
            j = self.joints[name]
            start, channels = self.channel_cols[name]
            values = motion_row[start : start + len(channels)]

            if j.parent is None:
                pos_by_axis = {
                    c[0]: v for c, v in zip(channels, values, strict=True) if c.endswith("position")
                }
                world_pos = np.array([pos_by_axis["X"], pos_by_axis["Y"], pos_by_axis["Z"]])
                world_rot = _local_rotation(channels, values)
            else:
                parent_pos, parent_rot = out[j.parent]
                world_pos = parent_pos + parent_rot.apply(j.offset)
                world_rot = parent_rot * _local_rotation(channels, values)

            out[name] = (world_pos, world_rot)
        return out
