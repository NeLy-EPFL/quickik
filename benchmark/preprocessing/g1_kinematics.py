"""Forward kinematics for a QuickIK body-plan JSON, matching `src/forward.rs`'s
exact algorithm (see `evaluate_frame_at_joint`): for each joint in parent-
before-child order,

    origin = parent.origin + parent.rotation.apply(offset_pos)
    rotation = parent.rotation * offset_quat
    for each dof (own local axis, applied in order):
        rotation = rotation * axis_angle(axis, angle)   # re-orients children
                                                         # only, doesn't move
                                                         # this node's own
                                                         # origin (see
                                                         # forward.rs's header
                                                         # comment)

Used to generate G1's synthetic exact-fit frames (from self-sampled angles)
-- see `lafan1_bvh.py` for the raw motion capture's own from-scratch FK,
used to turn *its* joint angles into `target_ego` keypoint positions.

This is a from-scratch reimplementation (does not call into QuickIK), same
role as ../quickik_python/bench.py's own `forward_kinematics` cross-check
against the fly's exported body plan.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_scipy(wxyz):
    """QuickIK's body-plan quaternion convention ([w, x, y, z]) -> scipy's
    ([x, y, z, w])."""
    w, x, y, z = wxyz
    return R.from_quat([x, y, z, w])


class G1Kinematics:
    """Precomputes parent-before-child traversal order and per-node DOF
    offsets from a G1 body-plan dict (see `g1_body_plan.py`), then evaluates
    world-frame positions for any (root_pos, root_rot, dof_angles)."""

    def __init__(self, body_plan):
        joints = body_plan["joints"]
        self.names = [j["name"] for j in joints]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.parent_idx = [
            self.index[j["parent"]] if j["parent"] is not None else -1 for j in joints
        ]
        self.offset_pos = [np.array(j["offset_pos"]) for j in joints]
        self.offset_rot = [quat_wxyz_to_scipy(j["offset_quat"]) for j in joints]
        # Each node has 0 or 1 dof in this body plan; dof_index[i] is this
        # node's flat index into a 29-long dof_angles array, or None for the
        # root and the zero-dof leaf keypoints (head/hands).
        self.dof_axis = []
        self.dof_index = []
        dof_cursor = 0
        for j in joints:
            if j["dofs"]:
                assert len(j["dofs"]) == 1, (
                    "G1's body plan has exactly one DOF per non-leaf node"
                )
                self.dof_axis.append(np.array(j["dofs"][0]["axis"]))
                self.dof_index.append(dof_cursor)
                dof_cursor += 1
            else:
                self.dof_axis.append(None)
                self.dof_index.append(None)
        self.n_dofs = dof_cursor

    def evaluate(self, root_pos, root_rot: R, dof_angles):
        """Returns {node_name: world_position (3,)} for every node, including
        the root itself."""
        origins = [None] * len(self.names)
        rotations: list[R] = [None] * len(self.names)
        positions = {}

        for i, name in enumerate(self.names):
            if self.parent_idx[i] == -1:
                parent_origin, parent_rot = root_pos, root_rot
            else:
                parent_origin, parent_rot = (
                    origins[self.parent_idx[i]],
                    rotations[self.parent_idx[i]],
                )

            origin = parent_origin + parent_rot.apply(self.offset_pos[i])
            rotation = parent_rot * self.offset_rot[i]
            if self.dof_axis[i] is not None:
                angle = dof_angles[self.dof_index[i]]
                rotation = rotation * R.from_rotvec(self.dof_axis[i] * angle)

            origins[i] = origin
            rotations[i] = rotation
            positions[name] = origin

        return positions
