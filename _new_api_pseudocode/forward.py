"""Forward kinematics with per-keypoint Jacobians.

Public in all three languages, for users who only need FK (e.g. to render a
known pose); the solver uses the same code. The traversal is a flat loop over
the tree's precomputed parent-before-child order instead of a recursive DFS, so
there are no children lists and no push/truncate stack of DOF frames.
"""

from .body_plan import KinematicTree
from .state import State


class ForwardKinematicsWorkspace:
    """Buffers allocated once per tree and reused across `forward_kinematics`
    calls."""

    def __init__(self, kinematic_tree: KinematicTree):
        n_joints = kinematic_tree.n_joints
        n_dofs = kinematic_tree.n_dofs

        # Public output (read-only): (n_joints, 3), world frame, joint order.
        self.keypoint_positions = ...

        # Crate-private. Compact per-keypoint Jacobians: keypoint k's is a
        # `3 x len(tree._upstream_state_indices[k])` block whose column i is the
        # derivative w.r.t. state index `tree._upstream_state_indices[k][i]`.
        # Rust: one flat buffer plus per-joint offsets. Replaces the dense
        # `3 * n_joints x state_dim` matrix, which was zero-filled every call and
        # then gathered from by the solver. Users get the dense form from
        # `jacobian()`.
        self._keypoint_jacobians = [...] * n_joints

        # Scratch, indexed by joint.
        self._joint_frames = [...] * n_joints  # (origin, rotation) incl. own DOFs
        # Scratch, indexed by flat DOF index (not a stack anymore).
        self._dof_axes_world = [...] * n_dofs
        self._dof_origins_world = [...] * n_dofs  # only read for hinges

    def jacobian(self):
        """Dense `(3 * n_joints, state_dim)` Jacobian of `keypoint_positions`
        w.r.t. the state, scattered from the compact blocks on demand. Only
        valid if the last `forward_kinematics` call had `with_jacobian=True`."""


def forward_kinematics(
    workspace: ForwardKinematicsWorkspace, state: State, with_jacobian: bool = True
):
    """Fills `workspace.keypoint_positions` and, if `with_jacobian`, the
    per-keypoint Jacobians. The solver always passes `True`.

    Pseudo-implementation, for clarity of the new traversal only:
    """
    tree = state.kinematic_tree
    for j in tree._traversal_order:
        joint = tree.joints[j]
        parent_frame = (
            (state.root_pos, state.root_quat)
            if joint.parent is None
            else workspace._joint_frames[joint.parent]
        )
        # Same math as today's `evaluate_frame_at_joint`: the keypoint sits at
        # the joint origin before its own DOFs are applied; own DOFs only move
        # descendants. Records world axis/origin of each own DOF into
        # `_dof_axes_world`/`_dof_origins_world` at its flat DOF index.
        keypoint_pos, frame = _evaluate_frame_at_joint(
            workspace, state, j, parent_frame
        )
        workspace._joint_frames[j] = frame
        workspace.keypoint_positions[j] = keypoint_pos
        # Writes column i for each `tree._upstream_state_indices[j][i]`: root
        # translation/rotation columns, then `axis x (p - origin)` for hinges or
        # `axis` for slides. Every upstream DOF is already recorded because
        # parents are visited first.
        if with_jacobian:
            _fill_keypoint_jacobian(workspace, state, j, keypoint_pos)


def _evaluate_frame_at_joint(workspace, state, joint_idx, parent_frame): ...


def _fill_keypoint_jacobian(workspace, state, joint_idx, keypoint_pos): ...
