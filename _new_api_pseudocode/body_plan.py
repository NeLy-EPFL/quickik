from dataclasses import dataclass, field
from enum import Enum

Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]

# Crate-private. DOFs of a free-floating root: 3 for position, 3 for rotation.
_N_ROOT_DOFS = 6


class DofType(Enum):
    HINGE = 0
    SLIDE = 1


@dataclass
class Dof:
    dof_type: DofType = DofType.HINGE
    # Rotation (hinge) or translation (slide) axis in the joint's local frame.
    # Normalized upon init (Rust: private field with getter/setter).
    axis: Vec3 = (0.0, 0.0, 1.0)
    # Neutral value (radians for hinges, length for slides). Also the target of
    # the DOF neutral prior.
    neutral: float = 0.0
    # Hard limits `(min, max)`, enforced by clamping after each solver step.
    limits: None | tuple[float, float] = None
    # Static multiplier on every DOF prior's weight (neutral and smoothness).
    weight_multiplier: float = 1.0


@dataclass
class Joint:
    name: str
    # Index of the parent joint in the list passed to `KinematicTree`, or
    # `None` for the root. The JSON format refers to parents by name instead.
    parent: None | int = None
    offset_pos: Vec3 = (0.0, 0.0, 0.0)
    offset_quat: QuatWxyz = (1.0, 0.0, 0.0, 0.0)
    # Applied in order (3D transformations don't commute).
    dofs: list[Dof] = field(default_factory=list)
    # Static multiplier on every observation's weight for this joint's keypoint.
    # Lower it for keypoints that are generally hard to detect.
    weight_multiplier: float = 1.0


class KinematicTree:
    """Immutable after construction. Shared by reference (Rust: `Arc`).

    Joint order is exactly the order of `joints` (the JSON order). This is the
    only joint order the user ever sees: observation joint indices, DOF order in
    `State.dof_values`, and Jacobian columns all follow it. Joints may be listed
    in any order; a parent-before-child traversal order is computed internally.
    """

    def __init__(self, joints: list[Joint], fixed_base: bool = False):
        """Validates the tree (exactly one root, valid parent indices, no cycles,
        unique names) and raises (Rust: returns `Err`) otherwise.

        `fixed_base`: the root is bolted to the world and has no DOFs of its
        own, same as today.
        """
        self.joints = joints
        self.fixed_base = fixed_base
        self.root_idx: int = ...  # the joint with `parent is None`
        self.n_joints = len(joints)
        self.n_dofs = sum(len(j.dofs) for j in joints)
        self.n_root_dofs = 0 if fixed_base else _N_ROOT_DOFS
        self.state_dim = self.n_root_dofs + self.n_dofs

        # Offset of each joint's first DOF in `State.dof_values`.
        self.joint_dof_start_indices: list[int] = ...

        # Crate-private static caches computed once here, read by FK and the
        # solver.
        # Parent-before-child order, e.g. BFS from the root. Equal to
        # `range(n_joints)` if `joints` is already listed parent-first.
        self._traversal_order: list[int] = ...
        # Per joint: the state-vector indices that can move its keypoint (root
        # DOFs, if any, plus every DOF on its ancestor joints), ascending. This
        # is fixed by topology, so it is no longer recomputed on every FK call.
        self._upstream_state_indices: list[list[int]] = ...

    @property
    def joint_names(self) -> list[str]: ...

    def joint_index(self, name: str) -> int:
        """Raises (Rust: `None`) if no joint has this name."""

    @classmethod
    def from_json_str(cls, json_str: str) -> "KinematicTree":
        """Raises (Rust: returns `Err`) on malformed JSON or an invalid tree."""

    @classmethod
    def from_json_file(cls, path: str) -> "KinematicTree": ...
