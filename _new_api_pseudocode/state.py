from dataclasses import dataclass

from .body_plan import KinematicTree, QuatWxyz, Vec3


@dataclass
class State:
    kinematic_tree: KinematicTree  # Rust: `Arc<KinematicTree>`
    # Ignored (kept at identity) for a fixed-base tree.
    root_pos: Vec3
    root_quat: QuatWxyz
    # One value per DOF, in the tree's joint order then each joint's DOF order.
    dof_values: list[float]

    @classmethod
    def neutral_pose(cls, kinematic_tree: KinematicTree) -> "State":
        """Every DOF at `Dof.neutral`, root at the origin with identity
        rotation."""

    # Crate-internal (Rust: `pub(crate)`), not part of the public API. Applies a
    # Gauss-Newton step and clamps DOFs to their limits.
    def _apply_delta(self, delta): ...
