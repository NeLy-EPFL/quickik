"""Solver configuration (including priors) and the three solvers.

All three solvers share one Gauss-Newton core and one `SolverConfig`. Every
method that can fail raises in Python/C++ and returns `Result` in Rust (no more
panics caught at the FFI boundary).

Not shown: crate-private helpers (the internal `Prior` enum, the Gauss-Newton
loop, Jacobian snapshotting, the normal-equations solve, the convergence check,
`resolve_n_workers`, chunk-bound computation). `SolverResult` and
`BatchedSolverResult` fields are read-only.
"""

from dataclasses import dataclass

import numpy as np

from .body_plan import KinematicTree, Vec3
from .observation import KeypointObservation, ObservationSequence
from .state import State

# =============================================================================
#  Configuration
# =============================================================================


class SolverConfig:
    """Iteration settings (fixed at init) plus priors (added with `add_*`).

    Priors are added with one method per prior type, spelled the same in all
    three languages (Rust `cfg.add_x(..)`, Python `cfg.add_x(..)`, C++
    `cfg->add_x(..)`); each type may be added at most once (raises otherwise).
    No priors by default. Note that today's solvers enable the neutral prior by
    default (`neutral_weight=1e-3`).

    Copied into each solver on construction and fixed for that solver's
    lifetime. Per-frame confidence goes into observation weights instead.
    Validated when a solver is constructed: root priors on a fixed-base tree,
    and smoothness priors in `SingleFrameSolver` or `BatchedSolver`, raise.
    """

    def __init__(
        self,
        n_iterations: int = 10,
        position_tolerance: float = 1e-3,
        angle_tolerance: float = 1e-3,
        damping: float = 1e-6,
    ):
        """Same meaning and defaults as today. Read-only afterwards."""

    def add_dof_neutral_prior(self, weight: float):
        """Pulls each DOF toward `Dof.neutral`, with per-DOF weight
        `weight * Dof.weight_multiplier`. Replaces today's `neutral_weight`."""

    def add_root_upright_prior(self, weight: float, body_up: Vec3, world_up: Vec3):
        """Pulls the root's local `body_up` axis toward `world_up` (e.g. +z, or a
        tilted vector on a slope). Leaves yaw about `world_up` free. Both
        vectors are normalized."""

    def add_dof_smoothness_prior(self, weight: float):
        """Pulls each DOF toward its value in the previous frame, with per-DOF
        weight `weight * Dof.weight_multiplier`. Inactive on a sequence's first
        frame."""

    def add_root_smoothness_prior(self, position_weight: float, rotation_weight: float):
        """Pulls the root position and rotation toward the previous frame's.
        Separate weights because their units differ (length vs radians).
        Inactive on a sequence's first frame."""


# =============================================================================
#  Results
# =============================================================================


@dataclass
class SolverResult:
    """One frame's result. Same fields as today."""

    state: State
    # (n_joints, 3), world frame, at the returned `state`. `None` unless
    # `with_fk`.
    keypoint_pos: None | np.ndarray
    # (n_rows, state_dim): one row block per observation, in the frame's input
    # order, 3 rows for `POSITION_3D` and 2 for `POSITION_2D` (already mapped
    # through that observation's projection). Weight-0 observations get rows
    # too. `None` unless `with_grad`.
    jacobian: None | np.ndarray
    # (state_dim, state_dim) lower-triangular factor of the normal-equations
    # matrix, priors included, at the same linearization. `None` unless
    # `with_grad` and that matrix was positive-definite.
    cholesky_l: None | np.ndarray


@dataclass
class BatchedSolverResult:
    """Struct of stacked arrays, as today (natural for torch). C++: flat
    row-major buffers."""

    dof_values: np.ndarray  # (batch, n_dofs)
    root_pos: np.ndarray  # (batch, 3)
    root_quat: np.ndarray  # (batch, 4), wxyz
    keypoint_pos: None | np.ndarray  # (batch, n_joints, 3)
    # (batch, n_rows, state_dim). Requires every item to have the same row
    # layout (always true for `ObservationSequence.from_arrays`); raises
    # otherwise.
    jacobian: None | np.ndarray
    cholesky_l: None | np.ndarray  # (batch, state_dim, state_dim), zero if invalid
    cholesky_valid: None | np.ndarray  # (batch,), bool


# =============================================================================
#  Solvers
# =============================================================================


class SingleFrameSolver:
    """One frame at a time, caller-managed state. Today's `Solver`."""

    def __init__(self, kinematic_tree: KinematicTree, config: SolverConfig):
        """Raises if `config` has smoothness priors (there is no previous
        frame), or root priors on a fixed-base tree."""

    def solve(
        self,
        state: State,
        observations: list[KeypointObservation],
        with_grad: bool = False,
        with_fk: bool = False,
    ) -> SolverResult:
        """Refines `state` in place, starting from its current value."""


class SequentialSolver:
    """Warm-started solving of one continuous stream, as today's
    `SequenceSolver`. Holds a single running `state` that `step` and `solve`
    advance; the previous frame's solution is both the warm start and the
    smoothness priors' target."""

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        config: SolverConfig,
        initial_state: None | State = None,  # default: neutral pose
    ): ...

    @property
    def state(self) -> State:
        """The running state after the last `step`/`solve` call."""

    def reset(self, state: None | State = None):
        """Starts a new stream: sets the running state (default: neutral pose)
        and forgets the previous frame, so smoothness priors are inactive on the
        next frame."""

    def step(
        self,
        observations: list[KeypointObservation],
        with_grad: bool = False,
        with_fk: bool = False,
    ) -> SolverResult:
        """Solves the next frame of the stream (real-time use)."""

    def solve(
        self,
        sequence: ObservationSequence,
        with_grad: bool = False,
        with_fk: bool = False,
    ) -> list[SolverResult]:
        """`step` for each frame in order."""

    def solve_parallel(
        self,
        sequence: ObservationSequence,
        n_workers: int = -1,
        n_overlap: int = 0,
        with_grad: bool = False,
        with_fk: bool = False,
    ) -> list[SolverResult]:
        """Offline: splits `sequence` into `n_workers` contiguous chunks solved
        in parallel. Never reads or writes the running state.

        Each chunk (except the first) starts at the neutral pose `n_overlap`
        frames before its own first frame and discards those warm-up results,
        so it arrives warm-started and with a previous frame for the smoothness
        priors. `n_overlap=0` gives today's behavior. The first chunk starts
        from the neutral pose. `n_workers` follows today's convention (negative
        counts back from all cores, 0 is invalid).
        """


class BatchedSolver:
    """Fully independent items solved in parallel, each from the neutral pose,
    for training with autodiff (see `quickik.torch`). As today, minus
    `keypoints_order`: reorder input arrays with `tree.joint_index` instead."""

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        config: SolverConfig,
        n_workers: int = -1,
    ):
        """Raises if `config` has smoothness priors or the tree is
        fixed-base."""

    def solve(
        self,
        batch: ObservationSequence,
        with_grad: bool = False,
        with_fk: bool = False,
    ) -> BatchedSolverResult:
        """Each item starts from the neutral pose."""
