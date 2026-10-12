"""Keypoint observations and the 2D projections they may carry.

The set of projections is closed (Rust: one `enum`); users can't subclass it.
Projections are immutable and shared by reference (Rust: `Arc<Projection>`), so
observations from the same camera point at one copy instead of each carrying
its own (a pinhole camera is 16 floats).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .body_plan import Vec3

Vec2 = tuple[float, float]
Mat2x3 = ...
Mat3 = ...


# =============================================================================
#  Projections
# =============================================================================


class _PositionProjection2D:
    """Maps a 3D world position to 2D. Rust: variants of one `Projection` enum,
    held as `Arc<Projection>`; the Python classes wrap that `Arc`. C++: a flat
    tagged struct passed by value, built by the constructor functions of each
    variant (as today)."""

    # Crate-internal. Returns `(pos_2d, d_pos2d_d_pos3d)`. The solver chains the
    # `2 x 3` derivative onto the keypoint's Jacobian itself, so a variant only
    # has to implement this.
    def _project(self, pos_3d: Vec3) -> tuple[Vec2, Mat2x3]: ...


class PinholeCamera(_PositionProjection2D):
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        world2cam_pos: Vec3,
        world2cam_rot_mat: Mat3,
    ):
        """`p_cam = world2cam_rot_mat @ p_world + world2cam_pos`, then
        `uv = (fx * x / z + cx, fy * y / z + cy)`, in pixels."""


class OrthographicViewXY(_PositionProjection2D):
    def __init__(self, scale: float = 1.0, offset: Vec2 = (0.0, 0.0)):
        """Drops world Z: `uv = scale * (x, y) + offset`."""


# =============================================================================
#  Observations
# =============================================================================


class ObservationType(Enum):
    POSITION_3D = 0
    POSITION_2D = 1


@dataclass(frozen=True)
class KeypointObservation:
    """One observation of one joint's keypoint.

    A frame's observations are a flat list in any order. A joint may appear any
    number of times (e.g. once per camera) or not at all (e.g. occluded):
    there is no MISSING type, just leave it out. An observation with weight 0
    contributes nothing to the solve but still gets its rows in
    `SolverResult.jacobian`.

    Only built with the named constructors below; fields are read-only (Rust:
    private fields with getters), so `projection` is set if and only if
    `obs_type` is `POSITION_2D`. C++ exception: this must be a cxx shared struct
    (so it can be passed in a slice), whose fields are always public; Rust
    therefore checks that consistency when an `ObservationSequence` or solver
    receives it, and raises otherwise.

    Rust: `Clone` but not `Copy`, since `projection` is an `Arc<Projection>`
    (about 32 bytes per observation instead of about 100 with the projection
    inline). Cloning only happens at construction; the solver only borrows, so
    the reference count is never touched in the solve loop. C++ holds the
    projection by value (no `Arc` across cxx); see `ObservationSequence`.
    """

    joint_idx: int
    obs_type: ObservationType
    pos: Vec3 | Vec2
    # Multiplied by `Joint.weight_multiplier` to give the final weight.
    weight: float
    projection: None | _PositionProjection2D

    @classmethod
    def position_3d(
        cls, joint_idx: int, pos: Vec3, weight: float = 1.0
    ) -> "KeypointObservation":
        """A 3D world position, e.g. triangulated from several cameras."""

    @classmethod
    def position_2d(
        cls,
        joint_idx: int,
        pos: Vec2,
        projection: _PositionProjection2D,
        weight: float = 1.0,
    ) -> "KeypointObservation":
        """A 2D position in the space `projection` maps into."""


class ObservationSequence:
    """Many frames of observations, owned on the Rust side and immutable.

    Input to `SequentialSolver.solve*` and `BatchedSolver.solve` (a frame is a
    batch item there). It exists for the bindings: Python avoids building one
    Python object per keypoint per frame (~2x slower, see
    `docs/technical/benchmarks.md`), and C++ avoids a nested `Vec`, which cxx
    can't pass. It also stores each distinct projection only once, which
    matters for long recordings (e.g. 100k frames x 30 keypoints).

    Crate-private layout:
    - `projections: Vec<Projection>`, one entry per distinct projection.
    - `observations`: flat records of about 24 bytes, `{joint_idx: u32, kind,
      proj_idx: u32, pos: [f32; 3], weight: f32}`, where `proj_idx` indexes
      into `projections` (unused for 3D).
    - `frame_offsets: Vec<usize>`: frame `i` is
      `observations[frame_offsets[i]..frame_offsets[i + 1]]`.
    Projections are deduplicated as described per constructor. Two equal but
    separately built projections that don't get merged are harmless: the math
    is the same either way.
    """

    @classmethod
    def from_lists(
        cls, observations: list[list[KeypointObservation]]
    ) -> "ObservationSequence":
        """One list per frame. Frames may have different observations (e.g.
        moving cameras). Not available in C++ (no nested `Vec`): use
        `from_flat`.

        Deduplicates projections by identity: a `HashMap` keyed on
        `Arc::as_ptr`. Only the address is hashed, so projections need not be
        hashable, and a new camera object per frame stays O(1) per
        observation."""

    @classmethod
    def from_flat(
        cls, observations: list[KeypointObservation], frame_lengths: list[int]
    ) -> "ObservationSequence":
        """Same as `from_lists`, with all frames concatenated: frame `i` is the
        next `frame_lengths[i]` observations. Mainly for C++.

        In C++ projections arrive by value, with no identity to compare, so they
        are deduplicated by value instead: a `HashMap` keyed on the bit patterns
        of their fields (`f32::to_bits`). Values that compare equal as floats but
        differ in bits (`-0.0` vs `0.0`, NaN) are simply not merged. Rust and
        Python callers get identity-based deduplication, as in `from_lists`."""

    @classmethod
    def from_arrays(
        cls,
        pos_3d: None | np.ndarray = None,  # (n_frames, n_joints, 3)
        weights_3d: None | np.ndarray = None,  # (n_frames, n_joints)
        projections: list[_PositionProjection2D] = (),  # (n_views,)
        pos_2d: None | np.ndarray = None,  # (n_frames, n_views, n_joints, 2)
        weights_2d: None | np.ndarray = None,  # (n_frames, n_views, n_joints)
    ) -> "ObservationSequence":
        """Builds every frame with a fixed layout: the 3D block (all joints, in
        tree order) first, then one 2D block per view in `projections` order.
        Every entry is kept, including weight 0 (or NaN, stored as weight 0 with
        a zeroed position), so every frame's Jacobian has the same rows:
        `3 * n_joints` (if 3D given) `+ 2 * n_joints * n_views`. The projections
        are the same for all frames; use `from_lists`/`from_flat` for moving
        cameras. `projections` is stored as-is (no deduplication needed). C++:
        the same, with flat row-major buffers plus sizes.
        """

    def __len__(self) -> int:
        """Number of frames."""
