//! cxx bridge for the QuickIK C++ bindings. Mirrors the Rust API where
//! reasonable; the main departure (as in `python/src/lib.rs`) is the mapper:
//! Rust's `Solver<M>`/`SequenceSolver<M>`/`BatchedSolver<M>` are generic over
//! the mapper type at compile time, but there's no C++ equivalent without
//! templating the whole binding, so every solver here is backed by a single
//! runtime `Mapper` value (`NoMapper`, `Camera`, or `XYView`) fixed at
//! construction.
//!
//! A second departure: sequences of per-frame (or per-batch-item) keypoint
//! observations are passed as one flat `observations` slice of length
//! `n_joints * n_frames` (frame `i` spanning `[i * n_joints, (i + 1) *
//! n_joints)`) rather than a nested container, and `SequenceSolver::solve`/
//! `solve_segments_parallel` return a `SolverResultList` (an indexable
//! handle, `len()`/`at(i)`) rather than a `Vec<SolverResult>`; cxx doesn't
//! support nested `Vec<Vec<T>>` or `Vec` of a non-trivial shared struct
//! across the bridge. For the same reason, `SolverResult`/
//! `BatchedSolverResult`'s `keypoint_pos`/`jacobian`/`cholesky_l` fields are
//! flattened `Vec<f32>` (row-major for the matrices) rather than nested
//! arrays; `has_keypoint_pos`/`has_jacobian`/`has_cholesky_l` flags stand in
//! for Rust's `Option` (empty/`false` when that piece wasn't requested, or,
//! for `cholesky_l`, when the linearization wasn't positive-definite).
//!
//! See `README.md` (top level) for build instructions and a usage example.

use std::sync::Arc;

use nalgebra::DMatrix;
use quickik_core::observation::Mapper3Dto2D;

#[allow(clippy::too_many_arguments)]
#[cxx::bridge(namespace = "quickik")]
mod ffi {
    /// Which kind of observation a `KeypointObservation` carries. cxx has no
    /// fielded enums, so the payload lives in `KeypointObservation`'s
    /// `pos`/`weight` fields instead of enum variants like Rust's
    /// `KeypointObservation`.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum ObservationKind {
        Missing,
        Position3D,
        Position2D,
    }

    /// A single keypoint observation. For `Position2D`, only `pos[0]`/
    /// `pos[1]` are used. Construct via `keypoint_missing`/
    /// `keypoint_position_3d`/`keypoint_position_2d` rather than setting
    /// fields directly.
    #[derive(Clone, Copy, Debug)]
    struct KeypointObservation {
        kind: ObservationKind,
        pos: [f32; 3],
        weight: f32,
    }

    /// A pinhole camera for inverse kinematics from 2D keypoint observations.
    #[derive(Clone, Copy, Debug)]
    struct Camera {
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        world2cam_pos: [f32; 3],
        /// Row-major 3x3.
        world2cam_rot_mat: [f32; 9],
    }

    /// Which mapper a `Mapper` value holds: no mapper, a `Camera`, or an
    /// X-Y view of world coordinates. See `Mapper`'s docs.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum MapperKind {
        NoMapper,
        CameraMapper,
        XYViewMapper,
    }

    /// Runtime stand-in for Rust's generic mapper type parameter `M`.
    /// `camera` is only meaningful when `kind == CameraMapper`. Construct via
    /// `no_mapper`/`camera_mapper`/`xyview_mapper`.
    #[derive(Clone, Copy, Debug)]
    struct Mapper {
        kind: MapperKind,
        camera: Camera,
    }

    /// The converged pose (and, optionally, linearization) from one
    /// `Solver::solve` call, or one item of a `SolverResultList`.
    struct SolverResult {
        /// Angles of all joint DOFs, in the `KinematicTree`'s own order.
        dof_angles: Vec<f32>,
        /// Position of the root joint in world coordinates.
        root_pos: [f32; 3],
        /// Rotation of the root joint in world coordinates, as
        /// `(w, x, y, z)`.
        root_rot: [f32; 4],
        /// World-space keypoint positions, flattened (`n_joints * 3` long, 3
        /// floats per keypoint), in the `KinematicTree`'s joint order. Empty
        /// unless `has_keypoint_pos`.
        keypoint_pos: Vec<f32>,
        /// Whether `solve` was called with `with_fk = true`.
        has_keypoint_pos: bool,
        /// The keypoint-position Jacobian at (approximately) the converged
        /// pose, flattened row-major (`(3 * n_joints) * state_dim` long; see
        /// `KinematicTree::state_dim`). Empty unless `has_jacobian`.
        jacobian: Vec<f32>,
        /// Whether `solve` was called with `with_grad = true`.
        has_jacobian: bool,
        /// Lower-triangular Cholesky factor `L` of the normal-equations
        /// matrix at the same linearization as `jacobian` (`jtj = L @
        /// L^T`), flattened row-major (`state_dim * state_dim` long). Empty
        /// unless `has_cholesky_l`.
        cholesky_l: Vec<f32>,
        /// Whether `with_grad` was `true` *and* that linearization's normal
        /// equations were positive-definite; gradients can't be computed
        /// from this solve otherwise.
        has_cholesky_l: bool,
    }

    /// Every `BatchedSolver::solve` item's converged pose and (optional)
    /// linearization, as flattened, batch-major arrays (see this module's
    /// top-level docs). `batch_size` is however many items were passed to
    /// `solve`; `n_dofs`/`n_joints`/`state_dim` come from the
    /// `KinematicTree` `BatchedSolver` was constructed with.
    struct BatchedSolverResult {
        /// Flattened `batch_size * n_dofs`.
        joint_angles: Vec<f32>,
        /// Flattened `batch_size * 3`.
        base_pos: Vec<f32>,
        /// Flattened `batch_size * 4` (`w, x, y, z` per item).
        base_quat: Vec<f32>,
        /// Flattened `batch_size * n_joints * 3`, in the `KinematicTree`'s
        /// internal joint order (*not* `keypoints_order`). Empty unless
        /// `has_keypoint_pos`.
        keypoint_pos: Vec<f32>,
        /// Whether `solve` was called with `with_fk = true`.
        has_keypoint_pos: bool,
        /// Flattened `batch_size * (3 * n_joints) * state_dim`, row-major
        /// per item, in the `KinematicTree`'s internal keypoint/state order
        /// (*not* `keypoints_order`). Empty unless `has_jacobian`.
        jacobian: Vec<f32>,
        /// Whether `solve` was called with `with_grad = true`.
        has_jacobian: bool,
        /// Flattened `batch_size * state_dim * state_dim`, row-major per
        /// item; zeroed for any item whose `valid` entry is `false`. Empty
        /// unless `has_cholesky_l`.
        cholesky_l: Vec<f32>,
        /// Whether `with_grad` was `true` (`cholesky_l`/`valid` are only
        /// meaningful then).
        has_cholesky_l: bool,
        /// Length `batch_size`; `false` where that item's last iteration
        /// wasn't positive-definite, so its `cholesky_l` block can't be used
        /// for gradients. Empty unless `has_cholesky_l`.
        valid: Vec<bool>,
    }

    extern "Rust" {
        /// A keypoint not observed this frame (e.g. occluded).
        fn keypoint_missing() -> KeypointObservation;
        /// A 3D world position, e.g. triangulated from multiple calibrated
        /// cameras.
        fn keypoint_position_3d(pos: [f32; 3], weight: f32) -> KeypointObservation;
        /// A 2D pixel position from the camera (or other mapper) that the
        /// consuming solver was constructed with.
        fn keypoint_position_2d(pos: [f32; 2], weight: f32) -> KeypointObservation;

        /// A `Mapper` for solvers that receive 3D keypoint observations only.
        fn no_mapper() -> Mapper;
        /// A `Mapper` that projects with the given pinhole `camera`.
        fn camera_mapper(camera: Camera) -> Mapper;
        /// A `Mapper` that takes a 3D keypoint's world X/Y coordinates as its
        /// 2D projection.
        fn xyview_mapper() -> Mapper;

        /// A kinematic tree, i.e. body plan, or skeleton.
        type KinematicTree;
        /// Parses a `KinematicTree` from a JSON body-plan string.
        fn kinematic_tree_from_json_str(json_str: &str) -> Result<Box<KinematicTree>>;
        /// Parses a `KinematicTree` from a JSON body-plan file at `path`.
        fn kinematic_tree_from_json_file(path: &str) -> Result<Box<KinematicTree>>;
        /// Number of joints in the tree.
        fn n_joints(self: &KinematicTree) -> usize;
        /// Total number of rotational degrees of freedom across all joints.
        fn n_dofs(self: &KinematicTree) -> usize;
        /// Dimensionality of the flattened solver state (`n_dofs` plus 6 for
        /// the free-floating root, or just `n_dofs` for a fixed-base tree).
        /// The column count of `SolverResult::jacobian`/`BatchedSolverResult::jacobian`,
        /// and the row/column count of `cholesky_l`.
        fn state_dim(self: &KinematicTree) -> usize;

        /// The pose being solved for.
        type State;
        /// A `State` for `tree` at its neutral pose: every DOF at its
        /// neutral angle, root at the origin with identity rotation.
        fn state_neutral_pose(tree: &KinematicTree) -> Box<State>;
        /// Angles of all joint DOFs.
        fn dof_angles(self: &State) -> Vec<f32>;
        /// Position of the root joint in world coordinates.
        fn root_pos(self: &State) -> [f32; 3];
        /// Rotation of the root joint in world coordinates, as
        /// `(w, x, y, z)`.
        fn root_rot(self: &State) -> [f32; 4];

        /// A list of `SolverResult`s returned by `SequenceSolver::solve`/
        /// `solve_segments_parallel`.
        type SolverResultList;
        /// Number of results in the list.
        fn len(self: &SolverResultList) -> usize;
        /// The result at index `i`. Raises an exception (rather than
        /// aborting the process) if `i >= len()`.
        fn at(self: &SolverResultList, i: usize) -> Result<SolverResult>;

        /// The inverse kinematics solver, backed by a single `Mapper` fixed
        /// at construction (see this module's top-level docs).
        type Solver;
        /// Constructs a `Solver` for `tree` with the given `mapper` and
        /// tuning parameters.
        fn new_solver(
            tree: &KinematicTree,
            mapper: Mapper,
            n_iterations: usize,
            neutral_weight: f32,
            position_tolerance: f32,
            angle_tolerance: f32,
            damping: f32,
        ) -> Box<Solver>;
        /// Runs up to `n_iterations` Gauss-Newton steps in place on `state`,
        /// given one observation per joint (some may be `Missing`), and
        /// returns the converged pose. `with_grad`/`with_fk` gate
        /// `SolverResult::jacobian`/`cholesky_l` and `keypoint_pos`
        /// respectively; each costs a little extra work, so only request
        /// what you'll use. Panics from the underlying solve (e.g. a
        /// `Position2D` observation given to a mapper-less solver) are
        /// caught and raised as an exception rather than aborting the
        /// process.
        fn solve(
            self: &mut Solver,
            state: &mut State,
            observations: &[KeypointObservation],
            with_grad: bool,
            with_fk: bool,
        ) -> Result<SolverResult>;
        /// Fixed at construction; there is no setter.
        fn mapper(self: &Solver) -> Mapper;
        /// Number of Gauss-Newton steps per `solve` call. Also the cap on
        /// early termination: see `position_tolerance`/`angle_tolerance`.
        fn n_iterations(self: &Solver) -> usize;
        fn set_n_iterations(self: &mut Solver, value: usize);
        /// Weight pulling every joint angle toward the neutral pose.
        fn neutral_weight(self: &Solver) -> f32;
        fn set_neutral_weight(self: &mut Solver, value: f32);
        /// Stop iterating early once an update step's largest root-position
        /// component drops below this value, and the largest angle update
        /// drops below `angle_tolerance`. 0 disables early termination.
        fn position_tolerance(self: &Solver) -> f32;
        fn set_position_tolerance(self: &mut Solver, value: f32);
        /// Angle-space counterpart to `position_tolerance`, in radians.
        fn angle_tolerance(self: &Solver) -> f32;
        fn set_angle_tolerance(self: &mut Solver, value: f32);
        /// Levenberg-Marquardt damping added to the normal equations'
        /// diagonal, for numerical stability only; keep it very small
        /// (e.g. 1e-6).
        fn damping(self: &Solver) -> f32;
        fn set_damping(self: &mut Solver, value: f32);

        /// Warm-started solving for a continuous sequence of frames, backed
        /// by a single `Mapper` fixed at construction. `solve` always
        /// continues from wherever the previous call left off, for this
        /// object's whole lifetime; `solve_segments_parallel` is unrelated
        /// to that continuity (a self-contained bulk operation that never
        /// reads or writes it). Unlike `Solver`, the tuning parameters
        /// aren't retunable after construction.
        type SequenceSolver;
        /// Starts a new sequence at the neutral pose, for `tree`, with the
        /// given `mapper` and tuning parameters.
        fn new_sequence_solver(
            tree: &KinematicTree,
            mapper: Mapper,
            n_iterations: usize,
            neutral_weight: f32,
            position_tolerance: f32,
            angle_tolerance: f32,
            damping: f32,
        ) -> Box<SequenceSolver>;
        /// Solves every frame in order, each warm-started from wherever this
        /// object's last `solve`/`solve_segments_parallel` call left off.
        /// `observations` is flattened: `n_joints * n_frames` long, frame
        /// `i` spanning `[i * n_joints, (i + 1) * n_joints)`. See
        /// `Solver::solve`'s docs for `with_grad`/`with_fk` and for panics
        /// being raised as exceptions.
        fn solve(
            self: &mut SequenceSolver,
            observations: &[KeypointObservation],
            n_joints: usize,
            with_grad: bool,
            with_fk: bool,
        ) -> Result<Box<SolverResultList>>;
        /// Solves `observations` in parallel by splitting them into exactly
        /// `n_workers` contiguous, non-overlapping segments, each
        /// cold-started at the neutral pose and then warm-started within
        /// itself. Never reads or writes this object's own `solve` state.
        /// `n_workers`: a positive value is used directly, unless it
        /// exceeds the number of available cores: it's then clipped to that
        /// count and a warning is logged. A negative value counts backward
        /// from all available cores: `-1` uses all, `-2` uses all but one,
        /// etc. `0` is invalid.
        fn solve_segments_parallel(
            self: &SequenceSolver,
            observations: &[KeypointObservation],
            n_joints: usize,
            n_workers: isize,
            with_grad: bool,
            with_fk: bool,
        ) -> Result<Box<SolverResultList>>;
        /// Fixed at construction; there is no setter.
        fn mapper(self: &SequenceSolver) -> Mapper;

        /// Solves a batch of fully independent (never warm-started) sets of
        /// keypoint observations in parallel, for training/inference with
        /// an autodiff framework, backed by a single `Mapper` fixed at
        /// construction.
        type BatchedSolver;
        /// `tree` must be free-floating (not fixed-base). `keypoints_order[i]`
        /// is the joint name that `solve`'s `observations` keypoint axis
        /// position `i` corresponds to; every joint in `tree` must appear in
        /// it exactly once. `n_workers` follows the same convention as
        /// `SequenceSolver::solve_segments_parallel`'s. Raises an exception
        /// if `tree` is fixed-base, `keypoints_order` is malformed, or
        /// `n_workers` is `0`.
        fn new_batched_solver(
            tree: &KinematicTree,
            mapper: Mapper,
            n_iterations: usize,
            neutral_weight: f32,
            position_tolerance: f32,
            angle_tolerance: f32,
            damping: f32,
            keypoints_order: Vec<String>,
            n_workers: isize,
        ) -> Result<Box<BatchedSolver>>;
        /// Solves every item in `observations` independently and in
        /// parallel, each starting from `tree`'s neutral pose. `observations`
        /// is flattened: `n_joints * batch_size` long, item `i` spanning
        /// `[i * n_joints, (i + 1) * n_joints)`, in this `BatchedSolver`'s
        /// own `keypoints_order` (*not* the `KinematicTree`'s internal joint
        /// order). See `Solver::solve`'s docs for `with_grad`/`with_fk` and
        /// for panics being raised as exceptions.
        fn solve(
            self: &BatchedSolver,
            observations: &[KeypointObservation],
            n_joints: usize,
            with_grad: bool,
            with_fk: bool,
        ) -> Result<BatchedSolverResult>;
        /// Fixed at construction; there is no setter.
        fn mapper(self: &BatchedSolver) -> Mapper;
        /// `keypoint_to_joint_idx()[i]` is the `KinematicTree`'s internal
        /// joint index that `solve`'s keypoint axis position `i` corresponds
        /// to (the resolved inverse of the by-name `keypoints_order` this
        /// solver was constructed with).
        fn keypoint_to_joint_idx(self: &BatchedSolver) -> Vec<usize>;
    }
}

// =============================================================================
//  KeypointObservation / Camera / Mapper: construction and conversion to/from
//  the core crate's own types.
// =============================================================================

fn keypoint_missing() -> ffi::KeypointObservation {
    ffi::KeypointObservation {
        kind: ffi::ObservationKind::Missing,
        pos: [0.0; 3],
        weight: 0.0,
    }
}

fn keypoint_position_3d(pos: [f32; 3], weight: f32) -> ffi::KeypointObservation {
    ffi::KeypointObservation {
        kind: ffi::ObservationKind::Position3D,
        pos,
        weight,
    }
}

fn keypoint_position_2d(pos: [f32; 2], weight: f32) -> ffi::KeypointObservation {
    ffi::KeypointObservation {
        kind: ffi::ObservationKind::Position2D,
        pos: [pos[0], pos[1], 0.0],
        weight,
    }
}

fn to_core_observation(
    obs: &ffi::KeypointObservation,
) -> quickik_core::observation::KeypointObservation {
    match obs.kind {
        ffi::ObservationKind::Missing => quickik_core::observation::KeypointObservation::Missing,
        ffi::ObservationKind::Position3D => {
            quickik_core::observation::KeypointObservation::Position3D {
                obs_pos: nalgebra::Vector3::new(obs.pos[0], obs.pos[1], obs.pos[2]),
                weight: obs.weight,
            }
        }
        ffi::ObservationKind::Position2D => {
            quickik_core::observation::KeypointObservation::Position2D {
                obs_pos: nalgebra::Vector2::new(obs.pos[0], obs.pos[1]),
                weight: obs.weight,
            }
        }
        _ => unreachable!("unknown ObservationKind"),
    }
}

/// Runtime stand-in for Rust's generic mapper type parameter `M`, mirroring
/// `python/src/observation.rs`'s `Mapper`. Unlike Rust, where "no mapper" is
/// a distinct compile-time type (`quickik_core::observation::NoMapper`),
/// every solver here always instantiates the same concrete
/// `Solver<RuntimeMapper>` (etc.), so `None` has to be one more runtime
/// variant of this same enum rather than a separate type.
#[derive(Clone, Copy, Debug)]
enum RuntimeMapper {
    None,
    Camera(quickik_core::observation::Camera),
    XYView,
}

impl Mapper3Dto2D for RuntimeMapper {
    fn project_3d_to_2d<S1, S2>(
        &self,
        pos_world3d: &nalgebra::Vector3<f32>,
        jacobian_world3d: &nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Dyn, S1>,
        jacobian_2d_out: &mut nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Dyn, S2>,
    ) -> nalgebra::Vector2<f32>
    where
        S1: nalgebra::Storage<f32, nalgebra::Dyn, nalgebra::Dyn>,
        S2: nalgebra::StorageMut<f32, nalgebra::Dyn, nalgebra::Dyn>,
    {
        match self {
            // Mirrors NoMapper::project_3d_to_2d's own panic.
            RuntimeMapper::None => unreachable!(
                "a Solver/SequenceSolver/BatchedSolver constructed with no_mapper() was given a \
                 Position2D observation"
            ),
            RuntimeMapper::Camera(camera) => {
                camera.project_3d_to_2d(pos_world3d, jacobian_world3d, jacobian_2d_out)
            }
            RuntimeMapper::XYView => quickik_core::observation::XYView.project_3d_to_2d(
                pos_world3d,
                jacobian_world3d,
                jacobian_2d_out,
            ),
        }
    }
}

fn to_core_camera(camera: &ffi::Camera) -> quickik_core::observation::Camera {
    quickik_core::observation::Camera {
        fx: camera.fx,
        fy: camera.fy,
        cx: camera.cx,
        cy: camera.cy,
        world2cam_pos: nalgebra::Vector3::from(camera.world2cam_pos),
        world2cam_rot_mat: nalgebra::Matrix3::from_row_slice(&camera.world2cam_rot_mat),
    }
}

fn to_runtime_mapper(mapper: &ffi::Mapper) -> RuntimeMapper {
    match mapper.kind {
        ffi::MapperKind::NoMapper => RuntimeMapper::None,
        ffi::MapperKind::CameraMapper => RuntimeMapper::Camera(to_core_camera(&mapper.camera)),
        ffi::MapperKind::XYViewMapper => RuntimeMapper::XYView,
        _ => unreachable!("unknown MapperKind"),
    }
}

fn no_mapper() -> ffi::Mapper {
    ffi::Mapper {
        kind: ffi::MapperKind::NoMapper,
        camera: ffi::Camera {
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            world2cam_pos: [0.0; 3],
            world2cam_rot_mat: [0.0; 9],
        },
    }
}

fn camera_mapper(camera: ffi::Camera) -> ffi::Mapper {
    ffi::Mapper {
        kind: ffi::MapperKind::CameraMapper,
        camera,
    }
}

fn xyview_mapper() -> ffi::Mapper {
    ffi::Mapper {
        kind: ffi::MapperKind::XYViewMapper,
        camera: ffi::Camera {
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            world2cam_pos: [0.0; 3],
            world2cam_rot_mat: [0.0; 9],
        },
    }
}

fn runtime_mapper_to_ffi(mapper: RuntimeMapper) -> ffi::Mapper {
    match mapper {
        RuntimeMapper::None => no_mapper(),
        RuntimeMapper::Camera(camera) => {
            let p = camera.world2cam_pos;
            camera_mapper(ffi::Camera {
                fx: camera.fx,
                fy: camera.fy,
                cx: camera.cx,
                cy: camera.cy,
                world2cam_pos: [p.x, p.y, p.z],
                world2cam_rot_mat: camera
                    .world2cam_rot_mat
                    .transpose()
                    .as_slice()
                    .try_into()
                    .unwrap(),
            })
        }
        RuntimeMapper::XYView => xyview_mapper(),
    }
}

// =============================================================================
//  KinematicTree
// =============================================================================

struct KinematicTree(Arc<quickik_core::body_plan::KinematicTree>);

/// Runs `f`, converting a panic (e.g. from malformed JSON, or a `Position2D`
/// observation given to a mapper-less solver) into an `Err`. In a plain
/// (non-`Result`) bridged function, an unwinding panic would instead abort
/// the whole process. Every mutation `f` might
/// have made before panicking is just plain data with no unsafe invariants
/// to uphold, so asserting unwind-safety here is fine.
fn catch_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic")
            .to_string()
    })
}

fn kinematic_tree_from_json_str(json_str: &str) -> Result<Box<KinematicTree>, String> {
    catch_panic(|| {
        KinematicTree(Arc::new(
            quickik_core::body_plan::KinematicTree::from_json_str(json_str),
        ))
    })
    .map(Box::new)
}

fn kinematic_tree_from_json_file(path: &str) -> Result<Box<KinematicTree>, String> {
    catch_panic(|| {
        KinematicTree(Arc::new(
            quickik_core::body_plan::KinematicTree::from_json_file(path),
        ))
    })
    .map(Box::new)
}

impl KinematicTree {
    fn n_joints(&self) -> usize {
        self.0.n_joints()
    }
    fn n_dofs(&self) -> usize {
        self.0.n_dofs()
    }
    fn state_dim(&self) -> usize {
        self.0.state_dim()
    }
}

// =============================================================================
//  State
// =============================================================================

struct State(quickik_core::state::State);

fn state_neutral_pose(tree: &KinematicTree) -> Box<State> {
    Box::new(State(quickik_core::state::State::neutral_pose(
        tree.0.clone(),
    )))
}

impl State {
    fn dof_angles(&self) -> Vec<f32> {
        self.0.dof_angles.clone()
    }
    fn root_pos(&self) -> [f32; 3] {
        let p = self.0.root_pos;
        [p.x, p.y, p.z]
    }
    fn root_rot(&self) -> [f32; 4] {
        let q = self.0.root_rot.quaternion();
        [q.w, q.i, q.j, q.k]
    }
}

/// Splits a flat, `n_joints * n_frames`-long slice into per-frame chunks and
/// converts each observation to the core crate's type. See this module's
/// top-level docs for why sequences (and batches) are passed flattened.
fn split_into_frames(
    flat: &[ffi::KeypointObservation],
    n_joints: usize,
) -> Vec<Vec<quickik_core::observation::KeypointObservation>> {
    assert_eq!(
        flat.len() % n_joints,
        0,
        "observations length must be a multiple of n_joints"
    );
    flat.chunks(n_joints)
        .map(|frame| frame.iter().map(to_core_observation).collect())
        .collect()
}

/// Flattens world-space keypoint positions into `n_joints * 3` floats (3 per
/// keypoint), matching how sequences/batches of observations are flattened
/// elsewhere in this bridge (see this module's top-level docs).
fn flatten_positions(positions: &[nalgebra::Vector3<f32>]) -> Vec<f32> {
    positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect()
}

/// Flattens a matrix row-major (row 0 first, then row 1, etc.), matching how
/// `Jacobian`/`Cholesky` factors are flattened elsewhere in this bridge.
fn flatten_matrix_row_major(mat: &DMatrix<f32>) -> Vec<f32> {
    let mut out = Vec::with_capacity(mat.nrows() * mat.ncols());
    for r in 0..mat.nrows() {
        for c in 0..mat.ncols() {
            out.push(mat[(r, c)]);
        }
    }
    out
}

/// Converts a core `SolverResult` into the bridge's flattened
/// `ffi::SolverResult`.
fn core_result_to_ffi(result: &quickik_core::solver::SolverResult) -> ffi::SolverResult {
    let root_pos = result.state.root_pos;
    let root_rot = result.state.root_rot.quaternion();
    ffi::SolverResult {
        dof_angles: result.state.dof_angles.clone(),
        root_pos: [root_pos.x, root_pos.y, root_pos.z],
        root_rot: [root_rot.w, root_rot.i, root_rot.j, root_rot.k],
        keypoint_pos: result
            .keypoint_pos
            .as_deref()
            .map(flatten_positions)
            .unwrap_or_default(),
        has_keypoint_pos: result.keypoint_pos.is_some(),
        jacobian: result
            .jacobian
            .as_ref()
            .map(flatten_matrix_row_major)
            .unwrap_or_default(),
        has_jacobian: result.jacobian.is_some(),
        cholesky_l: result
            .cholesky_l
            .as_ref()
            .map(|chol| flatten_matrix_row_major(&chol.l()))
            .unwrap_or_default(),
        has_cholesky_l: result.cholesky_l.is_some(),
    }
}

// =============================================================================
//  SolverResultList
// =============================================================================

struct SolverResultList(Vec<quickik_core::solver::SolverResult>);

impl SolverResultList {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn at(&self, i: usize) -> Result<ffi::SolverResult, String> {
        let results = &self.0;
        catch_panic(move || core_result_to_ffi(&results[i]))
    }
}

// =============================================================================
//  Solver
// =============================================================================

struct Solver {
    inner: quickik_core::solver::Solver<RuntimeMapper>,
    mapper: RuntimeMapper,
}

#[allow(clippy::too_many_arguments)]
fn new_solver(
    tree: &KinematicTree,
    mapper: ffi::Mapper,
    n_iterations: usize,
    neutral_weight: f32,
    position_tolerance: f32,
    angle_tolerance: f32,
    damping: f32,
) -> Box<Solver> {
    let mapper = to_runtime_mapper(&mapper);
    Box::new(Solver {
        inner: quickik_core::solver::Solver::new(
            &tree.0,
            mapper,
            n_iterations,
            neutral_weight,
            position_tolerance,
            angle_tolerance,
            damping,
        ),
        mapper,
    })
}

impl Solver {
    fn solve(
        &mut self,
        state: &mut State,
        observations: &[ffi::KeypointObservation],
        with_grad: bool,
        with_fk: bool,
    ) -> Result<ffi::SolverResult, String> {
        let observations: Vec<_> = observations.iter().map(to_core_observation).collect();
        let inner = &mut self.inner;
        let state = &mut state.0;
        catch_panic(move || {
            core_result_to_ffi(&inner.solve(state, &observations, with_grad, with_fk))
        })
    }
    fn mapper(&self) -> ffi::Mapper {
        runtime_mapper_to_ffi(self.mapper)
    }
    fn n_iterations(&self) -> usize {
        self.inner.n_iterations
    }
    fn set_n_iterations(&mut self, value: usize) {
        self.inner.n_iterations = value;
    }
    fn neutral_weight(&self) -> f32 {
        self.inner.neutral_weight
    }
    fn set_neutral_weight(&mut self, value: f32) {
        self.inner.neutral_weight = value;
    }
    fn position_tolerance(&self) -> f32 {
        self.inner.position_tolerance
    }
    fn set_position_tolerance(&mut self, value: f32) {
        self.inner.position_tolerance = value;
    }
    fn angle_tolerance(&self) -> f32 {
        self.inner.angle_tolerance
    }
    fn set_angle_tolerance(&mut self, value: f32) {
        self.inner.angle_tolerance = value;
    }
    fn damping(&self) -> f32 {
        self.inner.damping
    }
    fn set_damping(&mut self, value: f32) {
        self.inner.damping = value;
    }
}

// =============================================================================
//  SequenceSolver
// =============================================================================

struct SequenceSolver {
    inner: quickik_core::sequential_solver::SequenceSolver<RuntimeMapper>,
    mapper: RuntimeMapper,
}

#[allow(clippy::too_many_arguments)]
fn new_sequence_solver(
    tree: &KinematicTree,
    mapper: ffi::Mapper,
    n_iterations: usize,
    neutral_weight: f32,
    position_tolerance: f32,
    angle_tolerance: f32,
    damping: f32,
) -> Box<SequenceSolver> {
    let mapper = to_runtime_mapper(&mapper);
    Box::new(SequenceSolver {
        inner: quickik_core::sequential_solver::SequenceSolver::new(
            &tree.0,
            mapper,
            n_iterations,
            neutral_weight,
            position_tolerance,
            angle_tolerance,
            damping,
        ),
        mapper,
    })
}

impl SequenceSolver {
    fn solve(
        &mut self,
        observations: &[ffi::KeypointObservation],
        n_joints: usize,
        with_grad: bool,
        with_fk: bool,
    ) -> Result<Box<SolverResultList>, String> {
        let inner = &mut self.inner;
        catch_panic(move || {
            let sequence = split_into_frames(observations, n_joints);
            Box::new(SolverResultList(inner.solve(&sequence, with_grad, with_fk)))
        })
    }
    fn solve_segments_parallel(
        &self,
        observations: &[ffi::KeypointObservation],
        n_joints: usize,
        n_workers: isize,
        with_grad: bool,
        with_fk: bool,
    ) -> Result<Box<SolverResultList>, String> {
        let inner = &self.inner;
        catch_panic(move || {
            let sequence = split_into_frames(observations, n_joints);
            Box::new(SolverResultList(inner.solve_segments_parallel(
                &sequence, n_workers, with_grad, with_fk,
            )))
        })
    }
    fn mapper(&self) -> ffi::Mapper {
        runtime_mapper_to_ffi(self.mapper)
    }
}

// =============================================================================
//  BatchedSolver
// =============================================================================

struct BatchedSolver {
    inner: quickik_core::batched_solver::BatchedSolver<RuntimeMapper>,
    mapper: RuntimeMapper,
}

#[allow(clippy::too_many_arguments)]
fn new_batched_solver(
    tree: &KinematicTree,
    mapper: ffi::Mapper,
    n_iterations: usize,
    neutral_weight: f32,
    position_tolerance: f32,
    angle_tolerance: f32,
    damping: f32,
    keypoints_order: Vec<String>,
    n_workers: isize,
) -> Result<Box<BatchedSolver>, String> {
    let mapper = to_runtime_mapper(&mapper);
    catch_panic(move || {
        Box::new(BatchedSolver {
            inner: quickik_core::batched_solver::BatchedSolver::new(
                &tree.0,
                mapper,
                n_iterations,
                neutral_weight,
                position_tolerance,
                angle_tolerance,
                damping,
                keypoints_order,
                n_workers,
            ),
            mapper,
        })
    })
}

/// Converts a core `BatchedSolverResult` into the bridge's flattened,
/// batch-major `ffi::BatchedSolverResult`.
fn batched_result_to_ffi(
    result: &quickik_core::batched_solver::BatchedSolverResult,
) -> ffi::BatchedSolverResult {
    let batch_size = result.joint_angles.len();

    let mut joint_angles = Vec::new();
    let mut base_pos = Vec::new();
    let mut base_quat = Vec::new();
    for i in 0..batch_size {
        joint_angles.extend_from_slice(&result.joint_angles[i]);
        let p = result.base_pos[i];
        base_pos.extend_from_slice(&[p.x, p.y, p.z]);
        let q = result.base_quat[i].quaternion();
        base_quat.extend_from_slice(&[q.w, q.i, q.j, q.k]);
    }

    let (keypoint_pos, has_keypoint_pos) = match &result.keypoint_pos {
        Some(batch_positions) => {
            let mut flat = Vec::new();
            for positions in batch_positions {
                flat.extend(flatten_positions(positions));
            }
            (flat, true)
        }
        None => (Vec::new(), false),
    };

    let (jacobian, has_jacobian) = match &result.jacobian {
        Some(batch_jacobians) => {
            let mut flat = Vec::new();
            for jac in batch_jacobians {
                flat.extend(flatten_matrix_row_major(jac));
            }
            (flat, true)
        }
        None => (Vec::new(), false),
    };

    let (cholesky_l, valid, has_cholesky_l) = match &result.cholesky_l {
        Some(batch_cholesky) => {
            // Every item shares the same state_dim as the jacobian
            // (with_grad gates both the same way), so invalid items can be
            // zero-filled to that same width.
            let state_dim = result
                .jacobian
                .as_ref()
                .and_then(|jacs| jacs.first())
                .map_or(0, |jac| jac.ncols());
            let mut flat = Vec::with_capacity(batch_size * state_dim * state_dim);
            let mut valid_flags = Vec::with_capacity(batch_size);
            for chol in batch_cholesky {
                match chol {
                    Some(chol) => {
                        flat.extend(flatten_matrix_row_major(&chol.l()));
                        valid_flags.push(true);
                    }
                    None => {
                        flat.extend(std::iter::repeat_n(0.0, state_dim * state_dim));
                        valid_flags.push(false);
                    }
                }
            }
            (flat, valid_flags, true)
        }
        None => (Vec::new(), Vec::new(), false),
    };

    ffi::BatchedSolverResult {
        joint_angles,
        base_pos,
        base_quat,
        keypoint_pos,
        has_keypoint_pos,
        jacobian,
        has_jacobian,
        cholesky_l,
        has_cholesky_l,
        valid,
    }
}

impl BatchedSolver {
    fn solve(
        &self,
        observations: &[ffi::KeypointObservation],
        n_joints: usize,
        with_grad: bool,
        with_fk: bool,
    ) -> Result<ffi::BatchedSolverResult, String> {
        let inner = &self.inner;
        catch_panic(move || {
            let observations_array = split_into_frames(observations, n_joints);
            batched_result_to_ffi(&inner.solve(&observations_array, with_grad, with_fk))
        })
    }
    fn mapper(&self) -> ffi::Mapper {
        runtime_mapper_to_ffi(self.mapper)
    }
    fn keypoint_to_joint_idx(&self) -> Vec<usize> {
        self.inner.keypoint_to_joint_idx().to_vec()
    }
}
