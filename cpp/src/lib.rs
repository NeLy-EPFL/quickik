//! cxx bridge for the QuickIK C++ bindings. Mirrors the Rust API where
//! reasonable; the main departure (as in `python/src/lib.rs`) is the mapper:
//! Rust's `Solver<M>` is generic over the mapper type at compile time, but
//! there's no C++ equivalent without templating the whole binding, so every
//! `Solver`/`SequenceSolver` here is backed by a single runtime `Mapper`
//! value (`NoMapper`, `Camera`, or `XYView`) fixed at construction.
//!
//! A second departure: sequences of per-frame keypoint observations are
//! passed as one flat `observations` slice of length `n_joints * n_frames`
//! (frame `i` spanning `[i * n_joints, (i + 1) * n_joints)`) rather than a
//! nested container, and `solve_sequence`/`solve_sequence_segmented_parallel`
//! return a `StateList` (an indexable handle, `len()`/`at(i)`) rather than a
//! `Vec<State>` -- cxx doesn't support nested `Vec<Vec<T>>` or `Vec` of an
//! opaque Rust type across the bridge.
//!
//! See `README.md` (top level) for build instructions and a usage example.

use std::sync::Arc;

use quickik_core::observation::Mapper3Dto2D;

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

    /// Configuration for the inverse kinematics solver. See Rust's
    /// `quickik::solver::SolverConfig` for field docs. Unlike the Rust/Python
    /// APIs, this is a plain value type here (no shared live handle):
    /// retune a solver by calling `set_config` again.
    #[derive(Clone, Copy, Debug)]
    struct SolverConfig {
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    }

    /// Configuration for `solve_sequence_segmented_parallel`.
    #[derive(Clone, Copy, Debug)]
    struct ParallelSolveConfig {
        segment_len: usize,
        overlap_len: usize,
        overlap_tolerance: f32,
        /// Number of worker threads. A positive value is used directly,
        /// unless it exceeds the number of available cores: it's then
        /// clipped to that count and a warning is logged. A negative
        /// value counts backward from all available cores: `-1` uses all,
        /// `-2` uses all but one, etc. `0` is invalid.
        n_workers: isize,
    }

    extern "Rust" {
        /// A keypoint not observed this frame (e.g. occluded).
        fn keypoint_missing() -> KeypointObservation;
        /// A 3D world position, e.g. triangulated from multiple calibrated
        /// cameras.
        fn keypoint_position_3d(pos: [f32; 3], weight: f32) -> KeypointObservation;
        /// A 2D pixel position from the camera (or other mapper) that the
        /// consuming `Solver`/`SequenceSolver` was constructed with.
        fn keypoint_position_2d(pos: [f32; 2], weight: f32) -> KeypointObservation;

        /// A `Mapper` for solvers that receive 3D keypoint observations only.
        fn no_mapper() -> Mapper;
        /// A `Mapper` that projects with the given pinhole `camera`.
        fn camera_mapper(camera: Camera) -> Mapper;
        /// A `Mapper` that takes a 3D keypoint's world X/Y coordinates as its
        /// 2D projection.
        fn xyview_mapper() -> Mapper;

        /// A `SolverConfig` with reasonable defaults: 10 iterations, small
        /// damping and neutral-pose weight, and tolerances of 1e-3.
        fn default_solver_config() -> SolverConfig;

        /// A `ParallelSolveConfig` that spreads `total_len` frames evenly
        /// across every available core: one segment per core, `total_len /
        /// n_workers` frames each (plus a fixed default overlap). For finer
        /// control over cold-start frequency, build a `ParallelSolveConfig`
        /// directly instead.
        fn parallel_solve_config_for_recording(total_len: usize) -> ParallelSolveConfig;

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

        /// A list of `State`s returned by `solve_sequence`/
        /// `solve_sequence_segmented_parallel`.
        type StateList;
        /// Number of states in the list.
        fn len(self: &StateList) -> usize;
        /// The state at index `i`.
        fn at(self: &StateList, i: usize) -> Box<State>;

        /// The inverse kinematics solver, backed by a single `Mapper` fixed
        /// at construction (see this module's top-level docs).
        type Solver;
        /// Constructs a `Solver` for `tree` with the given `config` and
        /// `mapper`.
        fn new_solver(tree: &KinematicTree, config: SolverConfig, mapper: Mapper) -> Box<Solver>;
        /// Runs `config.n_iterations` Gauss-Newton steps in place on `state`,
        /// given one observation per joint (some may be `Missing`).
        /// Panics from the underlying solve (e.g. a `Position2D` observation
        /// given to a mapper-less solver) are caught and raised as an
        /// exception rather than aborting the process.
        fn solve(
            self: &mut Solver,
            state: &mut State,
            observations: &[KeypointObservation],
        ) -> Result<()>;
        /// The solver's current configuration.
        fn config(self: &Solver) -> SolverConfig;
        /// Replaces the solver's configuration in place.
        fn set_config(self: &mut Solver, config: SolverConfig);
        /// Fixed at construction; there is no setter.
        fn mapper(self: &Solver) -> Mapper;

        /// Solves a continuous sequence of frames for a single tracked body,
        /// warm starting each frame from the previous frame's converged pose.
        type SequenceSolver;
        /// Starts a new sequence at the neutral pose, for `tree`, with the
        /// given `config` and `mapper`.
        fn new_sequence_solver(
            tree: &KinematicTree,
            config: SolverConfig,
            mapper: Mapper,
        ) -> Box<SequenceSolver>;
        /// Solves the next frame in place, warm-started from the current
        /// pose, and returns the converged state.
        /// See `Solver::solve`'s docs on panics being raised as exceptions.
        fn solve_frame(
            self: &mut SequenceSolver,
            observations: &[KeypointObservation],
        ) -> Result<Box<State>>;
        /// Solves every frame in order, each warm-started from the previous
        /// one, returning the converged pose after each frame.
        /// `observations` is flattened: `n_joints * n_frames` long, frame `i`
        /// spanning `[i * n_joints, (i + 1) * n_joints)`.
        fn solve_sequence(
            self: &mut SequenceSolver,
            observations: &[KeypointObservation],
            n_joints: usize,
        ) -> Result<Box<StateList>>;
        /// The most recently converged pose (a snapshot).
        fn state(self: &SequenceSolver) -> Box<State>;
        /// The solver's current configuration.
        fn config(self: &SequenceSolver) -> SolverConfig;
        /// Replaces the solver's configuration in place.
        fn set_config(self: &mut SequenceSolver, config: SolverConfig);
        /// Fixed at construction; there is no setter.
        fn mapper(self: &SequenceSolver) -> Mapper;

        /// Solves a single long sequence in parallel by splitting it into
        /// slightly overlapping segments, each solved on its own thread. See
        /// Rust's `quickik::high_level::solve_sequence_segmented_parallel`.
        /// See `Solver::solve`'s docs on panics being raised as exceptions.
        fn solve_sequence_segmented_parallel(
            tree: &KinematicTree,
            config: SolverConfig,
            observations: &[KeypointObservation],
            n_joints: usize,
            parallel_config: ParallelSolveConfig,
            mapper: Mapper,
        ) -> Result<Box<StateList>>;
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
/// `python/src/observation.rs`'s `Mapper`.
#[derive(Clone, Copy, Debug)]
enum RuntimeMapper {
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

fn to_runtime_mapper(mapper: &ffi::Mapper) -> Option<RuntimeMapper> {
    match mapper.kind {
        ffi::MapperKind::NoMapper => None,
        ffi::MapperKind::CameraMapper => {
            Some(RuntimeMapper::Camera(to_core_camera(&mapper.camera)))
        }
        ffi::MapperKind::XYViewMapper => Some(RuntimeMapper::XYView),
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

fn runtime_mapper_to_ffi(mapper: Option<RuntimeMapper>) -> ffi::Mapper {
    match mapper {
        None => no_mapper(),
        Some(RuntimeMapper::Camera(camera)) => {
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
        Some(RuntimeMapper::XYView) => xyview_mapper(),
    }
}

// =============================================================================
//  SolverConfig
// =============================================================================

fn default_solver_config() -> ffi::SolverConfig {
    from_core_config(&quickik_core::solver::SolverConfig::<RuntimeMapper>::default())
}

fn to_core_config(
    config: ffi::SolverConfig,
    mapper: Option<RuntimeMapper>,
) -> quickik_core::solver::SolverConfig<RuntimeMapper> {
    quickik_core::solver::SolverConfig {
        n_iterations: config.n_iterations,
        neutral_weight: config.neutral_weight,
        position_tolerance: config.position_tolerance,
        angle_tolerance: config.angle_tolerance,
        damping: config.damping,
        mapper,
    }
}

fn from_core_config(
    config: &quickik_core::solver::SolverConfig<RuntimeMapper>,
) -> ffi::SolverConfig {
    ffi::SolverConfig {
        n_iterations: config.n_iterations,
        neutral_weight: config.neutral_weight,
        position_tolerance: config.position_tolerance,
        angle_tolerance: config.angle_tolerance,
        damping: config.damping,
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

// =============================================================================
//  StateList
// =============================================================================

struct StateList(Vec<quickik_core::state::State>);

impl StateList {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn at(&self, i: usize) -> Box<State> {
        Box::new(State(self.0[i].clone()))
    }
}

/// Splits a flat, `n_joints * n_frames`-long slice into per-frame chunks and
/// converts each observation to the core crate's type. See this module's
/// top-level docs for why sequences are passed flattened.
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

// =============================================================================
//  Solver
// =============================================================================

struct Solver {
    inner: quickik_core::solver::Solver<RuntimeMapper>,
    mapper: Option<RuntimeMapper>,
}

fn new_solver(tree: &KinematicTree, config: ffi::SolverConfig, mapper: ffi::Mapper) -> Box<Solver> {
    let mapper = to_runtime_mapper(&mapper);
    Box::new(Solver {
        inner: quickik_core::solver::Solver::new(&tree.0, to_core_config(config, mapper)),
        mapper,
    })
}

impl Solver {
    fn solve(
        &mut self,
        state: &mut State,
        observations: &[ffi::KeypointObservation],
    ) -> Result<(), String> {
        let observations: Vec<_> = observations.iter().map(to_core_observation).collect();
        let inner = &mut self.inner;
        let state = &mut state.0;
        catch_panic(move || inner.solve(state, &observations))
    }
    fn config(&self) -> ffi::SolverConfig {
        from_core_config(&self.inner.config)
    }
    fn set_config(&mut self, config: ffi::SolverConfig) {
        self.inner.config = to_core_config(config, self.mapper);
    }
    fn mapper(&self) -> ffi::Mapper {
        runtime_mapper_to_ffi(self.mapper)
    }
}

// =============================================================================
//  SequenceSolver
// =============================================================================

struct SequenceSolver {
    inner: quickik_core::high_level::SequenceSolver<RuntimeMapper>,
    mapper: Option<RuntimeMapper>,
}

fn new_sequence_solver(
    tree: &KinematicTree,
    config: ffi::SolverConfig,
    mapper: ffi::Mapper,
) -> Box<SequenceSolver> {
    let mapper = to_runtime_mapper(&mapper);
    Box::new(SequenceSolver {
        inner: quickik_core::high_level::SequenceSolver::new(
            tree.0.clone(),
            to_core_config(config, mapper),
        ),
        mapper,
    })
}

impl SequenceSolver {
    fn solve_frame(
        &mut self,
        observations: &[ffi::KeypointObservation],
    ) -> Result<Box<State>, String> {
        let observations: Vec<_> = observations.iter().map(to_core_observation).collect();
        let inner = &mut self.inner;
        catch_panic(move || Box::new(State(inner.solve_frame(&observations).clone())))
    }
    fn solve_sequence(
        &mut self,
        observations: &[ffi::KeypointObservation],
        n_joints: usize,
    ) -> Result<Box<StateList>, String> {
        let sequence = split_into_frames(observations, n_joints);
        let inner = &mut self.inner;
        catch_panic(move || Box::new(StateList(inner.solve_sequence(&sequence))))
    }
    fn state(&self) -> Box<State> {
        Box::new(State(self.inner.state.clone()))
    }
    fn config(&self) -> ffi::SolverConfig {
        from_core_config(&self.inner.solver.config)
    }
    fn set_config(&mut self, config: ffi::SolverConfig) {
        self.inner.solver.config = to_core_config(config, self.mapper);
    }
    fn mapper(&self) -> ffi::Mapper {
        runtime_mapper_to_ffi(self.mapper)
    }
}

// =============================================================================
//  ParallelSolveConfig
// =============================================================================

fn parallel_solve_config_for_recording(total_len: usize) -> ffi::ParallelSolveConfig {
    let config = quickik_core::high_level::ParallelSolveConfig::for_recording(total_len);
    ffi::ParallelSolveConfig {
        segment_len: config.segment_len,
        overlap_len: config.overlap_len,
        overlap_tolerance: config.overlap_tolerance,
        n_workers: config.n_workers,
    }
}

// =============================================================================
//  solve_sequence_segmented_parallel
// =============================================================================

fn solve_sequence_segmented_parallel(
    tree: &KinematicTree,
    config: ffi::SolverConfig,
    observations: &[ffi::KeypointObservation],
    n_joints: usize,
    parallel_config: ffi::ParallelSolveConfig,
    mapper: ffi::Mapper,
) -> Result<Box<StateList>, String> {
    let mapper = to_runtime_mapper(&mapper);
    let sequence = split_into_frames(observations, n_joints);
    let core_parallel_config = quickik_core::high_level::ParallelSolveConfig {
        segment_len: parallel_config.segment_len,
        overlap_len: parallel_config.overlap_len,
        overlap_tolerance: parallel_config.overlap_tolerance,
        n_workers: parallel_config.n_workers,
    };
    let core_config = to_core_config(config, mapper);
    catch_panic(move || {
        Box::new(StateList(
            quickik_core::high_level::solve_sequence_segmented_parallel(
                &tree.0,
                core_config,
                &sequence,
                core_parallel_config,
            ),
        ))
    })
}
