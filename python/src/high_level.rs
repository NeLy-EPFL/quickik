use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLike2, PyArrayLike3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::catch_panic;
use crate::observation::{
    KeypointObservation, Mapper, extract_mapper, extract_observations, fk_positions_to_pyarray,
    mapper_to_py, observations_from_arrays, validate_position_weight_shapes,
};
use crate::solver::SolverConfig;
use crate::state::State;

/// Solves a continuous sequence of frames for a single tracked body, warm
/// starting each frame from the previous frame's converged pose. See
/// [`Solver`](crate::solver::Solver) for `mapper` and `config` semantics
/// (both flattened here from Rust's nested `solver.solver`).
#[pyclass(module = "quickik")]
pub(crate) struct SequenceSolver {
    inner: quickik_core::high_level::SequenceSolver<Mapper>,
    config: Py<SolverConfig>,
    mapper: Option<Mapper>,
}

#[pymethods]
impl SequenceSolver {
    /// Starts a new sequence at the neutral pose. Raises `ValueError` if
    /// `mapper` is not a `Camera`, an `XYView`, or `None`.
    #[new]
    #[pyo3(signature = (kinematic_tree, config, mapper=None))]
    fn new(
        py: Python<'_>,
        kinematic_tree: KinematicTree,
        config: SolverConfig,
        mapper: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mapper = extract_mapper(mapper.as_ref())?;
        Ok(SequenceSolver {
            inner: quickik_core::high_level::SequenceSolver::new(
                kinematic_tree.inner,
                config.as_rust(mapper),
            ),
            config: Py::new(py, config)?,
            mapper,
        })
    }

    /// Solves the next frame, warm-started from the current pose, and
    /// returns the converged state (also available as `.state`). Raises
    /// `ValueError` if `len(observations) != kinematic_tree.n_joints`.
    fn solve_frame(
        &mut self,
        py: Python<'_>,
        observations: Vec<PyRef<'_, KeypointObservation>>,
    ) -> PyResult<State> {
        self.sync_config(py);
        let observations = extract_observations(observations);
        let inner = &mut self.inner;
        catch_panic(move || State {
            inner: inner.solve_frame(&observations).clone(),
        })
    }

    /// Solves every frame in order, each warm-started from the previous one;
    /// returns the converged pose after each frame. `weights` is
    /// `(n_frames, n_joints)`; `positions` is `(n_frames, n_joints, 3)` if
    /// this solver has no mapper (3D observations), or `(n_frames, n_joints,
    /// 2)` if it does (2D observations, projected by that mapper) -- see
    /// `mapper`. Both are in `kinematic_tree.joints` order; a keypoint with
    /// `weight <= 0` (or NaN) is treated as
    /// [`missing`](crate::observation::KeypointObservation::missing). Given
    /// as raw arrays rather than a list of per-frame `KeypointObservation`
    /// lists so this never constructs one Python object per keypoint per
    /// frame, which otherwise dominates call overhead for long recordings.
    /// Any dtype is accepted and cast to `float32` (e.g. the common case of
    /// a `float64` array), following NumPy's own casting rules.
    fn solve_sequence(
        &mut self,
        py: Python<'_>,
        positions: PyArrayLike3<'_, f32, AllowTypeChange>,
        weights: PyArrayLike2<'_, f32, AllowTypeChange>,
    ) -> PyResult<Vec<State>> {
        self.sync_config(py);
        let positions_arr = positions.as_array();
        let weights_arr = weights.as_array();
        let n_joints = self.inner.state.kinematic_tree.n_joints();
        validate_position_weight_shapes(
            &positions_arr,
            &weights_arr,
            n_joints,
            self.mapper.is_some(),
        )?;
        let sequence = observations_from_arrays(positions_arr, weights_arr);
        Ok(self
            .inner
            .solve_sequence(&sequence)
            .into_iter()
            .map(|inner| State { inner })
            .collect())
    }

    /// Same as `solve_sequence`, but additionally returns each frame's
    /// forward-kinematics keypoint positions alongside its converged state:
    /// `(states, fk_positions)`, where `fk_positions` is `(n_frames,
    /// n_joints, 3)` float32. `fk_positions` is always the raw 3D world
    /// position of every keypoint (in `kinematic_tree.joints` order),
    /// regardless of `mapper` -- unlike `positions`/`weights`, it's never
    /// projected to 2D.
    fn solve_sequence_with_fk<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyArrayLike3<'_, f32, AllowTypeChange>,
        weights: PyArrayLike2<'_, f32, AllowTypeChange>,
    ) -> PyResult<(Vec<State>, Bound<'py, PyArray3<f32>>)> {
        self.sync_config(py);
        let positions_arr = positions.as_array();
        let weights_arr = weights.as_array();
        let n_joints = self.inner.state.kinematic_tree.n_joints();
        validate_position_weight_shapes(
            &positions_arr,
            &weights_arr,
            n_joints,
            self.mapper.is_some(),
        )?;
        let sequence = observations_from_arrays(positions_arr, weights_arr);
        let results = self.inner.solve_sequence_with_fk(&sequence);

        let n_frames = results.len();
        let mut fk_arr = Array3::<f32>::zeros((n_frames, n_joints, 3));
        let mut states = Vec::with_capacity(n_frames);
        for (i, (state, fk)) in results.into_iter().enumerate() {
            let mut frame_view = fk_arr.index_axis_mut(Axis(0), i);
            for (mut row, pos) in frame_view.rows_mut().into_iter().zip(&fk) {
                row[0] = pos.x;
                row[1] = pos.y;
                row[2] = pos.z;
            }
            states.push(State { inner: state });
        }

        Ok((states, fk_arr.into_pyarray(py)))
    }

    /// The most recently converged pose (a snapshot; mutating it has no
    /// effect on the solver).
    #[getter]
    fn state(&self) -> State {
        State {
            inner: self.inner.state.clone(),
        }
    }

    /// World-space keypoint positions (`(n_joints, 3)` float32, always 3D
    /// regardless of `mapper`) at the most recently converged pose, in
    /// `kinematic_tree.joints` order.
    #[getter]
    fn last_fk_positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        fk_positions_to_pyarray(py, self.inner.last_fk_positions())
    }

    /// The live config, shared with the underlying `Solver`; see `Solver`'s
    /// docstring for mutation semantics.
    #[getter]
    fn config(&self, py: Python<'_>) -> Py<SolverConfig> {
        self.config.clone_ref(py)
    }
    #[setter]
    fn set_config(&mut self, config: Py<SolverConfig>) {
        self.config = config;
    }

    /// Fixed at construction (read-only); mutating the returned object has
    /// no effect on this solver.
    #[getter]
    fn mapper(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        mapper_to_py(py, self.mapper)
    }
}

impl SequenceSolver {
    fn sync_config(&mut self, py: Python<'_>) {
        self.inner.solver.config = self.config.borrow(py).as_rust(self.mapper);
    }
}

/// Configuration for [`solve_sequence_segmented_parallel`].
#[pyclass(module = "quickik", from_py_object, frozen)]
#[derive(Clone, Copy)]
pub(crate) struct ParallelSolveConfig {
    inner: quickik_core::high_level::ParallelSolveConfig,
}

#[pymethods]
impl ParallelSolveConfig {
    /// `segment_len`: frames per segment, including overlap (must exceed
    /// `overlap_len`). `overlap_len`: frames shared with the next segment,
    /// for warm-starting and consistency checking. `overlap_tolerance`: max
    /// per-DOF angle disagreement (radians) tolerated between neighboring
    /// segments' overlapping frames before logging a warning. `n_workers`:
    /// number of worker threads. A positive value is used directly, unless it
    /// exceeds the number of available cores: it's then clipped to that
    /// count and a warning is logged. A negative value counts backward
    /// from all available cores: `-1` uses all, `-2` uses all but one, etc.
    /// `0` is invalid.
    #[new]
    fn new(
        segment_len: usize,
        overlap_len: usize,
        overlap_tolerance: f32,
        n_workers: isize,
    ) -> Self {
        ParallelSolveConfig {
            inner: quickik_core::high_level::ParallelSolveConfig {
                segment_len,
                overlap_len,
                overlap_tolerance,
                n_workers,
            },
        }
    }

    /// A `ParallelSolveConfig` that spreads `total_len` frames evenly across
    /// every available core: one segment per core, `total_len / n_workers`
    /// frames each (plus a fixed default overlap). For finer control over
    /// cold-start frequency (how often a segment restarts from the neutral
    /// pose, trading accuracy for finer-grained parallelism), build a
    /// `ParallelSolveConfig` directly instead.
    #[staticmethod]
    fn for_recording(total_len: usize) -> Self {
        ParallelSolveConfig {
            inner: quickik_core::high_level::ParallelSolveConfig::for_recording(total_len),
        }
    }
}

/// Solves a single long sequence in parallel by splitting it into slightly
/// overlapping segments, each solved on its own thread. `mapper` is a
/// `Camera`, an `XYView`, or `None`; see [`Solver`](crate::solver::Solver).
///
/// Observations are given as raw arrays rather than a list of per-frame
/// `KeypointObservation` lists: `weights` is `(n_frames, n_keypoints)`;
/// `positions` is `(n_frames, n_keypoints, 3)` if `mapper` is `None` (3D
/// observations), or `(n_frames, n_keypoints, 2)` if it's set (2D
/// observations, projected by that mapper). Both are in
/// `kinematic_tree.joints` order; a keypoint with `weight <= 0` (or NaN) is
/// treated as [`missing`](crate::observation::KeypointObservation::missing).
/// This avoids constructing one Python `KeypointObservation` object per
/// keypoint per frame, which otherwise dominates call overhead for large
/// sequences (e.g. a whole recording's worth of frames in one call). Any
/// dtype is accepted and cast to `float32` (e.g. the common case of a
/// `float64` array), following NumPy's own casting rules.
#[pyfunction]
#[pyo3(signature = (kinematic_tree, config, positions, weights, parallel_config, mapper=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_sequence_segmented_parallel(
    py: Python<'_>,
    kinematic_tree: KinematicTree,
    config: SolverConfig,
    positions: PyArrayLike3<'_, f32, AllowTypeChange>,
    weights: PyArrayLike2<'_, f32, AllowTypeChange>,
    parallel_config: ParallelSolveConfig,
    mapper: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<State>> {
    let n_joints = kinematic_tree.inner.n_joints();
    let mapper = extract_mapper(mapper.as_ref())?;
    let positions_arr = positions.as_array();
    let weights_arr = weights.as_array();
    validate_position_weight_shapes(&positions_arr, &weights_arr, n_joints, mapper.is_some())?;

    let config = config.as_rust(mapper);
    Ok(py
        .detach(|| {
            catch_panic(|| {
                let sequence = observations_from_arrays(positions_arr, weights_arr);
                quickik_core::high_level::solve_sequence_segmented_parallel(
                    &kinematic_tree.inner,
                    config,
                    &sequence,
                    parallel_config.inner,
                )
            })
        })?
        .into_iter()
        .map(|inner| State { inner })
        .collect())
}

/// Solves a batch of fully independent sets of keypoint observations in
/// parallel, each starting from `kinematic_tree`'s neutral pose, and returns
/// enough of each item's linearization (Jacobian and Cholesky factor) to
/// support implicit differentiation of the solve on the Python/PyTorch side;
/// see `quickik.torch.SolveIK`.
///
/// `mapper` must be `None` (3D observations) or an `XYView` (2D); a `Camera`
/// is rejected. The returned Jacobian is always the raw 3D
/// keypoint-position Jacobian: for `XYView`, whose 2D projection is exactly
/// that Jacobian's first two rows (a fixed, position-independent linear
/// map), that's enough for the Python/PyTorch side to differentiate
/// correctly by slicing it appropriately (see `quickik.torch.SolveIK`). A
/// `Camera`'s projection Jacobian genuinely depends on position and isn't
/// retained, so it can't be supported this way.
///
/// `kinematic_tree` must be free-floating (not fixed-base), since the
/// returned pose always includes `base_pos`/`base_quat`.
///
/// `keypoints_order[i]` is the joint name (see `KinematicTree.joint_names`)
/// that `positions`/`weights`'s keypoint axis position `i` corresponds to;
/// every joint in `kinematic_tree` must appear in it exactly once.
/// `positions` is `(batch_size, n_joints, 3)` if `mapper` is `None`, or
/// `(batch_size, n_joints, 2)` if it's an `XYView`; `weights` is
/// `(batch_size, n_joints)`; a keypoint with `weight <= 0` (or NaN) is
/// treated as missing, same convention as `solve_sequence_segmented_parallel`.
///
/// Returns `(joint_angles, base_pos, base_quat, jacobian, cholesky_l, valid)`:
/// - `joint_angles`: `(batch_size, n_dofs)`, in `kinematic_tree`'s own DOF
///   order -- unrelated to `keypoints_order`, since DOF order is already
///   fully caller-controlled via how the tree was built.
/// - `base_pos`: `(batch_size, 3)`.
/// - `base_quat`: `(batch_size, 4)`, `(w, x, y, z)`.
/// - `jacobian`: `(batch_size, 3 * n_joints, state_dim)` -- always the raw
///   3D Jacobian, regardless of `mapper` (see above) -- in `kinematic_tree`'s
///   internal keypoint/state order, *not* `keypoints_order`.
/// - `cholesky_l`: `(batch_size, state_dim, state_dim)`; zeroed for any item
///   where `valid` is `False`.
/// - `valid`: `(batch_size,)`; `False` where that item's last iteration
///   wasn't positive-definite, so its `cholesky_l` can't be used for
///   gradients.
type SolveBatchWithGradResult<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray1<bool>>,
);

#[pyfunction]
#[pyo3(signature = (kinematic_tree, config, keypoints_order, positions, weights, mapper=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_batch_with_grad<'py>(
    py: Python<'py>,
    kinematic_tree: KinematicTree,
    config: SolverConfig,
    keypoints_order: Vec<String>,
    positions: PyArrayLike3<'_, f32, AllowTypeChange>,
    weights: PyArrayLike2<'_, f32, AllowTypeChange>,
    mapper: Option<Bound<'_, PyAny>>,
) -> PyResult<SolveBatchWithGradResult<'py>> {
    let mapper = extract_mapper(mapper.as_ref())?;
    if matches!(mapper, Some(Mapper::Camera(_))) {
        return Err(PyValueError::new_err(
            "solve_batch_with_grad doesn't support Camera: its returned Jacobian is always the \
             raw 3D keypoint-position Jacobian, and unlike XYView's (which is just that \
             Jacobian's first two rows), Camera's own projection Jacobian depends on position \
             and isn't retained. Use mapper=None (3D) or mapper=XYView (2D) instead.",
        ));
    }
    let n_joints = kinematic_tree.inner.n_joints();
    let positions_arr = positions.as_array();
    let weights_arr = weights.as_array();
    validate_position_weight_shapes(&positions_arr, &weights_arr, n_joints, mapper.is_some())?;

    let config = config.as_rust(mapper);
    let result = py.detach(|| {
        catch_panic(|| {
            let observations_array = observations_from_arrays(positions_arr, weights_arr);
            quickik_core::high_level::solve_batch_with_grad(
                &kinematic_tree.inner,
                config,
                &keypoints_order,
                &observations_array,
            )
        })
    })?;

    let batch_size = result.joint_angles.len();
    let n_dofs = kinematic_tree.inner.n_dofs();
    let state_dim = kinematic_tree.inner.state_dim();

    let mut joint_angles_arr = Array2::<f32>::zeros((batch_size, n_dofs));
    let mut base_pos_arr = Array2::<f32>::zeros((batch_size, 3));
    let mut base_quat_arr = Array2::<f32>::zeros((batch_size, 4));
    let mut jacobian_arr = Array3::<f32>::zeros((batch_size, 3 * n_joints, state_dim));
    let mut cholesky_l_arr = Array3::<f32>::zeros((batch_size, state_dim, state_dim));
    let mut valid_arr = Array1::<bool>::from_elem(batch_size, false);

    for i in 0..batch_size {
        joint_angles_arr
            .row_mut(i)
            .iter_mut()
            .zip(&result.joint_angles[i])
            .for_each(|(dst, &src)| *dst = src);

        let p = result.base_pos[i];
        base_pos_arr
            .row_mut(i)
            .assign(&Array1::from_vec(vec![p.x, p.y, p.z]));

        let q = result.base_quat[i].quaternion();
        base_quat_arr
            .row_mut(i)
            .assign(&Array1::from_vec(vec![q.w, q.i, q.j, q.k]));

        let jac = &result.jacobian[i];
        let mut jac_view = jacobian_arr.index_axis_mut(Axis(0), i);
        for r in 0..jac.nrows() {
            for c in 0..jac.ncols() {
                jac_view[[r, c]] = jac[(r, c)];
            }
        }

        if let Some(chol) = &result.cholesky_l[i] {
            valid_arr[i] = true;
            let l = chol.l();
            let mut l_view = cholesky_l_arr.index_axis_mut(Axis(0), i);
            for r in 0..l.nrows() {
                for c in 0..l.ncols() {
                    l_view[[r, c]] = l[(r, c)];
                }
            }
        }
    }

    Ok((
        joint_angles_arr.into_pyarray(py),
        base_pos_arr.into_pyarray(py),
        base_quat_arr.into_pyarray(py),
        jacobian_arr.into_pyarray(py),
        cholesky_l_arr.into_pyarray(py),
        valid_arr.into_pyarray(py),
    ))
}
