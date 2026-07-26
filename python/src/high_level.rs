use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::observation::{
    KeypointObservation, Mapper, extract_mapper, extract_observations, mapper_to_py,
    observations_from_arrays,
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
    /// returns the converged state (also available as `.state`).
    fn solve_frame(
        &mut self,
        py: Python<'_>,
        observations: Vec<PyRef<'_, KeypointObservation>>,
    ) -> State {
        self.sync_config(py);
        let state = self.inner.solve_frame(&extract_observations(observations));
        State {
            inner: state.clone(),
        }
    }

    /// Solves every frame in `sequence` (a list of per-frame observation
    /// lists) in order, each warm-started from the previous one; returns
    /// the converged pose after each frame.
    fn solve_sequence(
        &mut self,
        py: Python<'_>,
        sequence: Vec<Vec<PyRef<'_, KeypointObservation>>>,
    ) -> Vec<State> {
        self.sync_config(py);
        let sequence: Vec<Vec<_>> = sequence.into_iter().map(extract_observations).collect();
        self.inner
            .solve_sequence(&sequence)
            .into_iter()
            .map(|inner| State { inner })
            .collect()
    }

    /// The most recently converged pose (a snapshot; mutating it has no
    /// effect on the solver).
    #[getter]
    fn state(&self) -> State {
        State {
            inner: self.inner.state.clone(),
        }
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
    /// exceeds the number of available cores -- in that case it's clipped to
    /// that count and a warning is logged. A negative value counts backward
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
    /// cold-start frequency -- how often a segment restarts from the neutral
    /// pose, trading accuracy for finer-grained parallelism -- build a
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
/// `Camera`, an `XYView`, or `None` -- see [`Solver`](crate::solver::Solver).
///
/// Observations are given as raw arrays rather than a list of per-frame
/// `KeypointObservation` lists: `positions` is `(n_frames, n_keypoints, 3)`
/// and `weights` is `(n_frames, n_keypoints)`, both in
/// `kinematic_tree.joints` order; a keypoint with `weight <= 0` is treated
/// as [`missing`](crate::observation::KeypointObservation::missing). This
/// avoids constructing one Python `KeypointObservation` object per keypoint
/// per frame, which otherwise dominates call overhead for large sequences
/// (e.g. a whole recording's worth of frames in one call).
#[pyfunction]
#[pyo3(signature = (kinematic_tree, config, positions, weights, parallel_config, mapper=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_sequence_segmented_parallel(
    py: Python<'_>,
    kinematic_tree: KinematicTree,
    config: SolverConfig,
    positions: PyReadonlyArray3<'_, f32>,
    weights: PyReadonlyArray2<'_, f32>,
    parallel_config: ParallelSolveConfig,
    mapper: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<State>> {
    let n_joints = kinematic_tree.inner.n_joints();
    let positions_arr = positions.as_array();
    let weights_arr = weights.as_array();
    let (n_frames, n_keypoints, dim) = positions_arr.dim();
    if dim != 3 {
        return Err(PyValueError::new_err(format!(
            "positions must have shape (n_frames, n_keypoints, 3), got last dimension {dim}"
        )));
    }
    if n_keypoints != n_joints {
        return Err(PyValueError::new_err(format!(
            "positions has {n_keypoints} keypoints, but kinematic_tree has {n_joints} joints"
        )));
    }
    if weights_arr.dim() != (n_frames, n_keypoints) {
        return Err(PyValueError::new_err(format!(
            "weights must have shape (n_frames, n_keypoints) = ({n_frames}, {n_keypoints}), got {:?}",
            weights_arr.dim()
        )));
    }

    let config = config.as_rust(extract_mapper(mapper.as_ref())?);
    Ok(py
        .detach(|| {
            let sequence = observations_from_arrays(positions_arr, weights_arr);
            quickik_core::high_level::solve_sequence_segmented_parallel(
                &kinematic_tree.inner,
                config,
                &sequence,
                parallel_config.inner,
            )
        })
        .into_iter()
        .map(|inner| State { inner })
        .collect())
}
