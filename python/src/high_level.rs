use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::observation::{KeypointObservation, Mapper, extract_mapper, extract_observations, mapper_to_py};
use crate::solver::SolverConfig;
use crate::state::State;

/// Solves a continuous sequence of frames for a single tracked body, warm
/// starting each frame from the previous frame's converged pose. See
/// [`Solver`](crate::solver::Solver) for `mapper` and `config` semantics
/// (both flattened here from Rust's nested `solver.solver`).
#[pyclass(module = "fastik")]
pub(crate) struct SequenceSolver {
    inner: fastik_core::high_level::SequenceSolver<Mapper>,
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
            inner: fastik_core::high_level::SequenceSolver::new(kinematic_tree.inner, config.as_rust(mapper)),
            config: Py::new(py, config)?,
            mapper,
        })
    }

    /// Solves the next frame, warm-started from the current pose, and
    /// returns the converged state (also available as `.state`).
    fn solve_frame(&mut self, py: Python<'_>, observations: Vec<PyRef<'_, KeypointObservation>>) -> State {
        self.sync_config(py);
        let state = self.inner.solve_frame(&extract_observations(observations));
        State { inner: state.clone() }
    }

    /// Solves every frame in `sequence` (a list of per-frame observation
    /// lists) in order, each warm-started from the previous one; returns
    /// the converged pose after each frame.
    fn solve_sequence(&mut self, py: Python<'_>, sequence: Vec<Vec<PyRef<'_, KeypointObservation>>>) -> Vec<State> {
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
#[pyclass(module = "fastik", from_py_object, frozen)]
#[derive(Clone, Copy)]
pub(crate) struct SegmentedSolveConfig {
    inner: fastik_core::high_level::SegmentedSolveConfig,
}

#[pymethods]
impl SegmentedSolveConfig {
    /// `segment_len`: frames per segment, including overlap (must exceed
    /// `overlap_len`). `overlap_len`: frames shared with the next segment,
    /// for warm-starting and consistency checking. `overlap_tolerance`: max
    /// per-DOF angle disagreement (radians) tolerated between neighboring
    /// segments' overlapping frames before logging a warning.
    #[new]
    fn new(segment_len: usize, overlap_len: usize, overlap_tolerance: f32) -> Self {
        SegmentedSolveConfig {
            inner: fastik_core::high_level::SegmentedSolveConfig {
                segment_len,
                overlap_len,
                overlap_tolerance,
            },
        }
    }
}

/// Solves a single long sequence in parallel by splitting it into slightly
/// overlapping segments, each solved on its own thread. `mapper` is a
/// `Camera`, an `XYView`, or `None` -- see [`Solver`](crate::solver::Solver).
#[pyfunction]
#[pyo3(signature = (kinematic_tree, config, sequence, segmented_config, mapper=None))]
pub(crate) fn solve_sequence_segmented_parallel(
    py: Python<'_>,
    kinematic_tree: KinematicTree,
    config: SolverConfig,
    sequence: Vec<Vec<PyRef<'_, KeypointObservation>>>,
    segmented_config: SegmentedSolveConfig,
    mapper: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<State>> {
    let config = config.as_rust(extract_mapper(mapper.as_ref())?);
    let sequence: Vec<Vec<_>> = sequence.into_iter().map(extract_observations).collect();
    Ok(py
        .detach(|| {
            fastik_core::high_level::solve_sequence_segmented_parallel(
                &kinematic_tree.inner,
                config,
                &sequence,
                segmented_config.inner,
            )
        })
        .into_iter()
        .map(|inner| State { inner })
        .collect())
}
