use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::observation::{KeypointObservation, ObservationSequence, extract};
use crate::solver::{SolverConfig, SolverResult, to_py_results};
use crate::state::State;
use crate::to_py_err;

/// Warm-started solving of one continuous stream.
///
/// Holds a single running state that `step` and `solve` advance: each frame
/// starts from the previous frame's solution, which is also the smoothness
/// priors' target. Construct one per independent stream (e.g. per tracked
/// subject).
#[pyclass(module = "quickik")]
pub(crate) struct SequentialSolver {
    inner: quickik_core::sequential_solver::SequentialSolver,
}

#[pymethods]
impl SequentialSolver {
    /// Starts a stream at `initial_state`, or the neutral pose if `None`.
    /// Raises `ValueError` if `config` has root priors and the tree is
    /// fixed-base, or if `initial_state` doesn't match the tree.
    #[new]
    #[pyo3(signature = (kinematic_tree, config, initial_state=None))]
    fn new(
        kinematic_tree: KinematicTree,
        config: SolverConfig,
        initial_state: Option<State>,
    ) -> PyResult<Self> {
        let solver = quickik_core::sequential_solver::SequentialSolver::new(
            &kinematic_tree.inner,
            &config.inner,
            initial_state.map(|state| state.inner),
        );
        Ok(SequentialSolver {
            inner: solver.map_err(to_py_err)?,
        })
    }

    #[getter]
    fn kinematic_tree(&self) -> KinematicTree {
        KinematicTree {
            inner: self.inner.kinematic_tree().clone(),
        }
    }

    /// A copy of the solver's configuration.
    #[getter]
    fn config(&self) -> SolverConfig {
        SolverConfig {
            inner: self.inner.config().clone(),
        }
    }

    /// A copy of the running state after the last `step`/`solve` call.
    #[getter]
    fn state(&self) -> State {
        State {
            inner: self.inner.state().clone(),
        }
    }

    /// Starts a new stream: sets the running state (neutral pose if `None`) and
    /// forgets the previous frame, so smoothness priors are inactive on the
    /// next frame.
    #[pyo3(signature = (state=None))]
    fn reset(&mut self, state: Option<State>) -> PyResult<()> {
        let state = state.map(|state| state.inner);
        self.inner.reset(state).map_err(to_py_err)
    }

    /// Solves the next frame of the stream, given its observations.
    #[pyo3(signature = (observations, with_grad=false, with_fk=false))]
    fn step(
        &mut self,
        observations: Vec<PyRef<'_, KeypointObservation>>,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<SolverResult> {
        let observations = extract(observations);
        let result = self.inner.step(&observations, with_grad, with_fk);
        Ok(result.map_err(to_py_err)?.into())
    }

    /// `step` for each frame of `sequence` in order. Returns one result per
    /// frame.
    #[pyo3(signature = (sequence, with_grad=false, with_fk=false))]
    fn solve(
        &mut self,
        py: Python<'_>,
        sequence: PyRef<'_, ObservationSequence>,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<Vec<SolverResult>> {
        let sequence = &sequence.inner;
        let inner = &mut self.inner;
        let results = py.detach(|| inner.solve(sequence, with_grad, with_fk));
        Ok(to_py_results(results.map_err(to_py_err)?))
    }

    /// Offline: splits `sequence` into up to `n_workers` contiguous chunks of
    /// near-equal length, solved in parallel. Never reads or writes the running
    /// state.
    ///
    /// Each chunk starts from the neutral pose. Every chunk but the first
    /// starts `n_overlap` frames early (or at the sequence start if that's
    /// closer) and discards those warm-up results, so its first real frame is
    /// warm-started and has a previous frame for the smoothness priors.
    /// `n_overlap=0` makes chunks fully independent.
    ///
    /// `n_workers`: a positive value is used directly, unless it exceeds the
    /// number of available cores, in which case it's clipped to that. A
    /// negative value counts back from all available cores: `-1` uses all,
    /// `-2` all but one, etc. `0` raises `ValueError`.
    #[pyo3(signature = (sequence, n_workers=-1, n_overlap=0, with_grad=false, with_fk=false))]
    fn solve_parallel(
        &self,
        py: Python<'_>,
        sequence: PyRef<'_, ObservationSequence>,
        n_workers: isize,
        n_overlap: usize,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<Vec<SolverResult>> {
        let sequence = &sequence.inner;
        let inner = &self.inner;
        let results =
            py.detach(|| inner.solve_parallel(sequence, n_workers, n_overlap, with_grad, with_fk));
        Ok(to_py_results(results.map_err(to_py_err)?))
    }
}
