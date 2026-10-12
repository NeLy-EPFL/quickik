use numpy::{AllowTypeChange, PyArray2, PyArrayLike1};
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::observation::{KeypointObservation, extract};
use crate::state::State;
use crate::{matrix_to_py, points_to_py, to_py_err, to_vector};

/// Iteration settings (fixed at construction) plus priors (added with the
/// `add_*` methods, each at most once). No priors by default.
///
/// Copied into each solver on construction. Checked there too: root priors on
/// a fixed-base tree, and smoothness priors in `SingleFrameSolver` or
/// `BatchedSolver`, raise `ValueError`.
#[pyclass(module = "quickik", from_py_object)]
#[derive(Clone)]
pub(crate) struct SolverConfig {
    pub(crate) inner: quickik_core::solver::SolverConfig,
}

#[pymethods]
impl SolverConfig {
    /// - `n_iterations`: maximum number of Gauss-Newton steps per frame.
    /// - `position_tolerance`, `angle_tolerance`: stop early once a step's
    ///   largest root-position component is at most `position_tolerance` and
    ///   its largest angle component (root rotation and every DOF) is at most
    ///   `angle_tolerance` (radians). Set both to 0 to always run
    ///   `n_iterations` steps.
    /// - `damping`: Levenberg-Marquardt damping, relative to the normal
    ///   equations' diagonal (floored at 1). Only for numerical stability, so
    ///   keep it very small.
    #[new]
    #[pyo3(signature = (
        n_iterations=10, position_tolerance=1e-3, angle_tolerance=1e-3, damping=1e-6,
    ))]
    fn new(
        n_iterations: usize,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    ) -> Self {
        SolverConfig {
            inner: quickik_core::solver::SolverConfig::new(
                n_iterations,
                position_tolerance,
                angle_tolerance,
                damping,
            ),
        }
    }

    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.n_iterations()
    }

    #[getter]
    fn position_tolerance(&self) -> f32 {
        self.inner.position_tolerance()
    }

    #[getter]
    fn angle_tolerance(&self) -> f32 {
        self.inner.angle_tolerance()
    }

    #[getter]
    fn damping(&self) -> f32 {
        self.inner.damping()
    }

    /// Pulls each DOF toward its `Dof.neutral` value, with per-DOF weight
    /// `weight * Dof.weight_multiplier`. Improves robustness to missing or
    /// noisy keypoints, but biases the solution toward the neutral pose.
    fn add_dof_neutral_prior(&mut self, weight: f32) -> PyResult<()> {
        self.inner.add_dof_neutral_prior(weight).map_err(to_py_err)
    }

    /// Pulls the root's local `body_up` axis toward the world-frame `world_up`
    /// (e.g. +z, or a tilted vector on a slope), leaving rotation about the up
    /// axis free. Both vectors are normalized.
    fn add_root_upright_prior(
        &mut self,
        weight: f32,
        body_up: PyArrayLike1<'_, f32, AllowTypeChange>,
        world_up: PyArrayLike1<'_, f32, AllowTypeChange>,
    ) -> PyResult<()> {
        let body_up = to_vector::<3>(body_up, "body_up")?.into();
        let world_up = to_vector::<3>(world_up, "world_up")?.into();
        let result = self.inner.add_root_upright_prior(weight, body_up, world_up);
        result.map_err(to_py_err)
    }

    /// Pulls each DOF toward its value in the previous frame, with per-DOF
    /// weight `weight * Dof.weight_multiplier`. Inactive on a sequence's first
    /// frame.
    fn add_dof_smoothness_prior(&mut self, weight: f32) -> PyResult<()> {
        self.inner
            .add_dof_smoothness_prior(weight)
            .map_err(to_py_err)
    }

    /// Pulls the root position and rotation toward the previous frame's, with
    /// separate weights since their units differ (length vs radians). Inactive
    /// on a sequence's first frame.
    fn add_root_smoothness_prior(
        &mut self,
        position_weight: f32,
        rotation_weight: f32,
    ) -> PyResult<()> {
        let result = self
            .inner
            .add_root_smoothness_prior(position_weight, rotation_weight);
        result.map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// One frame's converged pose and, optionally, its linearization. Immutable.
#[pyclass(module = "quickik", frozen)]
pub(crate) struct SolverResult {
    inner: quickik_core::solver::SolverResult,
}

impl From<quickik_core::solver::SolverResult> for SolverResult {
    fn from(inner: quickik_core::solver::SolverResult) -> Self {
        SolverResult { inner }
    }
}

/// Converts core results to Python ones.
pub(crate) fn to_py_results(results: Vec<quickik_core::solver::SolverResult>) -> Vec<SolverResult> {
    results.into_iter().map(SolverResult::from).collect()
}

#[pymethods]
impl SolverResult {
    /// The converged pose (a copy).
    #[getter]
    fn state(&self) -> State {
        State {
            inner: self.inner.state.clone(),
        }
    }

    /// `(n_joints, 3)` world-frame keypoint positions at `state`, in joint
    /// order. `None` unless the solve had `with_fk=True`.
    #[getter]
    fn keypoint_pos<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        let keypoint_pos = self.inner.keypoint_pos.as_deref();
        keypoint_pos.map(|positions| points_to_py(py, positions))
    }

    /// Residual Jacobian `(n_rows, state_dim)`: one row block per observation,
    /// in the frame's input order, 3 rows for a 3D observation and 2 for a 2D
    /// one (already mapped through its projection). Weight-0 observations get
    /// rows too. Linearized at the pose just before the last iteration's
    /// update, which is small at convergence. `None` unless the solve had
    /// `with_grad=True`.
    #[getter]
    fn jacobian<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        let jacobian = self.inner.jacobian.as_ref();
        jacobian.map(|jacobian| matrix_to_py(py, jacobian))
    }

    /// Lower-triangular Cholesky factor `L` `(state_dim, state_dim)` of the
    /// normal-equations matrix (priors and damping included) at the same
    /// linearization as `jacobian`. `None` unless the solve had
    /// `with_grad=True` and that matrix was positive-definite.
    #[getter]
    fn cholesky_l<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        let cholesky = self.inner.cholesky_l.as_ref();
        cholesky.map(|cholesky| matrix_to_py(py, &cholesky.l()))
    }

    fn __repr__(&self) -> String {
        format!("SolverResult(state={:?})", self.inner.state)
    }
}

/// Solves one frame at a time, with the caller managing the state.
#[pyclass(module = "quickik")]
pub(crate) struct SingleFrameSolver {
    inner: quickik_core::solver::SingleFrameSolver,
}

#[pymethods]
impl SingleFrameSolver {
    /// Raises `ValueError` if `config` has smoothness priors (there is no
    /// previous frame) or root priors on a fixed-base tree.
    #[new]
    fn new(kinematic_tree: KinematicTree, config: SolverConfig) -> PyResult<Self> {
        let solver =
            quickik_core::solver::SingleFrameSolver::new(&kinematic_tree.inner, &config.inner);
        Ok(SingleFrameSolver {
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

    /// Refines `state` in place, starting from its current value, and returns
    /// the result. `with_grad` and `with_fk` request `SolverResult.jacobian`/
    /// `SolverResult.cholesky_l` and `SolverResult.keypoint_pos`
    /// respectively; each costs a little extra.
    #[pyo3(signature = (state, observations, with_grad=false, with_fk=false))]
    fn solve(
        &mut self,
        mut state: PyRefMut<'_, State>,
        observations: Vec<PyRef<'_, KeypointObservation>>,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<SolverResult> {
        let observations = extract(observations);
        let result = self
            .inner
            .solve(&mut state.inner, &observations, with_grad, with_fk);
        Ok(result.map_err(to_py_err)?.into())
    }
}
