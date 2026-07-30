use nalgebra::DMatrix;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::catch_panic;
use crate::observation::{
    KeypointObservation, Mapper, extract_mapper, extract_observations, mapper_to_py,
    positions_to_pyarray,
};
use crate::state::State;

/// Converts a Jacobian or Cholesky-factor matrix into an `(nrows, ncols)`
/// float32 NumPy array.
pub(crate) fn matrix_to_pyarray<'py>(
    py: Python<'py>,
    mat: &DMatrix<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let mut arr = Array2::<f32>::zeros((mat.nrows(), mat.ncols()));
    for r in 0..mat.nrows() {
        for c in 0..mat.ncols() {
            arr[[r, c]] = mat[(r, c)];
        }
    }
    arr.into_pyarray(py)
}

/// The converged pose (and, optionally, linearization) from one
/// `Solver.solve` call, or one item of a `SequenceSolver` sequence.
#[pyclass(module = "quickik", frozen)]
pub(crate) struct SolverResult {
    inner: quickik_core::solver::SolverResult,
}

impl SolverResult {
    pub(crate) fn new(inner: quickik_core::solver::SolverResult) -> Self {
        SolverResult { inner }
    }
}

#[pymethods]
impl SolverResult {
    /// Angles of all joint DOFs, in `KinematicTree`'s own order. Shorthand
    /// for `.state.dof_angles`.
    #[getter]
    fn dof_angles(&self) -> Vec<f32> {
        self.inner.state.dof_angles.clone()
    }

    /// Position of the root joint in world coordinates. Shorthand for
    /// `.state.root_pos`.
    #[getter]
    fn root_pos(&self) -> (f32, f32, f32) {
        let p = self.inner.state.root_pos;
        (p.x, p.y, p.z)
    }

    /// `(w, x, y, z)`. Shorthand for `.state.root_rot`.
    #[getter]
    fn root_rot(&self) -> (f32, f32, f32, f32) {
        let q = self.inner.state.root_rot.quaternion();
        (q.w, q.i, q.j, q.k)
    }

    /// The converged pose as a full `State` object (e.g. to feed into
    /// another `Solver.solve` call). Built on demand: prefer `dof_angles`/
    /// `root_pos`/`root_rot` directly if that's all you need, since those
    /// don't pay for constructing this.
    #[getter]
    fn state(&self) -> State {
        State {
            inner: self.inner.state.clone(),
        }
    }

    /// World-space keypoint positions (`(n_joints, 3)` float32, always 3D
    /// regardless of the solver's mapper), in `KinematicTree`'s joint order.
    /// `None` unless `solve` was called with `with_fk=True`.
    #[getter]
    fn keypoint_pos<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        self.inner
            .keypoint_pos
            .as_deref()
            .map(|p| positions_to_pyarray(py, p))
    }

    /// The keypoint-position Jacobian (`(3 * n_joints, state_dim)` float32)
    /// at (approximately) the converged pose -- see
    /// `quickik_core::solver::Solver::solve`'s docs for exactly which pose.
    /// `None` unless `solve` was called with `with_grad=True`.
    #[getter]
    fn jacobian<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        self.inner
            .jacobian
            .as_ref()
            .map(|j| matrix_to_pyarray(py, j))
    }

    /// Lower-triangular Cholesky factor `L` of the normal-equations matrix at
    /// the same linearization as `jacobian` (`jtj = L @ L.T`). `None` if
    /// `with_grad=False`, or if that linearization wasn't positive-definite
    /// (gradients can't be computed from this solve).
    #[getter]
    fn cholesky_l<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        self.inner
            .cholesky_l
            .as_ref()
            .map(|chol| matrix_to_pyarray(py, &chol.l()))
    }

    fn __repr__(&self) -> String {
        format!(
            "SolverResult(dof_angles={:?}, root_pos={:?}, root_rot={:?})",
            self.dof_angles(),
            self.root_pos(),
            self.root_rot()
        )
    }
}

/// The inverse kinematics solver.
///
/// `mapper` is a `Camera`, an `XYView`, or `None` (the default, for 3D-only
/// observations); it's fixed for this `Solver`'s lifetime, mirroring Rust's
/// `Solver<M>` generic parameter -- there's no setter, only the read-only
/// `mapper` property. The other tuning parameters (`n_iterations`,
/// `neutral_weight`, `position_tolerance`, `angle_tolerance`, `damping`) are
/// plain attributes, freely retunable between `solve` calls.
#[pyclass(module = "quickik")]
pub(crate) struct Solver {
    inner: quickik_core::solver::Solver<Mapper>,
    mapper: Mapper,
}

#[pymethods]
impl Solver {
    /// Raises `ValueError` if `mapper` is not a `Camera`, an `XYView`, or
    /// `None`.
    #[new]
    #[pyo3(signature = (
        kinematic_tree, mapper=None, n_iterations=10, neutral_weight=1e-3,
        position_tolerance=1e-3, angle_tolerance=1e-3, damping=1e-6,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kinematic_tree: KinematicTree,
        mapper: Option<Bound<'_, PyAny>>,
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    ) -> PyResult<Self> {
        let mapper = extract_mapper(mapper.as_ref())?;
        Ok(Solver {
            inner: quickik_core::solver::Solver::new(
                &kinematic_tree.inner,
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

    /// Runs up to `self.n_iterations` Gauss-Newton steps in place on `state`,
    /// given one `KeypointObservation` per joint (in `kinematic_tree.joints`
    /// order; use `KeypointObservation.missing()` for keypoints not observed
    /// this frame), and returns the converged pose. `with_grad`/`with_fk`
    /// gate `SolverResult.jacobian`/`SolverResult.cholesky_l` and
    /// `SolverResult.keypoint_pos` respectively -- each costs a little extra
    /// work, so only request what you'll use. Raises `ValueError` if
    /// `len(observations) != kinematic_tree.n_joints`.
    #[pyo3(signature = (state, observations, with_grad=false, with_fk=false))]
    fn solve(
        &mut self,
        mut state: PyRefMut<'_, State>,
        observations: Vec<PyRef<'_, KeypointObservation>>,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<SolverResult> {
        let observations = extract_observations(observations);
        let inner = &mut self.inner;
        catch_panic(move || {
            SolverResult::new(inner.solve(&mut state.inner, &observations, with_grad, with_fk))
        })
    }

    /// Fixed at construction (read-only); mutating the returned object has
    /// no effect on this solver.
    #[getter]
    fn mapper(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        mapper_to_py(py, self.mapper)
    }

    /// Number of Gauss-Newton steps per `solve` call. Also the cap on early
    /// termination: see `position_tolerance`/`angle_tolerance`.
    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.n_iterations
    }
    #[setter]
    fn set_n_iterations(&mut self, value: usize) {
        self.inner.n_iterations = value;
    }

    /// Weight pulling every joint angle toward the neutral pose. Improves
    /// robustness to missing/noisy keypoints, at the cost of some bias.
    #[getter]
    fn neutral_weight(&self) -> f32 {
        self.inner.neutral_weight
    }
    #[setter]
    fn set_neutral_weight(&mut self, value: f32) {
        self.inner.neutral_weight = value;
    }

    /// Stop iterating early once an update step's largest root-position
    /// component drops below this value, and the largest angle update drops
    /// below `angle_tolerance`. 0 disables early termination.
    #[getter]
    fn position_tolerance(&self) -> f32 {
        self.inner.position_tolerance
    }
    #[setter]
    fn set_position_tolerance(&mut self, value: f32) {
        self.inner.position_tolerance = value;
    }

    /// Angle-space counterpart to `position_tolerance`, in radians.
    #[getter]
    fn angle_tolerance(&self) -> f32 {
        self.inner.angle_tolerance
    }
    #[setter]
    fn set_angle_tolerance(&mut self, value: f32) {
        self.inner.angle_tolerance = value;
    }

    /// Levenberg-Marquardt damping added to the normal equations' diagonal,
    /// for numerical stability only; keep it very small (e.g. 1e-6).
    #[getter]
    fn damping(&self) -> f32 {
        self.inner.damping
    }
    #[setter]
    fn set_damping(&mut self, value: f32) {
        self.inner.damping = value;
    }
}
