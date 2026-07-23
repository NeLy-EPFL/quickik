use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::observation::{
    KeypointObservation, Mapper, extract_mapper, extract_observations, mapper_to_py,
};
use crate::state::State;

/// Configuration for the inverse kinematics solver. Does not include the
/// mapper -- see [`Solver`]'s and [`SequenceSolver`](crate::high_level::SequenceSolver)'s
/// `mapper` argument.
#[pyclass(module = "fastik", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct SolverConfig {
    /// Number of Gauss-Newton steps per `solve` call. Also the cap on early
    /// termination -- see `position_tolerance`/`angle_tolerance`.
    #[pyo3(get, set)]
    n_iterations: usize,
    /// Levenberg-Marquardt damping added to the normal equations' diagonal,
    /// for numerical stability only; keep it very small (e.g. 1e-6).
    #[pyo3(get, set)]
    damping: f32,
    /// Weight pulling every joint angle toward the neutral pose. Improves
    /// robustness to missing/noisy keypoints, at the cost of some bias.
    #[pyo3(get, set)]
    neutral_pose_weight: f32,
    /// Stop iterating early once an update step's largest root-position
    /// component drops below this value, and the largest angle update drops
    /// below `angle_tolerance`. 0 disables early termination.
    #[pyo3(get, set)]
    position_tolerance: f32,
    /// Angle-space counterpart to `position_tolerance`, in radians.
    #[pyo3(get, set)]
    angle_tolerance: f32,
}

impl SolverConfig {
    pub(crate) fn as_rust(
        &self,
        mapper: Option<Mapper>,
    ) -> fastik_core::solver::SolverConfig<Mapper> {
        fastik_core::solver::SolverConfig {
            n_iterations: self.n_iterations,
            damping: self.damping,
            neutral_pose_weight: self.neutral_pose_weight,
            position_tolerance: self.position_tolerance,
            angle_tolerance: self.angle_tolerance,
            mapper,
        }
    }
}

#[pymethods]
impl SolverConfig {
    /// All arguments are optional; see the attributes above for their
    /// defaults and meaning.
    #[new]
    #[pyo3(signature = (n_iterations=10, damping=1e-6, neutral_pose_weight=1e-3, position_tolerance=1e-3, angle_tolerance=1e-3))]
    fn new(
        n_iterations: usize,
        damping: f32,
        neutral_pose_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
    ) -> Self {
        SolverConfig {
            n_iterations,
            damping,
            neutral_pose_weight,
            position_tolerance,
            angle_tolerance,
        }
    }
}

/// The inverse kinematics solver.
///
/// `mapper` is a `Camera`, an `XYView`, or `None` (the default, for 3D-only
/// observations); it's fixed for this `Solver`'s lifetime, mirroring Rust's
/// `Solver<M>` generic parameter -- there's no setter, only the read-only
/// `mapper` property.
///
/// `config` is a live, shared handle: `solver.config` always returns the
/// same Python `SolverConfig` object, so mutating it (e.g.
/// `solver.config.n_iterations = 5`) takes effect on the next `solve` call,
/// mirroring Rust's `pub config` field. Assigning `solver.config = other`
/// re-points it at `other` (which then also mutates in place, same as any
/// other Python object reference).
#[pyclass(module = "fastik")]
pub(crate) struct Solver {
    inner: fastik_core::solver::Solver<Mapper>,
    config: Py<SolverConfig>,
    mapper: Option<Mapper>,
}

#[pymethods]
impl Solver {
    /// `config` is copied in (see the `config` property to retune it
    /// afterward). Raises `ValueError` if `mapper` is not a `Camera`, an
    /// `XYView`, or `None`.
    #[new]
    #[pyo3(signature = (kinematic_tree, config, mapper=None))]
    fn new(
        py: Python<'_>,
        kinematic_tree: KinematicTree,
        config: SolverConfig,
        mapper: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mapper = extract_mapper(mapper.as_ref())?;
        Ok(Solver {
            inner: fastik_core::solver::Solver::new(&kinematic_tree.inner, config.as_rust(mapper)),
            config: Py::new(py, config)?,
            mapper,
        })
    }

    /// Runs up to `config.n_iterations` Gauss-Newton steps in place on
    /// `state`, given one `KeypointObservation` per joint (in
    /// `kinematic_tree.joints` order; use `KeypointObservation.missing()`
    /// for keypoints not observed this frame).
    fn solve(
        &mut self,
        py: Python<'_>,
        mut state: PyRefMut<'_, State>,
        observations: Vec<PyRef<'_, KeypointObservation>>,
    ) {
        self.sync_config(py);
        self.inner
            .solve(&mut state.inner, &extract_observations(observations));
    }

    /// The live config; see the class docstring for mutation semantics.
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

impl Solver {
    fn sync_config(&mut self, py: Python<'_>) {
        self.inner.config = self.config.borrow(py).as_rust(self.mapper);
    }
}
