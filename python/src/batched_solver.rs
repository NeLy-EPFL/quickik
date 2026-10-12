use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::body_plan::{KinematicTree, wxyz};
use crate::observation::ObservationSequence;
use crate::solver::SolverConfig;
use crate::to_py_err;

/// Every `BatchedSolver.solve` item's result, as stacked arrays (natural for
/// PyTorch). Immutable.
#[pyclass(module = "quickik", frozen)]
pub(crate) struct BatchedSolverResult {
    /// `(batch_size, n_dofs)`.
    #[pyo3(get)]
    dof_values: Py<PyArray2<f32>>,
    /// `(batch_size, 3)`.
    #[pyo3(get)]
    root_pos: Py<PyArray2<f32>>,
    /// `(batch_size, 4)`, `(w, x, y, z)`.
    #[pyo3(get)]
    root_quat: Py<PyArray2<f32>>,
    /// `(batch_size, n_joints, 3)` world-frame keypoint positions, in joint
    /// order. `None` unless `solve` had `with_fk=True`.
    #[pyo3(get)]
    keypoint_pos: Option<Py<PyArray3<f32>>>,
    /// `(batch_size, n_rows, state_dim)` residual Jacobians (see
    /// `SolverResult.jacobian`). `None` unless `solve` had `with_grad=True`.
    #[pyo3(get)]
    jacobian: Option<Py<PyArray3<f32>>>,
    /// `(batch_size, state_dim, state_dim)` Cholesky factors (see
    /// `SolverResult.cholesky_l`), zero where `cholesky_valid` is `False`.
    /// `None` unless `solve` had `with_grad=True`.
    #[pyo3(get)]
    cholesky_l: Option<Py<PyArray3<f32>>>,
    /// `(batch_size,)`: whether each item's normal-equations matrix was
    /// positive-definite, i.e. whether its `cholesky_l` is usable. `None`
    /// unless `solve` had `with_grad=True`.
    #[pyo3(get)]
    cholesky_valid: Option<Py<PyArray1<bool>>>,
}

/// Solves a batch of fully independent items in parallel, each starting from
/// the neutral pose, e.g. for training with autodiff (see `quickik.torch`).
#[pyclass(module = "quickik")]
pub(crate) struct BatchedSolver {
    inner: quickik_core::batched_solver::BatchedSolver,
}

#[pymethods]
impl BatchedSolver {
    /// Raises `ValueError` if the tree is fixed-base (results always report the
    /// root pose), if `config` has smoothness priors (items are independent),
    /// or if `n_workers` is 0. `n_workers` follows the convention of
    /// `SequentialSolver.solve_parallel`.
    #[new]
    #[pyo3(signature = (kinematic_tree, config, n_workers=-1))]
    fn new(
        kinematic_tree: KinematicTree,
        config: SolverConfig,
        n_workers: isize,
    ) -> PyResult<Self> {
        let solver = quickik_core::batched_solver::BatchedSolver::new(
            &kinematic_tree.inner,
            &config.inner,
            n_workers,
        );
        Ok(BatchedSolver {
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

    /// Solves every frame of `batch` as an independent item. With
    /// `with_grad=True`, raises `ValueError` unless every item has the same
    /// number of Jacobian rows (always true for
    /// `ObservationSequence.from_arrays`), so that they can be stacked.
    #[pyo3(signature = (batch, with_grad=false, with_fk=false))]
    fn solve(
        &self,
        py: Python<'_>,
        batch: PyRef<'_, ObservationSequence>,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<BatchedSolverResult> {
        let batch = &batch.inner;
        let inner = &self.inner;
        let result = py.detach(|| inner.solve(batch, with_grad, with_fk));
        let result = result.map_err(to_py_err)?;
        to_py_result(py, &self.inner, result)
    }
}

fn to_py_result(
    py: Python<'_>,
    solver: &quickik_core::batched_solver::BatchedSolver,
    result: quickik_core::batched_solver::BatchedSolverResult,
) -> PyResult<BatchedSolverResult> {
    let tree = solver.kinematic_tree();
    let batch_size = result.dof_values.len();
    let state_dim = tree.state_dim();

    let dof_values = Array2::from_shape_fn((batch_size, tree.n_dofs()), |(i, d)| {
        result.dof_values[i][d]
    });
    let root_pos = Array2::from_shape_fn((batch_size, 3), |(i, axis)| result.root_pos[i][axis]);
    let root_quat = Array2::from_shape_fn((batch_size, 4), |(i, c)| wxyz(&result.root_quat[i])[c]);
    let keypoint_pos = result.keypoint_pos.map(|batch| {
        let shape = (batch_size, tree.n_joints(), 3);
        let array = Array3::from_shape_fn(shape, |(i, k, axis)| batch[i][k][axis]);
        array.into_pyarray(py).unbind()
    });

    let jacobian = match result.jacobian {
        Some(batch) => {
            let n_rows = batch.first().map_or(0, |jacobian| jacobian.nrows());
            if batch.iter().any(|jacobian| jacobian.nrows() != n_rows) {
                return Err(PyValueError::new_err(
                    "can't stack Jacobians: batch items have different numbers of observation \
                     rows (use ObservationSequence.from_arrays for a fixed layout)",
                ));
            }
            let mut array = Array3::zeros((batch_size, n_rows, state_dim));
            for (mut item, jacobian) in array.axis_iter_mut(Axis(0)).zip(&batch) {
                item.indexed_iter_mut()
                    .for_each(|(ij, value)| *value = jacobian[ij]);
            }
            Some(array.into_pyarray(py).unbind())
        }
        None => None,
    };

    let (cholesky_l, cholesky_valid) = match result.cholesky_l {
        Some(batch) => {
            let mut array = Array3::zeros((batch_size, state_dim, state_dim));
            for (mut item, cholesky) in array.axis_iter_mut(Axis(0)).zip(&batch) {
                if let Some(cholesky) = cholesky {
                    let l = cholesky.l();
                    item.indexed_iter_mut()
                        .for_each(|(ij, value)| *value = l[ij]);
                }
            }
            let valid = Array1::from_iter(batch.iter().map(Option::is_some));
            (
                Some(array.into_pyarray(py).unbind()),
                Some(valid.into_pyarray(py).unbind()),
            )
        }
        None => (None, None),
    };

    Ok(BatchedSolverResult {
        dof_values: dof_values.into_pyarray(py).unbind(),
        root_pos: root_pos.into_pyarray(py).unbind(),
        root_quat: root_quat.into_pyarray(py).unbind(),
        keypoint_pos,
        jacobian,
        cholesky_l,
        cholesky_valid,
    })
}
