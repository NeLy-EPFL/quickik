use nalgebra::DMatrix;
use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLike2, PyArrayLike3,
};
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::catch_panic;
use crate::observation::{
    Projection, as_row_major, observations_from_arrays, validate_position_weight_shapes,
};

/// Every `BatchedSolver.solve` item's converged pose and (optional)
/// linearization, as a struct of batched NumPy arrays, matching how a batch
/// is naturally represented on the PyTorch side. See
/// `quickik_core::batched_solver::BatchedSolverResult`'s docs for each
/// field's exact shape and ordering.
#[pyclass(module = "quickik", frozen)]
pub(crate) struct BatchedSolverResult {
    /// `(batch_size, n_dofs)`, in `kinematic_tree`'s own DOF order --
    /// unrelated to `keypoints_order`, since DOF order is already fully
    /// caller-controlled via how the tree was built.
    #[pyo3(get)]
    joint_angles: Py<PyArray2<f32>>,
    /// `(batch_size, 3)`.
    #[pyo3(get)]
    base_pos: Py<PyArray2<f32>>,
    /// `(batch_size, 4)`, `(w, x, y, z)`.
    #[pyo3(get)]
    base_quat: Py<PyArray2<f32>>,
    /// `(batch_size, n_joints, 3)`, in `kinematic_tree`'s internal joint order
    /// (*not* `keypoints_order`). `None` unless `solve` was called with
    /// `with_fk=True`.
    #[pyo3(get)]
    keypoint_pos: Option<Py<PyArray3<f32>>>,
    /// The residual Jacobian: `(batch_size, 3 * n_joints, state_dim)` for a 3D
    /// projection, or `(batch_size, 2 * n_joints, state_dim)` otherwise, in
    /// which case it's already mapped through that projection. Rows and columns
    /// are in `kinematic_tree`'s internal keypoint/state order (*not*
    /// `keypoints_order`). `None` unless `solve` was called with
    /// `with_grad=True`.
    #[pyo3(get)]
    jacobian: Option<Py<PyArray3<f32>>>,
    /// `(batch_size, state_dim, state_dim)`; zeroed for any item where `valid`
    /// is `False`. `None` unless `solve` was called with `with_grad=True`.
    #[pyo3(get)]
    cholesky_l: Option<Py<PyArray3<f32>>>,
    /// `(batch_size,)`; `False` where that item's last iteration wasn't
    /// positive-definite, so its `cholesky_l` can't be used for gradients.
    /// `None` unless `solve` was called with `with_grad=True`.
    #[pyo3(get)]
    valid: Option<Py<PyArray1<bool>>>,
}

/// Stacks one `DMatrix` per batch item into a `(batch_size, rows, cols)`
/// array. `shape` is taken from the matrices themselves rather than recomputed
/// here, since the Jacobian's row count depends on the solver's projection.
fn stack_matrices<'a>(
    py: Python<'_>,
    mats: impl ExactSizeIterator<Item = Option<&'a DMatrix<f32>>>,
    fallback_shape: (usize, usize),
) -> Py<PyArray3<f32>> {
    let batch_size = mats.len();
    let mut arr: Option<Array3<f32>> = None;
    for (i, mat) in mats.enumerate() {
        // Items with no matrix (a non-positive-definite Cholesky) stay zeroed.
        let Some(mat) = mat else { continue };
        let arr = arr.get_or_insert_with(|| Array3::zeros((batch_size, mat.nrows(), mat.ncols())));
        arr.index_axis_mut(Axis(0), i).assign(&as_row_major(mat));
    }
    arr.unwrap_or_else(|| Array3::zeros((batch_size, fallback_shape.0, fallback_shape.1)))
        .into_pyarray(py)
        .unbind()
}

fn to_py_result(
    py: Python<'_>,
    kinematic_tree: &quickik_core::body_plan::KinematicTree,
    result: quickik_core::batched_solver::BatchedSolverResult,
) -> BatchedSolverResult {
    let batch_size = result.joint_angles.len();
    let state_dim = kinematic_tree.state_dim();

    let joint_angles = Array2::from_shape_fn((batch_size, kinematic_tree.n_dofs()), |(i, d)| {
        result.joint_angles[i][d]
    });
    let base_pos = Array2::from_shape_fn((batch_size, 3), |(i, axis)| result.base_pos[i][axis]);
    let base_quat = Array2::from_shape_fn((batch_size, 4), |(i, c)| {
        let q = result.base_quat[i].quaternion();
        [q.w, q.i, q.j, q.k][c]
    });

    let keypoint_pos = result.keypoint_pos.map(|batch| {
        Array3::from_shape_fn(
            (batch_size, kinematic_tree.n_joints(), 3),
            |(i, k, axis)| batch[i][k][axis],
        )
        .into_pyarray(py)
        .unbind()
    });

    let jacobian = result
        .jacobian
        .map(|batch| stack_matrices(py, batch.iter().map(Some), (0, state_dim)));

    // `valid` marks the items whose last iteration was positive-definite;
    // the rest keep `cholesky_l`'s zeroed block.
    let valid = result.cholesky_l.as_ref().map(|batch| {
        Array1::from_iter(batch.iter().map(Option::is_some))
            .into_pyarray(py)
            .unbind()
    });
    let cholesky_l = result.cholesky_l.map(|batch| {
        let ls: Vec<Option<DMatrix<f32>>> = batch.into_iter().map(|c| c.map(|c| c.l())).collect();
        stack_matrices(py, ls.iter().map(Option::as_ref), (state_dim, state_dim))
    });

    BatchedSolverResult {
        joint_angles: joint_angles.into_pyarray(py).unbind(),
        base_pos: base_pos.into_pyarray(py).unbind(),
        base_quat: base_quat.into_pyarray(py).unbind(),
        keypoint_pos,
        jacobian,
        cholesky_l,
        valid,
    }
}

/// Solves a batch of fully independent sets of keypoint observations, for
/// training/inference with an autodiff framework (e.g. `quickik.torch`).
/// Every `solve` call starts each item from `kinematic_tree`'s neutral pose
/// (no warm-starting); see `quickik_core::batched_solver::BatchedSolver`'s
/// docs.
#[pyclass(module = "quickik")]
pub(crate) struct BatchedSolver {
    inner: quickik_core::batched_solver::BatchedSolver,
    kinematic_tree: KinematicTree,
}

#[pymethods]
impl BatchedSolver {
    /// `kinematic_tree` must be free-floating (not fixed-base), since
    /// `BatchedSolverResult` always reports `base_pos`/`base_quat`.
    ///
    /// `keypoints_order[i]` is the joint name (see `KinematicTree.joint_names`)
    /// that `solve`'s `positions`/`weights` keypoint axis position `i`
    /// corresponds to; every joint in `kinematic_tree` must appear in it
    /// exactly once.
    ///
    /// `n_workers`: number of threads in `solve`'s dedicated thread pool.
    /// A positive value is used directly, unless it exceeds the number of
    /// available cores: it's then clipped to that count and a warning is
    /// logged. A negative value counts backward from all available cores:
    /// `-1` (the default) uses all, `-2` uses all but one, etc. `0` is
    /// invalid.
    ///
    /// Raises `ValueError` if `kinematic_tree` is fixed-base,
    /// `keypoints_order` is malformed, or `n_workers` is `0`.
    #[new]
    #[pyo3(signature = (
        kinematic_tree, keypoints_order, projection=Projection::default(), n_iterations=10, neutral_weight=1e-3,
        position_tolerance=1e-3, angle_tolerance=1e-3, damping=1e-6, n_workers=-1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kinematic_tree: KinematicTree,
        keypoints_order: Vec<String>,
        projection: Projection,
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
        n_workers: isize,
    ) -> PyResult<Self> {
        let tree = &kinematic_tree.inner;
        let inner = catch_panic(|| {
            quickik_core::batched_solver::BatchedSolver::new(
                tree,
                projection.inner,
                n_iterations,
                neutral_weight,
                position_tolerance,
                angle_tolerance,
                damping,
                keypoints_order,
                n_workers,
            )
        })?;
        Ok(BatchedSolver {
            inner,
            kinematic_tree,
        })
    }

    /// Solves every item in `positions`/`weights` independently and in
    /// parallel, each starting from `kinematic_tree`'s neutral pose.
    ///
    /// `weights` is `(batch_size, n_joints)`; `positions` is `(batch_size,
    /// n_joints, 3)` for a 3D projection, or `(batch_size, n_joints, 2)`
    /// otherwise, both in this solver's own `keypoints_order` (*not*
    /// `kinematic_tree`'s internal joint order). A keypoint with `weight <=
    /// 0` (or NaN) is treated as missing. Any dtype is accepted and cast to
    /// `float32`, following NumPy's own casting rules.
    #[pyo3(signature = (positions, weights, with_grad=false, with_fk=false))]
    fn solve(
        &self,
        py: Python<'_>,
        positions: PyArrayLike3<'_, f32, AllowTypeChange>,
        weights: PyArrayLike2<'_, f32, AllowTypeChange>,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<BatchedSolverResult> {
        let positions_arr = positions.as_array();
        let weights_arr = weights.as_array();
        validate_position_weight_shapes(
            &positions_arr,
            &weights_arr,
            self.kinematic_tree.inner.n_joints(),
            !self.inner.projection().is_3d(),
        )?;
        let inner = &self.inner;
        let result = py.detach(|| {
            catch_panic(|| {
                let observations_array = observations_from_arrays(positions_arr, weights_arr);
                inner.solve(&observations_array, with_grad, with_fk)
            })
        })?;
        Ok(to_py_result(py, &self.kinematic_tree.inner, result))
    }

    /// Fixed at construction (read-only).
    #[getter]
    fn kinematic_tree(&self) -> KinematicTree {
        self.kinematic_tree.clone()
    }

    /// Fixed at construction (read-only).
    #[getter]
    fn projection(&self) -> Projection {
        Projection {
            inner: self.inner.projection(),
        }
    }

    /// `keypoint_to_joint_idx[i]` is `kinematic_tree`'s internal joint index
    /// that `solve`'s keypoint axis position `i` corresponds to (the
    /// resolved inverse of the by-name `keypoints_order` this solver was
    /// constructed with).
    #[getter]
    fn keypoint_to_joint_idx(&self) -> Vec<usize> {
        self.inner.keypoint_to_joint_idx().to_vec()
    }
}
