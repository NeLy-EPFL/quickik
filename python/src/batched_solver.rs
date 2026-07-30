use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLike2, PyArrayLike3,
};
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::catch_panic;
use crate::observation::{
    Mapper, extract_mapper, mapper_to_py, observations_from_arrays, validate_position_weight_shapes,
};

/// Every `BatchedSolver.solve` item's converged pose and (optional)
/// linearization, as a struct of batched NumPy arrays -- matching how a batch
/// is naturally represented on the PyTorch side. See
/// `quickik_core::batched_solver::BatchedSolverResult`'s docs for each
/// field's exact shape and ordering.
#[pyclass(module = "quickik", frozen)]
pub(crate) struct BatchedSolverResult {
    joint_angles: Py<PyArray2<f32>>,
    base_pos: Py<PyArray2<f32>>,
    base_quat: Py<PyArray2<f32>>,
    keypoint_pos: Option<Py<PyArray3<f32>>>,
    jacobian: Option<Py<PyArray3<f32>>>,
    cholesky_l: Option<Py<PyArray3<f32>>>,
    valid: Option<Py<PyArray1<bool>>>,
}

#[pymethods]
impl BatchedSolverResult {
    /// `(batch_size, n_dofs)`, in `kinematic_tree`'s own DOF order --
    /// unrelated to `keypoints_order`, since DOF order is already fully
    /// caller-controlled via how the tree was built.
    #[getter]
    fn joint_angles(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.joint_angles.clone_ref(py)
    }

    /// `(batch_size, 3)`.
    #[getter]
    fn base_pos(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.base_pos.clone_ref(py)
    }

    /// `(batch_size, 4)`, `(w, x, y, z)`.
    #[getter]
    fn base_quat(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.base_quat.clone_ref(py)
    }

    /// `(batch_size, n_joints, 3)`, in `kinematic_tree`'s internal joint
    /// order (*not* `keypoints_order`). `None` unless `solve` was called with
    /// `with_fk=True`.
    #[getter]
    fn keypoint_pos(&self, py: Python<'_>) -> Option<Py<PyArray3<f32>>> {
        self.keypoint_pos.as_ref().map(|a| a.clone_ref(py))
    }

    /// `(batch_size, 3 * n_joints, state_dim)` -- always the raw 3D Jacobian
    /// regardless of `mapper`, in `kinematic_tree`'s internal keypoint/state
    /// order (*not* `keypoints_order`). `None` unless `solve` was called with
    /// `with_grad=True`.
    #[getter]
    fn jacobian(&self, py: Python<'_>) -> Option<Py<PyArray3<f32>>> {
        self.jacobian.as_ref().map(|a| a.clone_ref(py))
    }

    /// `(batch_size, state_dim, state_dim)`; zeroed for any item where
    /// `valid` is `False`. `None` unless `solve` was called with
    /// `with_grad=True`.
    #[getter]
    fn cholesky_l(&self, py: Python<'_>) -> Option<Py<PyArray3<f32>>> {
        self.cholesky_l.as_ref().map(|a| a.clone_ref(py))
    }

    /// `(batch_size,)`; `False` where that item's last iteration wasn't
    /// positive-definite, so its `cholesky_l` can't be used for gradients.
    /// `None` unless `solve` was called with `with_grad=True`.
    #[getter]
    fn valid(&self, py: Python<'_>) -> Option<Py<PyArray1<bool>>> {
        self.valid.as_ref().map(|a| a.clone_ref(py))
    }
}

fn to_py_result(
    py: Python<'_>,
    kinematic_tree: &quickik_core::body_plan::KinematicTree,
    result: quickik_core::batched_solver::BatchedSolverResult,
) -> BatchedSolverResult {
    let batch_size = result.joint_angles.len();
    let n_dofs = kinematic_tree.n_dofs();
    let n_joints = kinematic_tree.n_joints();
    let state_dim = kinematic_tree.state_dim();

    let mut joint_angles_arr = Array2::<f32>::zeros((batch_size, n_dofs));
    let mut base_pos_arr = Array2::<f32>::zeros((batch_size, 3));
    let mut base_quat_arr = Array2::<f32>::zeros((batch_size, 4));
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
    }

    let keypoint_pos = result.keypoint_pos.map(|batch_positions| {
        let mut arr = Array3::<f32>::zeros((batch_size, n_joints, 3));
        for (i, positions) in batch_positions.iter().enumerate() {
            let mut frame_view = arr.index_axis_mut(Axis(0), i);
            for (mut row, pos) in frame_view.rows_mut().into_iter().zip(positions) {
                row[0] = pos.x;
                row[1] = pos.y;
                row[2] = pos.z;
            }
        }
        arr.into_pyarray(py).unbind()
    });

    let jacobian = result.jacobian.map(|batch_jacobians| {
        let mut arr = Array3::<f32>::zeros((batch_size, 3 * n_joints, state_dim));
        for (i, jac) in batch_jacobians.iter().enumerate() {
            let mut view = arr.index_axis_mut(Axis(0), i);
            for r in 0..jac.nrows() {
                for c in 0..jac.ncols() {
                    view[[r, c]] = jac[(r, c)];
                }
            }
        }
        arr.into_pyarray(py).unbind()
    });

    let (cholesky_l, valid) = match result.cholesky_l {
        None => (None, None),
        Some(batch_cholesky_l) => {
            let mut chol_arr = Array3::<f32>::zeros((batch_size, state_dim, state_dim));
            let mut valid_arr = Array1::<bool>::from_elem(batch_size, false);
            for (i, chol) in batch_cholesky_l.iter().enumerate() {
                if let Some(chol) = chol {
                    valid_arr[i] = true;
                    let l = chol.l();
                    let mut view = chol_arr.index_axis_mut(Axis(0), i);
                    for r in 0..l.nrows() {
                        for c in 0..l.ncols() {
                            view[[r, c]] = l[(r, c)];
                        }
                    }
                }
            }
            (
                Some(chol_arr.into_pyarray(py).unbind()),
                Some(valid_arr.into_pyarray(py).unbind()),
            )
        }
    };

    BatchedSolverResult {
        joint_angles: joint_angles_arr.into_pyarray(py).unbind(),
        base_pos: base_pos_arr.into_pyarray(py).unbind(),
        base_quat: base_quat_arr.into_pyarray(py).unbind(),
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
    inner: quickik_core::batched_solver::BatchedSolver<Mapper>,
    kinematic_tree: KinematicTree,
    mapper: Mapper,
}

#[pymethods]
impl BatchedSolver {
    /// `kinematic_tree` must be free-floating (not fixed-base), since
    /// `BatchedSolverResult` always reports `base_pos`/`base_quat`.
    ///
    /// `keypoints_order[i]` is the joint name (see `KinematicTree.joint_names`)
    /// that `solve`'s `positions`/`weights` keypoint axis position `i`
    /// corresponds to; every joint in `kinematic_tree` must appear in it
    /// exactly once. Raises `ValueError` if `kinematic_tree` is fixed-base,
    /// `keypoints_order` is malformed, or `mapper` is not a `Camera`, an
    /// `XYView`, or `None`.
    #[new]
    #[pyo3(signature = (
        kinematic_tree, keypoints_order, mapper=None, n_iterations=10, neutral_weight=1e-3,
        position_tolerance=1e-3, angle_tolerance=1e-3, damping=1e-6,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kinematic_tree: KinematicTree,
        keypoints_order: Vec<String>,
        mapper: Option<Bound<'_, PyAny>>,
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    ) -> PyResult<Self> {
        let mapper = extract_mapper(mapper.as_ref())?;
        let tree = &kinematic_tree.inner;
        let inner = catch_panic(|| {
            quickik_core::batched_solver::BatchedSolver::new(
                tree,
                mapper,
                n_iterations,
                neutral_weight,
                position_tolerance,
                angle_tolerance,
                damping,
                keypoints_order,
            )
        })?;
        Ok(BatchedSolver {
            inner,
            kinematic_tree,
            mapper,
        })
    }

    /// Solves every item in `positions`/`weights` independently and in
    /// parallel, each starting from `kinematic_tree`'s neutral pose.
    ///
    /// `weights` is `(batch_size, n_joints)`; `positions` is `(batch_size,
    /// n_joints, 3)` if `mapper` is `None`, or `(batch_size, n_joints, 2)` if
    /// set, both in this solver's own `keypoints_order` (*not*
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
            self.mapper.is_set(),
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
    fn mapper(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        mapper_to_py(py, self.mapper)
    }

    /// `keypoint_to_joint_idx[i]` is `kinematic_tree`'s internal joint index
    /// that `solve`'s keypoint axis position `i` corresponds to -- the
    /// resolved inverse of the by-name `keypoints_order` this solver was
    /// constructed with.
    #[getter]
    fn keypoint_to_joint_idx(&self) -> Vec<usize> {
        self.inner.keypoint_to_joint_idx().to_vec()
    }
}
