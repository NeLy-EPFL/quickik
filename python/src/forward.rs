use numpy::PyArray2;
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::state::State;
use crate::{matrix_to_py, points_to_py, to_py_err};

/// Buffers for `forward_kinematics`, allocated once per tree and reused across
/// calls.
#[pyclass(module = "quickik")]
pub(crate) struct ForwardKinematicsWorkspace {
    inner: quickik_core::forward::ForwardKinematicsWorkspace,
}

#[pymethods]
impl ForwardKinematicsWorkspace {
    #[new]
    fn new(kinematic_tree: KinematicTree) -> Self {
        ForwardKinematicsWorkspace {
            inner: quickik_core::forward::ForwardKinematicsWorkspace::new(&kinematic_tree.inner),
        }
    }

    #[getter]
    fn kinematic_tree(&self) -> KinematicTree {
        KinematicTree {
            inner: self.inner.kinematic_tree().clone(),
        }
    }

    /// `(n_joints, 3)` world-frame keypoint positions from the last
    /// `forward_kinematics` call, in joint order.
    #[getter]
    fn keypoint_positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        points_to_py(py, self.inner.keypoint_positions())
    }

    /// Dense `(3 * n_joints, state_dim)` Jacobian of `keypoint_positions`
    /// w.r.t. the state: keypoint `k`'s position is rows `3k` to `3k + 2`.
    fn jacobian<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        matrix_to_py(py, &self.inner.jacobian())
    }
}

/// Computes every keypoint's world position for `state`, and its Jacobian
/// w.r.t. the state. Results are read from `workspace`. Raises `ValueError` if
/// `state` doesn't have as many DOFs as the workspace's tree.
#[pyfunction]
pub(crate) fn forward_kinematics(
    mut workspace: PyRefMut<'_, ForwardKinematicsWorkspace>,
    state: PyRef<'_, State>,
) -> PyResult<()> {
    quickik_core::forward::forward_kinematics(&mut workspace.inner, &state.inner).map_err(to_py_err)
}
