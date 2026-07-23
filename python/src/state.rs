use pyo3::prelude::*;

use crate::body_plan::KinematicTree;

/// The pose being solved for.
#[pyclass(module = "quickik", from_py_object)]
#[derive(Clone)]
pub(crate) struct State {
    pub(crate) inner: quickik_core::state::State,
}

#[pymethods]
impl State {
    /// Creates a new state at the neutral pose for `kinematic_tree`.
    #[staticmethod]
    fn neutral_pose(kinematic_tree: KinematicTree) -> Self {
        State {
            inner: quickik_core::state::State::neutral_pose(kinematic_tree.inner),
        }
    }

    /// Angles of all joint DOFs, in body-plan order.
    #[getter]
    fn dof_angles(&self) -> Vec<f32> {
        self.inner.dof_angles.clone()
    }

    /// Position of the root joint in world coordinates.
    #[getter]
    fn root_pos(&self) -> (f32, f32, f32) {
        let p = self.inner.root_pos;
        (p.x, p.y, p.z)
    }

    /// `(w, x, y, z)`.
    #[getter]
    fn root_rot(&self) -> (f32, f32, f32, f32) {
        let q = self.inner.root_rot.quaternion();
        (q.w, q.i, q.j, q.k)
    }

    fn __repr__(&self) -> String {
        format!(
            "State(dof_angles={:?}, root_pos={:?}, root_rot={:?})",
            self.dof_angles(),
            self.root_pos(),
            self.root_rot()
        )
    }
}
