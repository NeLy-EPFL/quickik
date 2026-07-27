use std::sync::Arc;

use pyo3::prelude::*;

use crate::catch_panic;

/// A kinematic tree (body plan), loaded from JSON.
#[pyclass(module = "quickik", from_py_object, frozen)]
#[derive(Clone)]
pub(crate) struct KinematicTree {
    pub(crate) inner: Arc<quickik_core::body_plan::KinematicTree>,
}

#[pymethods]
impl KinematicTree {
    /// Parses a body plan from a JSON string. Raises `ValueError` if the
    /// JSON is malformed or the body plan is invalid (e.g. no single root
    /// joint).
    #[staticmethod]
    fn from_json_str(json_str: &str) -> PyResult<Self> {
        catch_panic(|| KinematicTree {
            inner: Arc::new(quickik_core::body_plan::KinematicTree::from_json_str(
                json_str,
            )),
        })
    }

    /// Same as `from_json_str`, but reads the JSON from a file at `path`.
    #[staticmethod]
    fn from_json_file(path: &str) -> PyResult<Self> {
        catch_panic(|| KinematicTree {
            inner: Arc::new(quickik_core::body_plan::KinematicTree::from_json_file(path)),
        })
    }

    /// Number of joints in the tree.
    #[getter]
    fn n_joints(&self) -> usize {
        self.inner.n_joints()
    }

    /// Total number of rotational DOFs across all joints.
    #[getter]
    fn n_dofs(&self) -> usize {
        self.inner.n_dofs()
    }
}
