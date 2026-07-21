use std::sync::Arc;

use pyo3::prelude::*;

/// A kinematic tree (body plan), loaded from JSON.
#[pyclass(module = "fastik", from_py_object, frozen)]
#[derive(Clone)]
pub(crate) struct KinematicTree {
    pub(crate) inner: Arc<fastik_core::body_plan::KinematicTree>,
}

#[pymethods]
impl KinematicTree {
    #[staticmethod]
    fn from_json_str(json_str: &str) -> Self {
        KinematicTree {
            inner: Arc::new(fastik_core::body_plan::KinematicTree::from_json_str(json_str)),
        }
    }

    #[staticmethod]
    fn from_json_file(path: &str) -> Self {
        KinematicTree {
            inner: Arc::new(fastik_core::body_plan::KinematicTree::from_json_file(path)),
        }
    }

    #[getter]
    fn n_joints(&self) -> usize {
        self.inner.n_joints()
    }

    #[getter]
    fn n_dofs(&self) -> usize {
        self.inner.n_dofs()
    }
}
