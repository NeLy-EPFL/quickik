use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::body_plan::{KinematicTree, wxyz};
use crate::{to_vector, vector_to_py};

/// The pose being solved for.
///
/// The array properties return read-only copies, so modify a state by
/// assigning a whole new value (e.g. `state.root_pos = (0, 0, 1)`), not in
/// place.
#[pyclass(module = "quickik", from_py_object)]
#[derive(Clone)]
pub(crate) struct State {
    pub(crate) inner: quickik_core::state::State,
}

#[pymethods]
impl State {
    /// A state at the neutral pose: every DOF at its `Dof.neutral` value, root
    /// at the origin with identity rotation.
    #[staticmethod]
    fn neutral_pose(kinematic_tree: KinematicTree) -> Self {
        State {
            inner: quickik_core::state::State::neutral_pose(kinematic_tree.inner),
        }
    }

    /// The kinematic tree that defines the body.
    #[getter]
    fn kinematic_tree(&self) -> KinematicTree {
        KinematicTree {
            inner: self.inner.kinematic_tree.clone(),
        }
    }

    /// Position of the root joint in world coordinates. Kept at the origin for
    /// a fixed-base tree.
    #[getter]
    fn root_pos<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        read_only(vector_to_py(py, self.inner.root_pos.as_slice()))
    }

    #[setter]
    fn set_root_pos(&mut self, value: PyArrayLike1<'_, f32, AllowTypeChange>) -> PyResult<()> {
        self.inner.root_pos = Vector3::from(to_vector::<3>(value, "root_pos")?);
        Ok(())
    }

    /// Rotation of the root joint in world coordinates, as `(w, x, y, z)`.
    /// Normalized when set. Kept at identity for a fixed-base tree.
    #[getter]
    fn root_quat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        read_only(vector_to_py(py, &wxyz(&self.inner.root_quat)))
    }

    #[setter]
    fn set_root_quat(&mut self, value: PyArrayLike1<'_, f32, AllowTypeChange>) -> PyResult<()> {
        let [w, x, y, z] = to_vector::<4>(value, "root_quat")?;
        self.inner.root_quat = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
        Ok(())
    }

    /// Values of all joint DOFs (radians for hinges, length for slides), in the
    /// tree's joint order, then each joint's DOF order.
    #[getter]
    fn dof_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        read_only(vector_to_py(py, &self.inner.dof_values))
    }

    #[setter]
    fn set_dof_values(&mut self, value: PyArrayLike1<'_, f32, AllowTypeChange>) -> PyResult<()> {
        let values: Vec<f32> = value.as_array().iter().copied().collect();
        let n_dofs = self.inner.kinematic_tree.n_dofs();
        if values.len() != n_dofs {
            return Err(PyValueError::new_err(format!(
                "dof_values must have {n_dofs} elements, got {}",
                values.len()
            )));
        }
        self.inner.dof_values = values;
        Ok(())
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        let state = &self.inner;
        format!(
            "State(root_pos={:?}, root_quat={:?}, dof_values={:?})",
            state.root_pos.as_slice(),
            wxyz(&state.root_quat),
            state.dof_values
        )
    }
}

/// Marks `array` read-only, so that modifying it in place (which wouldn't
/// change the state it was copied from) raises instead of silently doing
/// nothing.
fn read_only(array: Bound<'_, PyArray1<f32>>) -> PyResult<Bound<'_, PyArray1<f32>>> {
    let kwargs = PyDict::new(array.py());
    kwargs.set_item("write", false)?;
    array.call_method("setflags", (), Some(&kwargs))?;
    Ok(array)
}
