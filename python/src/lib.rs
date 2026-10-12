//! PyO3 bindings for the `quickik` crate. Classes, methods and argument names
//! mirror the Rust API, so the same program reads almost the same in both
//! languages. Python-specific differences:
//!
//! - Vectors and matrices are NumPy `float32` arrays. Arguments accept anything
//!   array-like (lists, tuples, arrays of any float dtype).
//! - Errors (`quickik::Error`) are raised as `ValueError`, or `OSError` for
//!   I/O errors.
//! - Projections are subclasses of one `Projection` base class (Rust: variants
//!   of the `Projection` enum, shared via `Arc`).
//! - Long-running solves release the GIL.

mod batched_solver;
mod body_plan;
mod forward;
mod observation;
mod sequential_solver;
mod solver;
mod state;

use numpy::ndarray::{Array1, Array2};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Converts a core error into the matching Python exception.
pub(crate) fn to_py_err(err: quickik_core::Error) -> PyErr {
    match err {
        quickik_core::Error::Io(err) => err.into(),
        err => PyValueError::new_err(err.to_string()),
    }
}

/// Extracts a length-`N` vector from any array-like argument.
pub(crate) fn to_vector<const N: usize>(
    value: PyArrayLike1<'_, f32, AllowTypeChange>,
    name: &str,
) -> PyResult<[f32; N]> {
    let array = value.as_array();
    let values: Vec<f32> = array.iter().copied().collect();
    values
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must have {N} elements")))
}

/// Extracts a 3x3 matrix from any array-like argument.
pub(crate) fn to_matrix3(
    value: PyArrayLike2<'_, f32, AllowTypeChange>,
    name: &str,
) -> PyResult<nalgebra::Matrix3<f32>> {
    let array = value.as_array();
    if array.dim() != (3, 3) {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (3, 3)"
        )));
    }
    Ok(nalgebra::Matrix3::from_fn(|i, j| array[(i, j)]))
}

/// A vector as a 1D NumPy array.
pub(crate) fn vector_to_py<'py>(py: Python<'py>, values: &[f32]) -> Bound<'py, PyArray1<f32>> {
    Array1::from(values.to_vec()).into_pyarray(py)
}

/// A list of 3D points as an `(n, 3)` NumPy array.
pub(crate) fn points_to_py<'py>(
    py: Python<'py>,
    points: &[nalgebra::Vector3<f32>],
) -> Bound<'py, PyArray2<f32>> {
    Array2::from_shape_fn((points.len(), 3), |(i, axis)| points[i][axis]).into_pyarray(py)
}

/// A matrix as a 2D NumPy array (nalgebra is column-major, NumPy row-major).
pub(crate) fn matrix_to_py<'py, R, C, S>(
    py: Python<'py>,
    matrix: &nalgebra::Matrix<f32, R, C, S>,
) -> Bound<'py, PyArray2<f32>>
where
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<f32, R, C>,
{
    Array2::from_shape_fn(matrix.shape(), |(i, j)| matrix[(i, j)]).into_pyarray(py)
}

#[pymodule]
fn quickik(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<body_plan::DofType>()?;
    m.add_class::<body_plan::Dof>()?;
    m.add_class::<body_plan::Joint>()?;
    m.add_class::<body_plan::KinematicTree>()?;
    m.add_class::<state::State>()?;
    m.add_class::<forward::ForwardKinematicsWorkspace>()?;
    m.add_function(wrap_pyfunction!(forward::forward_kinematics, m)?)?;
    m.add_class::<observation::Projection>()?;
    m.add_class::<observation::PinholeCamera>()?;
    m.add_class::<observation::OrthographicViewXY>()?;
    m.add_class::<observation::KeypointObservation>()?;
    m.add_class::<observation::ObservationSequence>()?;
    m.add_class::<solver::SolverConfig>()?;
    m.add_class::<solver::SolverResult>()?;
    m.add_class::<solver::SingleFrameSolver>()?;
    m.add_class::<sequential_solver::SequentialSolver>()?;
    m.add_class::<batched_solver::BatchedSolverResult>()?;
    m.add_class::<batched_solver::BatchedSolver>()?;
    Ok(())
}
