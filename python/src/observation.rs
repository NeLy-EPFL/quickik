use numpy::ndarray::{Array2, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Views a column-major nalgebra matrix as a row-major `ndarray`, without
/// copying. Both `numpy` and this crate's outputs are row-major, so every
/// matrix crossing into Python goes through here rather than an element-by-
/// element loop.
pub(crate) fn as_row_major(mat: &nalgebra::DMatrix<f32>) -> ArrayView2<'_, f32> {
    ArrayView2::from_shape((mat.ncols(), mat.nrows()), mat.as_slice())
        .expect("nalgebra matrices are contiguous")
        .reversed_axes()
}

fn vec_to_array<const N: usize>(v: &[f32], name: &str) -> PyResult<[f32; N]> {
    v.try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must have exactly {N} elements")))
}

/// What space a solver's keypoint observations live in. Build one with
/// `new_3d()`, `new_ortho_xy()`, or `new_pinhole_camera(...)`; a solver takes
/// it as its `projection` argument and holds it for its lifetime.
#[pyclass(module = "quickik", from_py_object, frozen)]
#[derive(Clone, Copy)]
pub(crate) struct Projection {
    pub(crate) inner: quickik_core::observation::Projection,
}

impl Default for Projection {
    fn default() -> Self {
        Projection {
            inner: quickik_core::observation::Projection::new_3d(),
        }
    }
}

#[pymethods]
impl Projection {
    /// Observations are 3D world positions; nothing is projected.
    #[staticmethod]
    fn new_3d() -> Self {
        Projection::default()
    }

    /// Observations are world X-Y coordinates with Z dropped.
    #[staticmethod]
    fn new_ortho_xy() -> Self {
        Projection {
            inner: quickik_core::observation::Projection::new_ortho_xy(),
        }
    }

    /// Observations are pixel coordinates from a calibrated pinhole camera.
    ///
    /// `fx`/`fy` are focal lengths in pixels and `cx`/`cy` the principal
    /// point. `world2cam_pos` must have exactly 3 elements and
    /// `world2cam_rot_mat` exactly 9 (row-major 3x3), giving the extrinsics as
    /// `p_cam = world2cam_rot_mat @ p_world + world2cam_pos`; raises
    /// `ValueError` otherwise.
    #[staticmethod]
    fn new_pinhole_camera(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        world2cam_pos: Vec<f32>,
        world2cam_rot_mat: Vec<f32>,
    ) -> PyResult<Self> {
        let pos = vec_to_array::<3>(&world2cam_pos, "world2cam_pos")?;
        let rot = vec_to_array::<9>(&world2cam_rot_mat, "world2cam_rot_mat")?;
        Ok(Projection {
            inner: quickik_core::observation::Projection::new_pinhole_camera(
                fx,
                fy,
                cx,
                cy,
                nalgebra::Vector3::from(pos),
                nalgebra::Matrix3::from_row_slice(&rot),
            ),
        })
    }

    /// Whether this is `new_3d()`, i.e. observations need no projecting.
    #[getter]
    fn is_3d(&self) -> bool {
        self.inner.is_3d()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Observation of a single keypoint: `missing()`, `position_3d(pos, weight)`,
/// or `position_2d(pos, weight)`.
#[pyclass(module = "quickik", from_py_object, frozen)]
#[derive(Clone, Copy)]
pub(crate) struct KeypointObservation {
    inner: quickik_core::observation::KeypointObservation,
}

#[pymethods]
impl KeypointObservation {
    /// Not observed this frame, e.g. occluded.
    #[staticmethod]
    fn missing() -> Self {
        KeypointObservation {
            inner: quickik_core::observation::KeypointObservation::Missing,
        }
    }

    /// A 3D world position, e.g. triangulated from multiple calibrated
    /// cameras. Raises `ValueError` if `pos` doesn't have exactly 3
    /// elements.
    #[staticmethod]
    fn position_3d(pos: Vec<f32>, weight: f32) -> PyResult<Self> {
        Ok(KeypointObservation {
            inner: quickik_core::observation::KeypointObservation::Position3D {
                obs_pos: nalgebra::Vector3::from(vec_to_array::<3>(&pos, "pos")?),
                weight,
            },
        })
    }

    /// A 2D position in whatever space the consuming `Solver`'s `projection`
    /// expects (e.g. camera pixel coordinates). Raises `ValueError` if `pos`
    /// doesn't have exactly 2 elements.
    #[staticmethod]
    fn position_2d(pos: Vec<f32>, weight: f32) -> PyResult<Self> {
        let [x, y] = vec_to_array::<2>(&pos, "pos")?;
        Ok(KeypointObservation {
            inner: quickik_core::observation::KeypointObservation::Position2D {
                obs_pos: nalgebra::Vector2::new(x, y),
                weight,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

pub(crate) fn extract_observations(
    observations: Vec<PyRef<'_, KeypointObservation>>,
) -> Vec<quickik_core::observation::KeypointObservation> {
    observations.iter().map(|obs| obs.inner).collect()
}

/// Builds one frame's-worth of `Position3D`/`Position2D`/`Missing`
/// observations per row of `positions`/`weights` (a keypoint with
/// `!(weight > 0.0)` (i.e. zero, negative, or NaN) is treated as
/// [`KeypointObservation::missing`]), without ever constructing a Python
/// `KeypointObservation` object, unlike [`extract_observations`], which
/// unwraps objects a caller already built one per keypoint. Used by
/// array-based entry points (`SequenceSolver.solve`, `BatchedSolver.solve`)
/// for callers that already have their data in numpy arrays, to avoid that
/// per-keypoint Python object construction and unwrapping.
///
/// `positions`'s last dimension selects the observation kind: 3 builds
/// `Position3D`, 2 builds `Position2D`. Call [`validate_position_weight_shapes`]
/// first to ensure this matches whether a projection is set. Used by
/// [`SequenceSolver`](crate::sequential_solver::SequenceSolver) and
/// [`BatchedSolver`](crate::batched_solver::BatchedSolver)'s array-based
/// `solve` methods.
pub(crate) fn observations_from_arrays(
    positions: ArrayView3<'_, f32>,
    weights: ArrayView2<'_, f32>,
) -> Vec<Vec<quickik_core::observation::KeypointObservation>> {
    let is_2d = positions.dim().2 == 2;
    positions
        .outer_iter()
        .zip(weights.outer_iter())
        .map(|(frame_positions, frame_weights)| {
            frame_positions
                .outer_iter()
                .zip(frame_weights.iter())
                .map(|(pos, &weight)| {
                    // A NaN weight counts as missing, not poison this frame's
                    // shared normal-equations matrices
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    if !(weight > 0.0) {
                        quickik_core::observation::KeypointObservation::Missing
                    } else if is_2d {
                        quickik_core::observation::KeypointObservation::Position2D {
                            obs_pos: nalgebra::Vector2::new(pos[0], pos[1]),
                            weight,
                        }
                    } else {
                        quickik_core::observation::KeypointObservation::Position3D {
                            obs_pos: nalgebra::Vector3::new(pos[0], pos[1], pos[2]),
                            weight,
                        }
                    }
                })
                .collect()
        })
        .collect()
}

/// Converts world-space keypoint positions (e.g. from
/// [`SolverResult::keypoint_pos`](quickik_core::solver::SolverResult::keypoint_pos))
/// into a `(n_joints, 3)` float32 NumPy array, in the same joint order.
pub(crate) fn positions_to_pyarray<'py>(
    py: Python<'py>,
    positions: &[nalgebra::Vector3<f32>],
) -> Bound<'py, PyArray2<f32>> {
    Array2::from_shape_fn((positions.len(), 3), |(i, axis)| positions[i][axis]).into_pyarray(py)
}

/// Checks that `positions`/`weights` have the shapes
/// [`observations_from_arrays`] expects: `(n_frames, n_joints)` for
/// `weights`, and for `positions`, `(n_frames, n_joints, 3)` if `is_2d`
/// is `false` (3D observations), or `(n_frames, n_joints, 2)` if `true` (2D
/// observations, projected by whichever projection the solver was constructed
/// with).
pub(crate) fn validate_position_weight_shapes(
    positions: &ArrayView3<'_, f32>,
    weights: &ArrayView2<'_, f32>,
    n_joints: usize,
    is_2d: bool,
) -> PyResult<()> {
    let (n_frames, n_keypoints, dim) = positions.dim();
    let expected_dim = if is_2d { 2 } else { 3 };
    if dim != expected_dim {
        return Err(PyValueError::new_err(format!(
            "positions must have shape (n_frames, n_keypoints, {expected_dim}) since this \
             solver's projection is {}, got last dimension {dim}",
            if is_2d { "2D" } else { "3D" }
        )));
    }
    if n_keypoints != n_joints {
        return Err(PyValueError::new_err(format!(
            "positions has {n_keypoints} keypoints, but kinematic_tree has {n_joints} joints"
        )));
    }
    if weights.dim() != (n_frames, n_keypoints) {
        return Err(PyValueError::new_err(format!(
            "weights must have shape (n_frames, n_keypoints) = ({n_frames}, {n_keypoints}), got {:?}",
            weights.dim()
        )));
    }
    Ok(())
}
