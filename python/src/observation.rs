use numpy::ndarray::{ArrayView2, ArrayView3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Runtime stand-in for Rust's generic mapper type parameter `M`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Mapper {
    Camera(quickik_core::observation::Camera),
    XYView,
}

impl quickik_core::observation::Mapper3Dto2D for Mapper {
    fn project_3d_to_2d(
        &self,
        pos_world3d: &nalgebra::Vector3<f32>,
        jacobian_world3d: &nalgebra::DMatrix<f32>,
    ) -> (nalgebra::Vector2<f32>, nalgebra::DMatrix<f32>) {
        match self {
            Mapper::Camera(camera) => camera.project_3d_to_2d(pos_world3d, jacobian_world3d),
            Mapper::XYView => {
                quickik_core::observation::XYView.project_3d_to_2d(pos_world3d, jacobian_world3d)
            }
        }
    }
}

pub(crate) fn extract_mapper(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Mapper>> {
    let Some(obj) = obj else {
        return Ok(None);
    };
    if let Ok(camera) = obj.extract::<Camera>() {
        Ok(Some(Mapper::Camera(camera.as_rust())))
    } else if obj.extract::<XYView>().is_ok() {
        Ok(Some(Mapper::XYView))
    } else {
        Err(PyValueError::new_err(
            "mapper must be a Camera, an XYView, or None",
        ))
    }
}

pub(crate) fn mapper_to_py(py: Python<'_>, mapper: Option<Mapper>) -> PyResult<Py<PyAny>> {
    match mapper {
        None => Ok(py.None()),
        Some(Mapper::Camera(inner)) => Ok(Py::new(py, Camera::from_rust(inner))?.into_any()),
        Some(Mapper::XYView) => Ok(Py::new(py, XYView)?.into_any()),
    }
}

fn vec_to_array<const N: usize>(v: &[f32], name: &str) -> PyResult<[f32; N]> {
    v.try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must have exactly {N} elements")))
}

/// A pinhole camera mapper for 2D keypoint observations.
#[pyclass(module = "quickik", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct Camera {
    /// Focal length in pixels (x).
    #[pyo3(get, set)]
    fx: f32,
    /// Focal length in pixels (y).
    #[pyo3(get, set)]
    fy: f32,
    /// Principal point (x).
    #[pyo3(get, set)]
    cx: f32,
    /// Principal point (y).
    #[pyo3(get, set)]
    cy: f32,
    world2cam_pos: [f32; 3],
    /// Row-major 3x3.
    world2cam_rot_mat: [f32; 9],
}

impl Camera {
    fn as_rust(&self) -> quickik_core::observation::Camera {
        quickik_core::observation::Camera {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            world2cam_pos: nalgebra::Vector3::from(self.world2cam_pos),
            world2cam_rot_mat: nalgebra::Matrix3::from_row_slice(&self.world2cam_rot_mat),
        }
    }

    fn from_rust(camera: quickik_core::observation::Camera) -> Self {
        let p = camera.world2cam_pos;
        Camera {
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
            world2cam_pos: [p.x, p.y, p.z],
            world2cam_rot_mat: camera
                .world2cam_rot_mat
                .transpose()
                .as_slice()
                .try_into()
                .unwrap(),
        }
    }
}

#[pymethods]
impl Camera {
    /// `world2cam_pos` must have exactly 3 elements and `world2cam_rot_mat`
    /// exactly 9 (row-major 3x3); raises `ValueError` otherwise.
    #[new]
    fn new(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        world2cam_pos: Vec<f32>,
        world2cam_rot_mat: Vec<f32>,
    ) -> PyResult<Self> {
        Ok(Camera {
            fx,
            fy,
            cx,
            cy,
            world2cam_pos: vec_to_array(&world2cam_pos, "world2cam_pos")?,
            world2cam_rot_mat: vec_to_array(&world2cam_rot_mat, "world2cam_rot_mat")?,
        })
    }

    /// World-to-camera translation, as `(x, y, z)`.
    #[getter]
    fn world2cam_pos(&self) -> [f32; 3] {
        self.world2cam_pos
    }
    #[setter]
    fn set_world2cam_pos(&mut self, value: Vec<f32>) -> PyResult<()> {
        self.world2cam_pos = vec_to_array(&value, "world2cam_pos")?;
        Ok(())
    }

    /// Row-major 3x3, as 9 values.
    #[getter]
    fn world2cam_rot_mat(&self) -> Vec<f32> {
        self.world2cam_rot_mat.to_vec()
    }
    #[setter]
    fn set_world2cam_rot_mat(&mut self, value: Vec<f32>) -> PyResult<()> {
        self.world2cam_rot_mat = vec_to_array(&value, "world2cam_rot_mat")?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.as_rust())
    }
}

/// A mapper for 2D keypoints already reprojected to physical X-Y coordinates.
#[pyclass(module = "quickik", from_py_object, frozen)]
#[derive(Clone, Copy)]
pub(crate) struct XYView;

#[pymethods]
impl XYView {
    #[new]
    fn new() -> Self {
        XYView
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

    /// A 2D position in whatever space the consuming `Solver`'s `mapper`
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

/// Builds one frame's-worth of `Position3D`/`Missing` observations per row
/// of `positions`/`weights` (a keypoint with `weight <= 0.0` is treated as
/// [`KeypointObservation::missing`]), without ever constructing a Python
/// `KeypointObservation` object -- unlike [`extract_observations`], which
/// unwraps objects a caller already built one per keypoint. Used by
/// array-based entry points (e.g.
/// [`solve_sequence_segmented_parallel`](crate::high_level::solve_sequence_segmented_parallel))
/// for callers that already have their data in numpy arrays, to avoid that
/// per-keypoint Python object construction and unwrapping.
pub(crate) fn observations_from_arrays(
    positions: ArrayView3<'_, f32>,
    weights: ArrayView2<'_, f32>,
) -> Vec<Vec<quickik_core::observation::KeypointObservation>> {
    positions
        .outer_iter()
        .zip(weights.outer_iter())
        .map(|(frame_positions, frame_weights)| {
            frame_positions
                .outer_iter()
                .zip(frame_weights.iter())
                .map(|(pos, &weight)| {
                    if weight <= 0.0 {
                        quickik_core::observation::KeypointObservation::Missing
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
