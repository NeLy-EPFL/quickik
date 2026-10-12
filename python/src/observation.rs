use std::sync::Arc;

use nalgebra::{Vector2, Vector3};
use numpy::ndarray::{ArrayView, CowArray, Dimension};
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayLike3};
use numpy::{PyArrayLike4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;

use crate::{matrix_to_py, to_matrix3, to_py_err, to_vector, vector_to_py};

use quickik_core::observation::{ObservedPosition, Projection as CoreProjection};

// =============================================================================
//  Projections
// =============================================================================

/// Maps a 3D world position to 2D, e.g. camera pixel coordinates. Not
/// constructible itself: build one of its subclasses, `PinholeCamera` or
/// `OrthographicViewXY`. Immutable.
///
/// Build each projection (e.g. each camera) once and pass the same object to
/// every observation made with it.
#[pyclass(module = "quickik", subclass, frozen)]
pub(crate) struct Projection {
    pub(crate) inner: Arc<CoreProjection>,
}

#[pymethods]
impl Projection {
    /// Returns `(uv, d_uv_d_pos)`: the 2D position `(2,)` that the world
    /// position `pos_3d` projects to, and its derivative `(2, 3)` w.r.t.
    /// `pos_3d`.
    #[allow(clippy::type_complexity)]
    fn project<'py>(
        &self,
        py: Python<'py>,
        pos_3d: PyArrayLike1<'_, f32, AllowTypeChange>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
        let pos_3d = Vector3::from(to_vector::<3>(pos_3d, "pos_3d")?);
        let (uv, d_uv_d_pos) = self.inner.project(&pos_3d);
        Ok((
            vector_to_py(py, uv.as_slice()),
            matrix_to_py(py, &d_uv_d_pos),
        ))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// A pinhole camera, for keypoints observed as pixel coordinates:
/// `p_cam = world2cam_rot_mat @ p_world + world2cam_pos`, then
/// `uv = (fx * x / z + cx, fy * y / z + cy)`.
#[pyclass(module = "quickik", extends = Projection, frozen)]
pub(crate) struct PinholeCamera {
    camera: quickik_core::observation::PinholeCamera,
}

#[pymethods]
impl PinholeCamera {
    /// `fx`, `fy` are the focal lengths and `cx`, `cy` the principal point, in
    /// pixels. `world2cam_pos` `(3,)` and `world2cam_rot_mat` `(3, 3)` are the
    /// extrinsics, as above.
    #[new]
    fn new(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        world2cam_pos: PyArrayLike1<'_, f32, AllowTypeChange>,
        world2cam_rot_mat: PyArrayLike2<'_, f32, AllowTypeChange>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let camera = quickik_core::observation::PinholeCamera::new(
            fx,
            fy,
            cx,
            cy,
            Vector3::from(to_vector::<3>(world2cam_pos, "world2cam_pos")?),
            to_matrix3(world2cam_rot_mat, "world2cam_rot_mat")?,
        );
        let base = Projection {
            inner: Arc::new(camera.into()),
        };
        Ok(PyClassInitializer::from(base).add_subclass(PinholeCamera { camera }))
    }

    #[getter]
    fn fx(&self) -> f32 {
        self.camera.fx
    }

    #[getter]
    fn fy(&self) -> f32 {
        self.camera.fy
    }

    #[getter]
    fn cx(&self) -> f32 {
        self.camera.cx
    }

    #[getter]
    fn cy(&self) -> f32 {
        self.camera.cy
    }

    #[getter]
    fn world2cam_pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        vector_to_py(py, self.camera.world2cam_pos.as_slice())
    }

    #[getter]
    fn world2cam_rot_mat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        matrix_to_py(py, &self.camera.world2cam_rot_mat)
    }
}

/// An orthographic view along world Z: `uv = scale * (x, y) + offset`.
#[pyclass(module = "quickik", extends = Projection, frozen)]
pub(crate) struct OrthographicViewXY {
    view: quickik_core::observation::OrthographicViewXY,
}

#[pymethods]
impl OrthographicViewXY {
    #[new]
    #[pyo3(signature = (scale=1.0, offset=None))]
    fn new(
        scale: f32,
        offset: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let offset = match offset {
            Some(offset) => Vector2::from(to_vector::<2>(offset, "offset")?),
            None => Vector2::zeros(),
        };
        let view = quickik_core::observation::OrthographicViewXY::new(scale, offset);
        let base = Projection {
            inner: Arc::new(view.into()),
        };
        Ok(PyClassInitializer::from(base).add_subclass(OrthographicViewXY { view }))
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.view.scale
    }

    #[getter]
    fn offset<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        vector_to_py(py, self.view.offset.as_slice())
    }
}

/// Wraps a core projection in the matching Python subclass.
fn projection_to_py(py: Python<'_>, projection: &Arc<CoreProjection>) -> PyResult<Py<PyAny>> {
    let base = PyClassInitializer::from(Projection {
        inner: Arc::clone(projection),
    });
    Ok(match **projection {
        CoreProjection::PinholeCamera(camera) => {
            Py::new(py, base.add_subclass(PinholeCamera { camera }))?.into_any()
        }
        CoreProjection::OrthographicViewXY(view) => {
            Py::new(py, base.add_subclass(OrthographicViewXY { view }))?.into_any()
        }
    })
}

// =============================================================================
//  Observations
// =============================================================================

/// One observation of one joint's keypoint. Immutable.
///
/// A frame's observations are a flat list in any order. A joint may appear any
/// number of times (e.g. once per camera) or not at all (e.g. occluded): leave
/// missing keypoints out. An observation with weight 0 contributes nothing to
/// the solve but still gets its rows in `SolverResult.jacobian`.
#[pyclass(module = "quickik", frozen)]
pub(crate) struct KeypointObservation {
    pub(crate) inner: quickik_core::observation::KeypointObservation,
}

#[pymethods]
impl KeypointObservation {
    /// A 3D world position `pos` `(3,)`, e.g. triangulated from several
    /// cameras. `weight` is multiplied by the joint's `weight_multiplier` to
    /// give the final weight.
    #[staticmethod]
    #[pyo3(signature = (joint_idx, pos, weight=1.0))]
    fn position_3d(
        joint_idx: usize,
        pos: PyArrayLike1<'_, f32, AllowTypeChange>,
        weight: f32,
    ) -> PyResult<Self> {
        let pos = Vector3::from(to_vector::<3>(pos, "pos")?);
        Ok(KeypointObservation {
            inner: quickik_core::observation::KeypointObservation::position_3d(
                joint_idx, pos, weight,
            ),
        })
    }

    /// A 2D position `pos` `(2,)` in the space `projection` maps into.
    /// `weight` is as in `position_3d`.
    #[staticmethod]
    #[pyo3(signature = (joint_idx, pos, projection, weight=1.0))]
    fn position_2d(
        joint_idx: usize,
        pos: PyArrayLike1<'_, f32, AllowTypeChange>,
        projection: PyRef<'_, Projection>,
        weight: f32,
    ) -> PyResult<Self> {
        let pos = Vector2::from(to_vector::<2>(pos, "pos")?);
        Ok(KeypointObservation {
            inner: quickik_core::observation::KeypointObservation::position_2d(
                joint_idx,
                pos,
                Arc::clone(&projection.inner),
                weight,
            ),
        })
    }

    #[getter]
    fn joint_idx(&self) -> usize {
        self.inner.joint_idx
    }

    /// The observed position: `(3,)` for a 3D observation, `(2,)` for a 2D
    /// one.
    #[getter]
    fn pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        match &self.inner.position {
            ObservedPosition::Position3D(pos) => vector_to_py(py, pos.as_slice()),
            ObservedPosition::Position2D(pos, _) => vector_to_py(py, pos.as_slice()),
        }
    }

    /// The projection of a 2D observation, or `None` for a 3D one.
    #[getter]
    fn projection(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.position {
            ObservedPosition::Position3D(_) => Ok(None),
            ObservedPosition::Position2D(_, projection) => {
                projection_to_py(py, projection).map(Some)
            }
        }
    }

    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

type CoreObservation = quickik_core::observation::KeypointObservation;

/// Converts Python observations (e.g. one frame's) to the core type.
pub(crate) fn extract(observations: Vec<PyRef<'_, KeypointObservation>>) -> Vec<CoreObservation> {
    let observations = observations.iter();
    observations
        .map(|observation| observation.inner.clone())
        .collect()
}

// =============================================================================
//  Sequences
// =============================================================================

/// Many frames of observations, e.g. a recording (for `SequentialSolver`) or a
/// batch of independent items (for `BatchedSolver`). Immutable.
#[pyclass(module = "quickik", frozen)]
pub(crate) struct ObservationSequence {
    pub(crate) inner: quickik_core::observation::ObservationSequence,
}

#[pymethods]
impl ObservationSequence {
    /// One list of observations per frame. Frames may differ in which and how
    /// many observations they have (e.g. moving cameras).
    #[staticmethod]
    fn from_lists(frames: Vec<Vec<PyRef<'_, KeypointObservation>>>) -> Self {
        let frames = frames.into_iter().map(extract).collect();
        ObservationSequence {
            inner: quickik_core::observation::ObservationSequence::from_lists(frames),
        }
    }

    /// Same as `from_lists`, with all frames concatenated: frame `i` is the
    /// next `frame_lengths[i]` observations. Raises `ValueError` if
    /// `frame_lengths` doesn't add up to `len(observations)`.
    #[staticmethod]
    fn from_flat(
        observations: Vec<PyRef<'_, KeypointObservation>>,
        frame_lengths: Vec<usize>,
    ) -> PyResult<Self> {
        let sequence = quickik_core::observation::ObservationSequence::from_flat(
            extract(observations),
            &frame_lengths,
        );
        Ok(ObservationSequence {
            inner: sequence.map_err(to_py_err)?,
        })
    }

    /// Builds frames with a fixed layout from arrays, much faster than building
    /// one `KeypointObservation` per keypoint per frame. Per frame: the 3D
    /// block (all joints, in joint order) if given, then one 2D block per view
    /// in `projections` order.
    ///
    /// - `pos_3d`: `(n_frames, n_joints, 3)`, with `weights_3d`:
    ///   `(n_frames, n_joints)`. Either both or neither.
    /// - `pos_2d`: `(n_frames, n_views, n_joints, 2)`, with `weights_2d`:
    ///   `(n_frames, n_views, n_joints)`, where `n_views = len(projections)`.
    ///   Required if and only if `projections` is non-empty.
    ///
    /// Every entry is kept, so every frame's Jacobian has the same rows. A
    /// weight that isn't positive (including NaN) marks a missing keypoint:
    /// it's stored as weight 0 with a zeroed position.
    #[staticmethod]
    #[pyo3(signature = (
        pos_3d=None, weights_3d=None, projections=Vec::new(), pos_2d=None, weights_2d=None,
    ))]
    fn from_arrays(
        pos_3d: Option<PyArrayLike3<'_, f32, AllowTypeChange>>,
        weights_3d: Option<PyArrayLike2<'_, f32, AllowTypeChange>>,
        projections: Vec<PyRef<'_, Projection>>,
        pos_2d: Option<PyArrayLike4<'_, f32, AllowTypeChange>>,
        weights_2d: Option<PyArrayLike3<'_, f32, AllowTypeChange>>,
    ) -> PyResult<Self> {
        // Every array's expected shape follows from these sizes.
        let n_views = projections.len();
        let (n_frames, n_joints) = match (&pos_3d, &pos_2d) {
            (Some(pos), _) => (pos.shape()[0], pos.shape()[1]),
            (None, Some(pos)) => (pos.shape()[0], pos.shape()[2]),
            (None, None) => return Err(PyValueError::new_err("pos_3d or pos_2d must be given")),
        };
        check_shape(
            "pos_3d",
            pos_3d.as_ref().map(|a| a.shape()),
            &[n_frames, n_joints, 3],
        )?;
        check_shape(
            "weights_3d",
            weights_3d.as_ref().map(|a| a.shape()),
            &[n_frames, n_joints],
        )?;
        let expected = [n_frames, n_views, n_joints, 2];
        check_shape("pos_2d", pos_2d.as_ref().map(|a| a.shape()), &expected)?;
        let expected = [n_frames, n_views, n_joints];
        check_shape(
            "weights_2d",
            weights_2d.as_ref().map(|a| a.shape()),
            &expected,
        )?;

        let pos_3d = pos_3d.as_ref().map(|a| contiguous(a.as_array()));
        let weights_3d = weights_3d.as_ref().map(|a| contiguous(a.as_array()));
        let pos_2d = pos_2d.as_ref().map(|a| contiguous(a.as_array()));
        let weights_2d = weights_2d.as_ref().map(|a| contiguous(a.as_array()));
        let projections: Vec<_> = projections.iter().map(|p| Arc::clone(&p.inner)).collect();
        let sequence = quickik_core::observation::ObservationSequence::from_arrays(
            n_frames,
            n_joints,
            pos_3d.as_ref().map(as_slice),
            weights_3d.as_ref().map(as_slice),
            &projections,
            pos_2d.as_ref().map(as_slice),
            weights_2d.as_ref().map(as_slice),
        );
        Ok(ObservationSequence {
            inner: sequence.map_err(to_py_err)?,
        })
    }

    /// Number of frames.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("ObservationSequence(n_frames={})", self.inner.len())
    }
}

fn check_shape(name: &str, shape: Option<&[usize]>, expected: &[usize]) -> PyResult<()> {
    match shape {
        Some(shape) if shape != expected => Err(PyValueError::new_err(format!(
            "{name} has shape {shape:?}, expected {expected:?}"
        ))),
        _ => Ok(()),
    }
}

/// `array` in standard (C-contiguous) layout, copied only if needed.
fn contiguous<'a, D: Dimension>(array: ArrayView<'a, f32, D>) -> CowArray<'a, f32, D> {
    match array.is_standard_layout() {
        true => array.into(),
        false => array.as_standard_layout().into_owned().into(),
    }
}

fn as_slice<'a, D: Dimension>(array: &'a CowArray<'_, f32, D>) -> &'a [f32] {
    array
        .as_slice()
        .expect("standard layout arrays are contiguous")
}
