use std::sync::Arc;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;

use crate::{to_py_err, to_vector, vector_to_py};

/// Whether a `Dof` is a hinge (rotational) or slide (translational) DOF.
#[pyclass(module = "quickik", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum DofType {
    #[pyo3(name = "HINGE")]
    Hinge,
    #[pyo3(name = "SLIDE")]
    Slide,
}

impl From<quickik_core::body_plan::DofType> for DofType {
    fn from(dof_type: quickik_core::body_plan::DofType) -> Self {
        match dof_type {
            quickik_core::body_plan::DofType::Hinge => DofType::Hinge,
            quickik_core::body_plan::DofType::Slide => DofType::Slide,
        }
    }
}

impl From<DofType> for quickik_core::body_plan::DofType {
    fn from(dof_type: DofType) -> Self {
        match dof_type {
            DofType::Hinge => quickik_core::body_plan::DofType::Hinge,
            DofType::Slide => quickik_core::body_plan::DofType::Slide,
        }
    }
}

/// A single degree of freedom (DOF) of a `Joint`. Immutable.
#[pyclass(module = "quickik", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct Dof {
    pub(crate) inner: quickik_core::body_plan::Dof,
}

#[pymethods]
impl Dof {
    /// - `dof_type`: hinge (rotational) or slide (translational).
    /// - `axis`: rotation or translation axis in the joint's local frame,
    ///   normalized to unit length.
    /// - `neutral`: neutral value (radians for hinges, length for slides), also
    ///   the target of the DOF neutral prior.
    /// - `limits`: optional hard limits `(min, max)`, enforced by clamping
    ///   after each solver step.
    /// - `weight_multiplier`: static multiplier on every DOF prior's weight
    ///   for this DOF.
    #[new]
    #[pyo3(signature = (
        dof_type=DofType::Hinge, axis=None, neutral=0.0, limits=None, weight_multiplier=1.0,
    ))]
    fn new(
        dof_type: DofType,
        axis: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
        neutral: f32,
        limits: Option<(f32, f32)>,
        weight_multiplier: f32,
    ) -> PyResult<Self> {
        let axis = match axis {
            Some(axis) => Vector3::from(to_vector::<3>(axis, "axis")?),
            None => Vector3::z(),
        };
        Ok(Dof {
            inner: quickik_core::body_plan::Dof::new(
                dof_type.into(),
                axis,
                neutral,
                limits.map(|(min, max)| [min, max]),
                weight_multiplier,
            ),
        })
    }

    #[getter]
    fn dof_type(&self) -> DofType {
        self.inner.dof_type.into()
    }

    #[getter]
    fn axis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        vector_to_py(py, self.inner.axis.as_slice())
    }

    #[getter]
    fn neutral(&self) -> f32 {
        self.inner.neutral
    }

    #[getter]
    fn limits(&self) -> Option<(f32, f32)> {
        self.inner.limits.map(|[min, max]| (min, max))
    }

    #[getter]
    fn weight_multiplier(&self) -> f32 {
        self.inner.weight_multiplier
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// A joint with any number of DOFs. Also serves as a keypoint: its position is
/// the joint's origin. Immutable.
#[pyclass(module = "quickik", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct Joint {
    pub(crate) inner: quickik_core::body_plan::Joint,
}

#[pymethods]
impl Joint {
    /// - `name`: unique within a tree.
    /// - `parent`: index of the parent joint in the list passed to
    ///   `KinematicTree`, or `None` for the root.
    /// - `offset_pos`, `offset_quat`: position and rotation (`(w, x, y, z)`)
    ///   of the joint relative to its parent.
    /// - `dofs`: applied in order.
    /// - `weight_multiplier`: static multiplier on every observation's weight
    ///   for this joint's keypoint.
    #[new]
    #[pyo3(signature = (
        name, parent=None, offset_pos=None, offset_quat=None, dofs=Vec::new(),
        weight_multiplier=1.0,
    ))]
    fn new(
        name: String,
        parent: Option<usize>,
        offset_pos: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
        offset_quat: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
        dofs: Vec<Dof>,
        weight_multiplier: f32,
    ) -> PyResult<Self> {
        let offset_pos = match offset_pos {
            Some(pos) => Vector3::from(to_vector::<3>(pos, "offset_pos")?),
            None => Vector3::zeros(),
        };
        let offset_quat = match offset_quat {
            Some(quat) => {
                let [w, x, y, z] = to_vector::<4>(quat, "offset_quat")?;
                UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z))
            }
            None => UnitQuaternion::identity(),
        };
        Ok(Joint {
            inner: quickik_core::body_plan::Joint {
                name,
                parent,
                offset_pos,
                offset_quat,
                dofs: dofs.into_iter().map(|dof| dof.inner).collect(),
                weight_multiplier,
            },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn parent(&self) -> Option<usize> {
        self.inner.parent
    }

    #[getter]
    fn offset_pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        vector_to_py(py, self.inner.offset_pos.as_slice())
    }

    /// `(w, x, y, z)`.
    #[getter]
    fn offset_quat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        vector_to_py(py, &wxyz(&self.inner.offset_quat))
    }

    #[getter]
    fn dofs(&self) -> Vec<Dof> {
        let dofs = self.inner.dofs.iter();
        dofs.map(|&inner| Dof { inner }).collect()
    }

    #[getter]
    fn weight_multiplier(&self) -> f32 {
        self.inner.weight_multiplier
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// A unit quaternion's components as `[w, x, y, z]`.
pub(crate) fn wxyz(quat: &UnitQuaternion<f32>) -> [f32; 4] {
    [quat.w, quat.i, quat.j, quat.k]
}

/// A kinematic tree, i.e. body plan, or skeleton. Immutable.
///
/// Joint order is exactly the order of the joints passed in (the JSON order).
/// Observation joint indices, DOF order in `State.dof_values`, and Jacobian
/// columns all follow it.
#[pyclass(module = "quickik", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct KinematicTree {
    pub(crate) inner: Arc<quickik_core::body_plan::KinematicTree>,
}

#[pymethods]
impl KinematicTree {
    /// Constructs and validates a tree. `fixed_base`: the root is bolted to the
    /// world and has no DOFs of its own, rather than being a free-floating
    /// base. Raises `ValueError` unless exactly one joint has no parent, every
    /// parent index is valid, there are no cycles, joint names are unique, and
    /// every DOF axis is finite and nonzero.
    #[new]
    #[pyo3(signature = (joints, fixed_base=false))]
    fn new(joints: Vec<Joint>, fixed_base: bool) -> PyResult<Self> {
        let joints = joints.into_iter().map(|joint| joint.inner).collect();
        let tree = quickik_core::body_plan::KinematicTree::new(joints, fixed_base);
        Ok(Self::from(tree.map_err(to_py_err)?))
    }

    /// Parses a body plan from a JSON string. Raises `ValueError` if the JSON
    /// is malformed or the tree is invalid.
    #[staticmethod]
    fn from_json_str(json_str: &str) -> PyResult<Self> {
        let tree = quickik_core::body_plan::KinematicTree::from_json_str(json_str);
        Ok(Self::from(tree.map_err(to_py_err)?))
    }

    /// Same as `from_json_str`, reading the JSON from the file at `path`.
    /// Raises `OSError` if the file can't be read.
    #[staticmethod]
    fn from_json_file(path: std::path::PathBuf) -> PyResult<Self> {
        let tree = quickik_core::body_plan::KinematicTree::from_json_file(path);
        Ok(Self::from(tree.map_err(to_py_err)?))
    }

    #[getter]
    fn joints(&self) -> Vec<Joint> {
        let joints = self.inner.joints().iter();
        joints
            .map(|joint| Joint {
                inner: joint.clone(),
            })
            .collect()
    }

    #[getter]
    fn fixed_base(&self) -> bool {
        self.inner.fixed_base()
    }

    /// Index of the joint without a parent.
    #[getter]
    fn root_idx(&self) -> usize {
        self.inner.root_idx()
    }

    #[getter]
    fn n_joints(&self) -> usize {
        self.inner.n_joints()
    }

    #[getter]
    fn n_dofs(&self) -> usize {
        self.inner.n_dofs()
    }

    /// Number of free root DOFs: 6 for a free-floating base (3 position, 3
    /// rotation), or 0 for a fixed base.
    #[getter]
    fn n_root_dofs(&self) -> usize {
        self.inner.n_root_dofs()
    }

    /// Length of the solver's state vector: root DOFs first, then every joint
    /// DOF in order.
    #[getter]
    fn state_dim(&self) -> usize {
        self.inner.state_dim()
    }

    #[getter]
    fn joint_names(&self) -> Vec<&str> {
        self.inner.joint_names().collect()
    }

    /// Index of the joint called `name`. Raises `KeyError` if there is none.
    fn joint_index(&self, name: &str) -> PyResult<usize> {
        let index = self.inner.joint_index(name);
        index.ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }

    /// Index of each joint's first DOF in `State.dof_values`.
    #[getter]
    fn joint_dof_start_indices(&self) -> Vec<usize> {
        self.inner.joint_dof_start_indices().to_vec()
    }

    fn __repr__(&self) -> String {
        let tree = &self.inner;
        format!(
            "KinematicTree(n_joints={}, n_dofs={}, fixed_base={})",
            tree.n_joints(),
            tree.n_dofs(),
            if tree.fixed_base() { "True" } else { "False" },
        )
    }
}

impl From<quickik_core::body_plan::KinematicTree> for KinematicTree {
    fn from(tree: quickik_core::body_plan::KinematicTree) -> Self {
        KinematicTree {
            inner: Arc::new(tree),
        }
    }
}
