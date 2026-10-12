//! This module defines data structures describing the kinematic tree.

use std::collections::{HashMap, VecDeque};

use nalgebra::{Unit, UnitQuaternion, Vector3};
use serde::Deserialize;

use crate::error::{Result, bail};
use crate::utils::quat_from_wxyz;

// =============================================================================
//  Data structures for the actual algorithm
// =============================================================================

/// DOFs of a free-floating root: 3 for position, 3 for rotation. See
/// [`KinematicTree::n_root_dofs`], which is `0` instead for a fixed-base tree.
const N_ROOT_DOFS: usize = 6;

/// Whether a [`Dof`] is a hinge (rotational) or slide (translational) DOF.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DofType {
    Hinge,
    Slide,
}

/// A single degree of freedom (DOF) on the kinematic tree.
#[derive(Clone, Copy, Debug)]
pub struct Dof {
    /// Whether this is a hinge or slide DOF.
    pub dof_type: DofType,
    /// Rotational or translational axis in the joint's local frame (i.e.
    /// relative to the joint's `offset_quat`). Build it with
    /// `Unit::new_normalize`.
    pub axis: Unit<Vector3<f32>>,
    /// Neutral value of the DOF: an angle in radians for a hinge DOF, or a
    /// position for a slide DOF. Also the target of the DOF neutral prior.
    pub neutral: f32,
    /// Optional hard limits `[min, max]`, enforced by clamping after each
    /// solver step. Unbounded if `None`.
    pub limits: Option<[f32; 2]>,
    /// Static multiplier on the weight of every DOF prior (neutral pose and
    /// smoothness) for this DOF.
    pub weight_multiplier: f32,
}

impl Dof {
    /// Constructs a DOF, normalizing `axis` to unit length.
    pub fn new(
        dof_type: DofType,
        axis: Vector3<f32>,
        neutral: f32,
        limits: Option<[f32; 2]>,
        weight_multiplier: f32,
    ) -> Self {
        Self {
            dof_type,
            axis: Unit::new_normalize(axis),
            neutral,
            limits,
            weight_multiplier,
        }
    }
}

/// An anatomical joint with any number of hinge and/or slide DOFs. Also serves
/// as a keypoint: its position is the joint's origin.
#[derive(Clone, Debug)]
pub struct Joint {
    /// Joint name. Must be unique within a tree.
    pub name: String,
    /// Index of the parent joint in the list passed to [`KinematicTree::new`],
    /// or `None` for the root. The JSON format refers to parents by name
    /// instead.
    pub parent: Option<usize>,
    /// Offset of the joint from its parent (i.e. length of the body segment
    /// connecting the parent to this joint).
    pub offset_pos: Vector3<f32>,
    /// Offset rotation of the joint from its parent (i.e. "curvature" of the
    /// body segment connecting the parent to this joint).
    pub offset_quat: UnitQuaternion<f32>,
    /// DOFs at this joint, applied in order (3D transformations don't
    /// commute). Can be empty.
    pub dofs: Vec<Dof>,
    /// Static multiplier on the weight of every observation of this joint's
    /// keypoint, e.g. lower for keypoints that are generally hard to detect.
    pub weight_multiplier: f32,
}

/// A kinematic tree, i.e. body plan, or skeleton. Immutable after
/// construction.
///
/// Joint order is exactly the order of the `joints` passed to
/// [`new`](Self::new) (the JSON order). This is the only joint order users see:
/// observation joint indices, DOF order in
/// [`State::dof_values`](crate::state::State::dof_values), and Jacobian
/// columns all follow it. Joints may be listed in any order; a parent-before-
/// child traversal order is computed internally.
#[derive(Clone, Debug)]
pub struct KinematicTree {
    joints: Vec<Joint>,
    fixed_base: bool,
    root_idx: usize,
    n_dofs: usize,
    joint_dof_start_indices: Vec<usize>,
    name_to_idx: HashMap<String, usize>,

    // Caches computed once in `new`, for the solver's hot loop.
    /// Parent-before-child visiting order (BFS from the root).
    traversal_order: Vec<usize>,
    /// Per joint: the state-vector indices that can move its keypoint, i.e.
    /// the root DOFs (if any) plus every DOF on its strict ancestors, sorted.
    upstream_state_indices: Vec<Vec<usize>>,
    /// Per-DOF properties in flat DOF order.
    dof_types: Vec<DofType>,
    dof_limits: Vec<Option<[f32; 2]>>,
    dof_neutrals: Vec<f32>,
    dof_weight_multipliers: Vec<f32>,
    /// Per-joint `Joint::weight_multiplier`.
    joint_weight_multipliers: Vec<f32>,
}

impl KinematicTree {
    /// Constructs and validates a tree. `fixed_base`: the root is bolted to the
    /// world (e.g. a robot arm on a table) and has no DOFs of its own, rather
    /// than being a free-floating base with 3 translational and 3 rotational
    /// DOFs.
    ///
    /// For a semi-fixed base with some but not all six DOFs, keep the base
    /// fixed and give the root a zero-offset child joint carrying the DOFs the
    /// base actually has, then attach the rest of the body to that joint.
    ///
    /// Returns an error unless exactly one joint has no parent, every parent
    /// index is valid, the parent links form a tree (no cycles), joint names
    /// are unique, and every DOF axis is finite and nonzero.
    pub fn new(joints: Vec<Joint>, fixed_base: bool) -> Result<Self> {
        let n_joints = joints.len();

        let mut name_to_idx = HashMap::with_capacity(n_joints);
        for (i, joint) in joints.iter().enumerate() {
            if name_to_idx.insert(joint.name.clone(), i).is_some() {
                bail!("duplicate joint name '{}'", joint.name);
            }
        }

        let mut root_idx = None;
        let mut children = vec![Vec::new(); n_joints];
        for (i, joint) in joints.iter().enumerate() {
            match joint.parent {
                None if root_idx.is_some() => {
                    bail!("multiple root joints: exactly one joint must have no parent")
                }
                None => root_idx = Some(i),
                Some(p) if p >= n_joints || p == i => {
                    bail!("joint '{}' has invalid parent index {p}", joint.name)
                }
                Some(p) => children[p].push(i),
            }
            for dof in &joint.dofs {
                if !dof.axis.iter().all(|x| x.is_finite()) {
                    bail!(
                        "joint '{}' has a DOF with a zero or non-finite axis",
                        joint.name
                    );
                }
            }
        }
        let Some(root_idx) = root_idx else {
            bail!("no root joint: exactly one joint must have no parent");
        };

        let mut traversal_order = Vec::with_capacity(n_joints);
        let mut queue = VecDeque::from([root_idx]);
        while let Some(j) = queue.pop_front() {
            traversal_order.push(j);
            queue.extend(&children[j]);
        }
        if traversal_order.len() != n_joints {
            bail!("joint parent links contain a cycle");
        }

        let mut joint_dof_start_indices = Vec::with_capacity(n_joints);
        let mut n_dofs = 0;
        for joint in &joints {
            joint_dof_start_indices.push(n_dofs);
            n_dofs += joint.dofs.len();
        }

        // A keypoint is moved by the root DOFs (if any) and by the DOFs of all
        // its strict ancestors, but not by its own joint's DOFs.
        let n_root_dofs = if fixed_base { 0 } else { N_ROOT_DOFS };
        let mut upstream_state_indices: Vec<Vec<usize>> = vec![Vec::new(); n_joints];
        upstream_state_indices[root_idx].extend(0..n_root_dofs);
        for &j in &traversal_order[1..] {
            let p = joints[j].parent.unwrap();
            let mut upstream = upstream_state_indices[p].clone();
            let parent_start = n_root_dofs + joint_dof_start_indices[p];
            upstream.extend(parent_start..parent_start + joints[p].dofs.len());
            upstream.sort();
            upstream_state_indices[j] = upstream;
        }

        let all_dofs = || joints.iter().flat_map(|j| &j.dofs);
        Ok(Self {
            fixed_base,
            root_idx,
            n_dofs,
            joint_dof_start_indices,
            name_to_idx,
            traversal_order,
            upstream_state_indices,
            dof_types: all_dofs().map(|d| d.dof_type).collect(),
            dof_limits: all_dofs().map(|d| d.limits).collect(),
            dof_neutrals: all_dofs().map(|d| d.neutral).collect(),
            dof_weight_multipliers: all_dofs().map(|d| d.weight_multiplier).collect(),
            joint_weight_multipliers: joints.iter().map(|j| j.weight_multiplier).collect(),
            joints,
        })
    }

    pub fn from_json_str(json_str: &str) -> Result<Self> {
        let spec: BodyPlanSpec = serde_json::from_str(json_str)?;
        Self::from_bodyplan_spec(spec)
    }

    pub fn from_json_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// See [`new`](Self::new).
    pub fn fixed_base(&self) -> bool {
        self.fixed_base
    }

    /// Index of the joint without a parent.
    pub fn root_idx(&self) -> usize {
        self.root_idx
    }

    pub fn n_joints(&self) -> usize {
        self.joints.len()
    }

    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Number of free root DOFs: 6 for a free-floating base (3 position, 3
    /// rotation), or 0 for a fixed base.
    pub fn n_root_dofs(&self) -> usize {
        if self.fixed_base { 0 } else { N_ROOT_DOFS }
    }

    /// Length of the solver's state vector: root DOFs first, then every joint
    /// DOF in flat order.
    pub fn state_dim(&self) -> usize {
        self.n_root_dofs() + self.n_dofs
    }

    pub fn joint_names(&self) -> impl Iterator<Item = &str> {
        self.joints.iter().map(|j| j.name.as_str())
    }

    /// Index of the joint called `name`, if any.
    pub fn joint_index(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    /// Index of each joint's first DOF in
    /// [`State::dof_values`](crate::state::State::dof_values).
    pub fn joint_dof_start_indices(&self) -> &[usize] {
        &self.joint_dof_start_indices
    }

    /// State-vector indices that can move joint `joint_idx`'s keypoint: the
    /// root DOFs (if any) plus every DOF on its strict ancestors, ascending.
    #[inline]
    pub(crate) fn upstream_state_indices(&self, joint_idx: usize) -> &[usize] {
        &self.upstream_state_indices[joint_idx]
    }

    /// Length of the longest upstream state index list of any joint, i.e. the
    /// most state variables any single keypoint depends on.
    pub(crate) fn max_n_upstream(&self) -> usize {
        let lengths = self.upstream_state_indices.iter().map(Vec::len);
        lengths.max().unwrap_or(0)
    }

    /// Parent-before-child visiting order of all joints.
    pub(crate) fn traversal_order(&self) -> &[usize] {
        &self.traversal_order
    }

    /// Every DOF's type, in flat DOF order.
    pub(crate) fn dof_types(&self) -> &[DofType] {
        &self.dof_types
    }

    /// Every DOF's limits, in flat DOF order.
    pub(crate) fn dof_limits(&self) -> &[Option<[f32; 2]>] {
        &self.dof_limits
    }

    /// Every DOF's neutral value, in flat DOF order.
    pub(crate) fn dof_neutrals(&self) -> &[f32] {
        &self.dof_neutrals
    }

    /// Every DOF's weight multiplier, in flat DOF order.
    pub(crate) fn dof_weight_multipliers(&self) -> &[f32] {
        &self.dof_weight_multipliers
    }

    /// Every joint's weight multiplier, in joint order.
    pub(crate) fn joint_weight_multipliers(&self) -> &[f32] {
        &self.joint_weight_multipliers
    }

    fn from_bodyplan_spec(spec: BodyPlanSpec) -> Result<Self> {
        let name_to_idx: HashMap<&str, usize> = spec
            .joints
            .iter()
            .enumerate()
            .map(|(i, j)| (j.name.as_str(), i))
            .collect();
        let mut parents = Vec::with_capacity(spec.joints.len());
        for joint in &spec.joints {
            parents.push(match &joint.parent {
                None => None,
                Some(name) => match name_to_idx.get(name.as_str()) {
                    Some(&idx) => Some(idx),
                    None => bail!("parent joint '{name}' of joint '{}' not found", joint.name),
                },
            });
        }
        let joints = spec
            .joints
            .into_iter()
            .zip(parents)
            .map(|(joint, parent)| Joint {
                name: joint.name,
                parent,
                offset_pos: Vector3::from(joint.offset_pos),
                offset_quat: quat_from_wxyz(joint.offset_quat),
                dofs: joint
                    .dofs
                    .into_iter()
                    .map(|d| {
                        Dof::new(
                            d.dof_type,
                            Vector3::from(d.axis),
                            d.neutral,
                            d.limits,
                            d.weight_multiplier,
                        )
                    })
                    .collect(),
                weight_multiplier: joint.weight_multiplier,
            })
            .collect();
        Self::new(joints, spec.fixed_base)
    }
}

// =============================================================================
//  Interfaces for parsing kinematic trees from JSON
// =============================================================================

/// Root-level data structure for body plans serialized in JSON (see
/// `etc/bodyplan_20260923.schema.json`). Unrecognized fields (e.g. an "x-name"
/// key) are ignored, for JSON self-documentation only.
#[derive(Deserialize)]
struct BodyPlanSpec {
    #[serde(default)]
    fixed_base: bool,
    joints: Vec<JointSpec>,
}

#[derive(Deserialize)]
struct JointSpec {
    name: String,
    parent: Option<String>,
    offset_pos: [f32; 3],
    offset_quat: [f32; 4],
    #[serde(default = "default_weight_multiplier")]
    weight_multiplier: f32,
    dofs: Vec<DofSpec>,
}

#[derive(Deserialize)]
struct DofSpec {
    #[serde(rename = "type")]
    dof_type: DofType,
    axis: [f32; 3],
    neutral: f32,
    limits: Option<[f32; 2]>,
    #[serde(default = "default_weight_multiplier")]
    weight_multiplier: f32,
}

fn default_weight_multiplier() -> f32 {
    1.0
}
