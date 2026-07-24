//! This module defines data structures describing the kinematic tree.

use std::collections::HashMap;

use nalgebra::{UnitQuaternion, Vector3};
use serde::Deserialize;

use crate::utils::quat_from_wxyz;

// =============================================================================
//  Data structures for the actual algorithm
// =============================================================================

pub const N_ROOT_DOFS: usize = 6; // 3 for root position, 3 for root rotation

/// Whether a [`Dof`] is a hinge (rotational) or slide (translational) DOF.
///
/// Only [`Hinge`](DofType::Hinge) is currently implemented -- constructing a
/// body plan with a `Slide` DOF panics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DofType {
    /// Rotates about `axis`.
    Hinge,
    /// Translates along `axis`. Not yet implemented.
    Slide,
}

/// A single degree of freedom (DOF) on the kinematic tree.
#[derive(Clone, Copy, Debug)]
pub struct Dof {
    /// Rotational or translational axis in local frame (relative to the
    /// joint's `offset_quat`). Must be unit length -- forward kinematics
    /// trusts this invariant (rather than re-normalizing on every solve
    /// iteration) instead of checking it, so a `Dof` built directly (rather
    /// than parsed from JSON, which normalizes it) must provide one itself.
    pub axis: Vector3<f32>,
    /// Whether this is a hinge or slide DOF.
    pub dof_type: DofType,
    /// Neutral value of the DOF: an angle in radians for a hinge DOF, or a
    /// position for a slide DOF.
    pub neutral: f32,
    /// Optional angle limits in [min, max]. Unbounded if `None`.
    pub limits: Option<[f32; 2]>,
    /// Scales this DOF's contribution to the deviation-from-neutral penalty,
    /// multiplied together with [`SolverConfig::weight`].
    ///
    /// [`SolverConfig::weight`]: crate::solver::SolverConfig::weight
    pub weight_scaler: f32,
}

/// An anatomical joint with up to three rotational DOF.
/// Also serves as a tracking keypoint in the MoCap data.
#[derive(Clone, Debug)]
pub struct Joint {
    /// Joint name, e.g. "lf_thorax_coxa"
    pub name: String,
    /// Offset of the joint from its parent in the kinematic tree
    pub offset_pos: Vector3<f32>,
    /// Offset rotation of the joint from its parent in the kinematic tree
    pub offset_quat: UnitQuaternion<f32>,
    /// Rotational DOFs at this joint. Can be empty or up to 3. The ordering is
    /// important (as SO(3) rotations are not commutative) and should be
    /// consistent with MoCap data.
    pub dofs: Vec<Dof>,
    /// Index of the parent joint in the body plan. Should be `None` for the
    /// root joint, which is attached to an imaginary floating base
    /// (i.e. connected to the world with a free joint).
    pub parent: Option<usize>,
    /// Indices of this joint's direct children. Populated by
    // `KinematicTree::new`. Redundant with `parent` but useful as a cache.
    pub children: Vec<usize>,
    /// Index of this joint's 0th DOF in the flattened DOF vector of the
    /// kinematic tree.
    pub dof_offset: usize,
    /// Scales this joint's keypoint residual, multiplied together with each
    /// frame's [`KeypointObservation`] weight for this keypoint.
    ///
    /// [`KeypointObservation`]: crate::observation::KeypointObservation
    pub weight_scaler: f32,
}

/// A kinematic tree, i.e. body plan, or skeleton.
#[derive(Clone, Debug)]
pub struct KinematicTree {
    /// Joints in the kinematic tree. The order should be consistent with
    /// `parent` indices in the `Joint` instances.
    pub joints: Vec<Joint>,
    /// Index of the root joint in `joints`. Should usually be 0.
    pub root_idx: usize,
}

impl KinematicTree {
    /// Construct a tree directly from parsed `joints` and a `root_idx`,
    /// populating each joint's `children` from `parent`.
    pub fn new(mut joints: Vec<Joint>, root_idx: usize) -> Self {
        for i in 0..joints.len() {
            if let Some(parent_idx) = joints[i].parent {
                joints[parent_idx].children.push(i);
            }
        }
        Self { joints, root_idx }
    }

    pub fn n_joints(&self) -> usize {
        self.joints.len()
    }

    pub fn n_dofs(&self) -> usize {
        self.joints.iter().map(|joint| joint.dofs.len()).sum()
    }

    pub fn state_dim(&self) -> usize {
        // 3 for root position, 3 for root rotation, plus DOFs on the body
        N_ROOT_DOFS + self.n_dofs()
    }

    /// Return indices of the direct children of the joint at the given index
    pub fn children_indices(&self, joint_idx: usize) -> &[usize] {
        &self.joints[joint_idx].children
    }

    fn from_bodyplan_spec(body: BodyPlanSpec) -> Self {
        let (parent_idxs, root_idx) = resolve_parents(&body);
        let mut joints = Vec::with_capacity(body.joints.len());
        let mut curr_dof_offset = 0;
        for (joint_spec, parent_idx) in body.joints.into_iter().zip(parent_idxs) {
            let n_dofs = joint_spec.dofs.len();
            let dofs = joint_spec
                .dofs
                .into_iter()
                .map(|dof| {
                    if dof.dof_type == DofType::Slide {
                        unimplemented!(
                            "Slide DOFs are not yet implemented (joint '{}')",
                            joint_spec.name
                        );
                    }
                    Dof {
                        // Normalized once here rather than on every solve
                        // iteration -- see `Dof::axis`'s doc comment.
                        axis: Vector3::from(dof.axis).normalize(),
                        dof_type: dof.dof_type,
                        neutral: dof.neutral,
                        limits: dof.limits,
                        weight_scaler: dof.weight_scaler,
                    }
                })
                .collect();
            let joint = Joint {
                name: joint_spec.name,
                offset_pos: Vector3::from(joint_spec.offset_pos),
                offset_quat: quat_from_wxyz(joint_spec.offset_quat),
                dofs,
                parent: parent_idx,
                children: Vec::new(),
                dof_offset: curr_dof_offset,
                weight_scaler: joint_spec.weight_scaler,
            };
            joints.push(joint);
            curr_dof_offset += n_dofs;
        }
        Self::new(joints, root_idx)
    }

    pub fn from_json_str(json_str: &str) -> Self {
        let body: BodyPlanSpec = serde_json::from_str(json_str)
            .unwrap_or_else(|e| panic!("Failed to parse body plan JSON: {}", e));
        Self::from_bodyplan_spec(body)
    }

    pub fn from_json_file(path: impl AsRef<std::path::Path>) -> Self {
        let path = path.as_ref();
        let json_str = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!(
                "Failed to read body plan JSON file '{}': {}",
                path.display(),
                e
            )
        });
        Self::from_json_str(&json_str)
    }
}

// =============================================================================
//  Interfaces for parsing kinematic trees from JSON
// =============================================================================

/// Root-level data structure for body plans serialized in JSON.
/// The "metadata" field in the JSON is ignored (it's for JSON self-documentation only).
#[derive(Deserialize)]
struct BodyPlanSpec {
    joints: Vec<JointSpec>,
}

#[derive(Deserialize)]
struct JointSpec {
    name: String,
    parent: Option<String>,
    offset_pos: [f32; 3],
    offset_quat: [f32; 4],
    #[serde(default = "default_weight_scaler")]
    weight_scaler: f32,
    dofs: Vec<DofSpec>,
}

#[derive(Deserialize)]
struct DofSpec {
    axis: [f32; 3],
    #[serde(rename = "type")]
    dof_type: DofType,
    neutral: f32,
    limits: Option<[f32; 2]>,
    #[serde(default = "default_weight_scaler")]
    weight_scaler: f32,
}

fn default_weight_scaler() -> f32 {
    1.0
}

/// Resolve the parent indices for each joint in the order of `body.joints`
/// using joint names specified in the JSON, and identify the index of the root
/// joint. Return a tuple of (parent_indices, root_index).
fn resolve_parents(body: &BodyPlanSpec) -> (Vec<Option<usize>>, usize) {
    // Record name-to-index mapping
    let mut name_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, joint) in body.joints.iter().enumerate() {
        if name_to_idx.contains_key(joint.name.as_str()) {
            panic!("Duplicate joint name found: {}", joint.name);
        }
        name_to_idx.insert(joint.name.as_str(), i);
    }

    // Resolve parents
    let mut root_idx: Option<usize> = None;
    let mut parent_idxs: Vec<Option<usize>> = Vec::with_capacity(body.joints.len());
    for (i, joint) in body.joints.iter().enumerate() {
        let parent_idx = match &joint.parent {
            None => {
                if root_idx.is_some() {
                    panic!("Multiple root joints found. Exactly one joint must have parent=null");
                }
                root_idx = Some(i);
                None
            }
            Some(parent_name) => {
                if let Some(&parent_idx) = name_to_idx.get(parent_name.as_str()) {
                    Some(parent_idx)
                } else {
                    panic!(
                        "Parent joint '{}' not found for joint '{}'",
                        parent_name, joint.name
                    );
                }
            }
        };
        parent_idxs.push(parent_idx);
    }

    let root_idx = root_idx
        .unwrap_or_else(|| panic!("No root joint found. Exactly one joint must have parent=null"));
    (parent_idxs, root_idx)
}
