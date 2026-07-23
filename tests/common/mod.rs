//! Shared test fixtures: a minimal 3-joint kinematic chain.
//!
//! root (free joint, no local DOFs)
//!   -> joint1: offset (1,0,0), 1 DOF about local Z, unbounded
//!     -> joint2: offset (1,0,0), 1 DOF about local Z, limited to [-0.5, 0.5]
//!       -> tip: offset (1,0,0), no DOFs
//!
//! A joint's own DOF never moves its own keypoint (only its descendants'), so
//! every DOF needs a keypoint downstream of it to be observable/recoverable --
//! hence the trailing fixed `tip` joint past `joint2`.

use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};
use quickik::body_plan::{Dof, DofType, Joint, KinematicTree};

pub fn two_joint_chain() -> Arc<KinematicTree> {
    let root = Joint {
        name: "root".to_string(),
        offset_pos: Vector3::zeros(),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: None,
        children: Vec::new(),
        dof_offset: 0,
        weight_scaler: 1.0,
    };
    let joint1 = Joint {
        name: "joint1".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![Dof {
            axis: Vector3::z(),
            dof_type: DofType::Hinge,
            neutral: 0.0,
            limits: None,
            weight_scaler: 1.0,
        }],
        parent: Some(0),
        children: Vec::new(),
        dof_offset: 0,
        weight_scaler: 1.0,
    };
    let joint2 = Joint {
        name: "joint2".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![Dof {
            axis: Vector3::z(),
            dof_type: DofType::Hinge,
            neutral: 0.0,
            limits: Some([-0.5, 0.5]),
            weight_scaler: 1.0,
        }],
        parent: Some(1),
        children: Vec::new(),
        dof_offset: 1,
        weight_scaler: 1.0,
    };
    let tip = Joint {
        name: "tip".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: Some(2),
        children: Vec::new(),
        dof_offset: 2,
        weight_scaler: 1.0,
    };
    Arc::new(KinematicTree::new(vec![root, joint1, joint2, tip], 0))
}

/// A root with two independent single-DOF branches (each a joint + a fixed
/// tip keypoint downstream of it), sharing no keypoints between them -- so
/// each DOF's contribution to the Gauss-Newton normal equations is entirely
/// decoupled from the other, unlike `two_joint_chain`'s single serial chain.
///
/// root (free joint, no local DOFs)
///   -> branch_a_joint: 1 DOF about local Z, unbounded
///     -> branch_a_tip
///   -> branch_b_joint: 1 DOF about local Z, unbounded
///     -> branch_b_tip
///
/// Joint order (and therefore keypoint/observation order) is: root (0),
/// branch_a_joint (1), branch_a_tip (2), branch_b_joint (3), branch_b_tip
/// (4). `branch_a_joint`'s DOF is flattened index 0, `branch_b_joint`'s is 1.
#[allow(dead_code)] // only used by tests/solver_test.rs, not every binary sharing this module
pub fn two_independent_single_dof_branches() -> Arc<KinematicTree> {
    let root = Joint {
        name: "root".to_string(),
        offset_pos: Vector3::zeros(),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: None,
        children: Vec::new(),
        dof_offset: 0,
        weight_scaler: 1.0,
    };
    let branch_a_joint = Joint {
        name: "branch_a_joint".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![Dof {
            axis: Vector3::z(),
            dof_type: DofType::Hinge,
            neutral: 0.0,
            limits: None,
            weight_scaler: 1.0,
        }],
        parent: Some(0),
        children: Vec::new(),
        dof_offset: 0,
        weight_scaler: 1.0,
    };
    let branch_a_tip = Joint {
        name: "branch_a_tip".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: Some(1),
        children: Vec::new(),
        dof_offset: 1,
        weight_scaler: 1.0,
    };
    let branch_b_joint = Joint {
        name: "branch_b_joint".to_string(),
        offset_pos: Vector3::new(-1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![Dof {
            axis: Vector3::z(),
            dof_type: DofType::Hinge,
            neutral: 0.0,
            limits: None,
            weight_scaler: 1.0,
        }],
        parent: Some(0),
        children: Vec::new(),
        dof_offset: 1,
        weight_scaler: 1.0,
    };
    let branch_b_tip = Joint {
        name: "branch_b_tip".to_string(),
        offset_pos: Vector3::new(-1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: Some(3),
        children: Vec::new(),
        dof_offset: 2,
        weight_scaler: 1.0,
    };
    Arc::new(KinematicTree::new(
        vec![
            root,
            branch_a_joint,
            branch_a_tip,
            branch_b_joint,
            branch_b_tip,
        ],
        0,
    ))
}
