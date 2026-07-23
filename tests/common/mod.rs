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
        residual_weight: 1.0,
    };
    let joint1 = Joint {
        name: "joint1".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![Dof {
            axis: Vector3::z(),
            dof_type: DofType::Hinge,
            neutral_angle: 0.0,
            limits: None,
            neutral_weight: 1.0,
        }],
        parent: Some(0),
        children: Vec::new(),
        dof_offset: 0,
        residual_weight: 1.0,
    };
    let joint2 = Joint {
        name: "joint2".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![Dof {
            axis: Vector3::z(),
            dof_type: DofType::Hinge,
            neutral_angle: 0.0,
            limits: Some([-0.5, 0.5]),
            neutral_weight: 1.0,
        }],
        parent: Some(1),
        children: Vec::new(),
        dof_offset: 1,
        residual_weight: 1.0,
    };
    let tip = Joint {
        name: "tip".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: Some(2),
        children: Vec::new(),
        dof_offset: 2,
        residual_weight: 1.0,
    };
    Arc::new(KinematicTree::new(vec![root, joint1, joint2, tip], 0))
}
