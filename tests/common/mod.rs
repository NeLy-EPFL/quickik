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

/// A chain with a single slide DOF.
///
/// root (free joint, no local DOFs)
///   -> joint1: offset (1,0,0), 1 slide DOF along local X, unbounded
///     -> tip: offset (1,0,0), no DOFs
///
/// Like a hinge DOF, joint1's own slide never moves joint1's own keypoint --
/// only `tip`'s, which is why `tip` is here at all (see this module's doc
/// comment).
#[allow(dead_code)] // only used by tests/forward_test.rs, not every binary sharing this module
pub fn slide_joint_chain() -> Arc<KinematicTree> {
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
            axis: Vector3::x(),
            dof_type: DofType::Slide,
            neutral: 0.0,
            limits: None,
            weight_scaler: 1.0,
        }],
        parent: Some(0),
        children: Vec::new(),
        dof_offset: 0,
        weight_scaler: 1.0,
    };
    let tip = Joint {
        name: "tip".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: Some(1),
        children: Vec::new(),
        dof_offset: 1,
        weight_scaler: 1.0,
    };
    Arc::new(KinematicTree::new(vec![root, joint1, tip], 0))
}

/// A chain combining a hinge and a slide DOF on separate joints, so the
/// slide's world-frame axis is rotated by the upstream hinge -- this is what
/// exercises the cross-term where perturbing the hinge angle also perturbs a
/// downstream slide's direction (and therefore the position it produces).
///
/// root (free joint, no local DOFs)
///   -> hinge_joint: offset (1,0,0), 1 hinge DOF about local Z, unbounded
///     -> slide_joint: offset (1,0,0), 1 slide DOF along local X, unbounded
///       -> tip: offset (1,0,0), no DOFs
#[allow(dead_code)] // only used by tests/forward_test.rs, not every binary sharing this module
pub fn hinge_then_slide_chain() -> Arc<KinematicTree> {
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
    let hinge_joint = Joint {
        name: "hinge_joint".to_string(),
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
    let slide_joint = Joint {
        name: "slide_joint".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![Dof {
            axis: Vector3::x(),
            dof_type: DofType::Slide,
            neutral: 0.0,
            limits: None,
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
    Arc::new(KinematicTree::new(
        vec![root, hinge_joint, slide_joint, tip],
        0,
    ))
}

/// A single joint carrying both a hinge DOF (applied first) and a slide DOF,
/// so the slide's own translation is expressed along an axis that the same
/// joint's own hinge has already rotated -- the tightest version of the
/// hinge/slide cross-term, entirely within one joint's own DOF list.
///
/// root (free joint, no local DOFs)
///   -> joint1: offset (1,0,0), dofs = [hinge about local Z, slide along
///     local X], unbounded
///     -> tip: offset (1,0,0), no DOFs
#[allow(dead_code)] // only used by tests/forward_test.rs, not every binary sharing this module
pub fn joint_with_hinge_and_slide() -> Arc<KinematicTree> {
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
        dofs: vec![
            Dof {
                axis: Vector3::z(),
                dof_type: DofType::Hinge,
                neutral: 0.0,
                limits: None,
                weight_scaler: 1.0,
            },
            Dof {
                axis: Vector3::x(),
                dof_type: DofType::Slide,
                neutral: 0.0,
                limits: None,
                weight_scaler: 1.0,
            },
        ],
        parent: Some(0),
        children: Vec::new(),
        dof_offset: 0,
        weight_scaler: 1.0,
    };
    let tip = Joint {
        name: "tip".to_string(),
        offset_pos: Vector3::new(1.0, 0.0, 0.0),
        offset_quat: UnitQuaternion::identity(),
        dofs: vec![],
        parent: Some(1),
        children: Vec::new(),
        dof_offset: 2,
        weight_scaler: 1.0,
    };
    Arc::new(KinematicTree::new(vec![root, joint1, tip], 0))
}
