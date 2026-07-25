//! This module implements forward kinematics, including the tracking of the
//! Jacobian of the keypoint positions with respect to the state variables.

use nalgebra::{DMatrix, Unit, UnitQuaternion, Vector3};

use crate::body_plan::{DofType, Joint, KinematicTree, N_ROOT_DOFS};
use crate::state::State;

#[derive(Clone, Copy, Debug)]
struct Frame {
    origin: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
}

/// A record of a single DOF's current configuration.
struct DofRecord {
    /// DOF's flat index in the state vector, starting from 6
    state_idx: usize,
    /// Hinge or slide.
    dof_type: DofType,
    /// DOF's rotation/translation axis in world coordinates
    axis_world: Vector3<f32>,
    /// Origin of the joint that this DOF belongs to, in world coordinates, as
    /// of just before this DOF's own contribution is applied. Only used for
    /// `Hinge` DOFs; irrelevant for slide DOFs, whose Jacobian column is
    /// `axis_world` regardless of position.
    origin_world: Vector3<f32>,
}

/// Reusable workspace for forward kinematics computations, including memory
/// buffers that should be allocated once and reused across multiple (typically
/// all) calls to [`evaluate_fwdkin`].
pub struct ForwardKinematicsWorkspace {
    /// Number of joints (and therefore keypoints) in the kinematic tree
    n_joints: usize,
    /// Keypoint positions in world coordinates
    pub kpt_positions: Vec<Vector3<f32>>,
    /// Stacked per-keypoint Jacobians:
    /// keypoint k's `3 x state_dim` block is rows `3*k .. 3*k+3`
    pub kpt_jacobian: DMatrix<f32>,
    /// Records of which DOFs can affect each keypoint's position. This enables
    /// sparse Jacobian computation in the solver, since most DOFs do not impact
    /// any given keypoint.
    pub relevant_dof_idxs_by_joint: Vec<Vec<usize>>,
    /// Records of all DOFs in the kinematic tree in order of their flat indices
    dof_records: Vec<DofRecord>,
}

impl ForwardKinematicsWorkspace {
    pub fn new(kinematic_tree: &KinematicTree) -> Self {
        let n_joints = kinematic_tree.n_joints();
        let state_dim = kinematic_tree.state_dim();
        Self {
            n_joints,
            kpt_positions: vec![Vector3::zeros(); n_joints],
            kpt_jacobian: DMatrix::zeros(3 * n_joints, state_dim),
            relevant_dof_idxs_by_joint: std::iter::repeat_with(|| Vec::with_capacity(state_dim))
                .take(n_joints)
                .collect(),
            dof_records: Vec::with_capacity(kinematic_tree.n_dofs()),
        }
    }
}
pub fn evaluate_fwdkin(workspace: &mut ForwardKinematicsWorkspace, state: &State) {
    debug_assert_eq!(workspace.n_joints, state.kinematic_tree.n_joints());

    // Matrix::fill is a generic, unspecialized per-element loop and it's ~60x
    // slower than filling the underlying contiguous storage directly using
    // matrix.as_mut_slice().fill.
    workspace.kpt_jacobian.as_mut_slice().fill(0.0);
    workspace.dof_records.clear();

    let root_frame = Frame {
        origin: state.root_pos,
        rotation: state.root_rot,
    };
    traverse_dfs(workspace, state, state.kinematic_tree.root_idx, root_frame);
}

/// Recursively traverses the kinematic tree in depth-first order, unrolling
/// forward kinematics and recording the Jacobian of each keypoint with respect
/// to the DOF states.
fn traverse_dfs(
    workspace: &mut ForwardKinematicsWorkspace,
    state: &State,
    curr_joint_idx: usize,
    parent_frame: Frame,
) {
    let joint = &state.kinematic_tree.joints[curr_joint_idx];

    let n_records_before = workspace.dof_records.len();
    let (joint_origin, frame) =
        evaluate_frame_at_joint(joint, parent_frame, state, &mut workspace.dof_records);

    // A joint's own DOFs (hinge or slide) never move its own keypoint, only its
    // descendants' (see this module's doc comment on `joint_origin` in
    // `evaluate_frame_at_joint`). So the keypoint and its Jacobian use
    // `joint_origin` (computed before this joint's own DOFs) rather than
    // `frame.origin` (which children use, and which does reflect them).
    workspace.kpt_positions[curr_joint_idx] = joint_origin;
    write_keypoint_jacobian(
        &mut workspace.kpt_jacobian,
        state,
        curr_joint_idx,
        joint_origin,
        &workspace.dof_records[..n_records_before],
    );

    // Record which DOFs can affect this keypoint. Root pos/rot affect all.
    workspace.relevant_dof_idxs_by_joint[curr_joint_idx].clear();
    workspace.relevant_dof_idxs_by_joint[curr_joint_idx].extend(0..N_ROOT_DOFS);
    for i in 0..n_records_before {
        let state_idx = workspace.dof_records[i].state_idx;
        workspace.relevant_dof_idxs_by_joint[curr_joint_idx].push(state_idx);
    }

    for &child_idx in state.kinematic_tree.children_indices(curr_joint_idx) {
        traverse_dfs(workspace, state, child_idx, frame);
    }
    workspace.dof_records.truncate(n_records_before);
}

/// Compute the frame of a single joint in world coordinates.
///
/// Returns `(joint_origin, frame)`: `joint_origin` is this joint's own
/// keypoint position, fixed by the parent frame and this joint's constant
/// offset alone (not moved by this joint's own DOFs, only by its ancestors).
/// `frame` additionally contains this joint's own DOFs (both its rotation and
/// origin). The origin is shifted by any slide DOFs and is what its children
/// are positioned relative to.
fn evaluate_frame_at_joint(
    joint: &Joint,
    parent_frame: Frame,
    state: &State,
    dof_records: &mut Vec<DofRecord>,
) -> (Vector3<f32>, Frame) {
    // Start with parent frame...
    let own_origin = parent_frame.origin + parent_frame.rotation * joint.offset_pos;
    let mut rotation = parent_frame.rotation * joint.offset_quat;
    let mut origin_for_children = own_origin;

    // ... then apply the joint's own DOFs
    for (i, dof) in joint.dofs.iter().enumerate() {
        let axis_local = dof.axis;
        let axis_world = rotation * axis_local;
        let record = DofRecord {
            state_idx: N_ROOT_DOFS + joint.dof_offset + i,
            dof_type: dof.dof_type,
            axis_world,
            origin_world: origin_for_children,
        };
        dof_records.push(record);

        let value = state.dof_angles[joint.dof_offset + i]; // angle or slide pos
        match dof.dof_type {
            DofType::Hinge => {
                // `Dof::axis` is already unit. Skip re-normalization here in
                // the hot loop.
                let unit_axis_local = Unit::new_unchecked(axis_local);
                rotation *= UnitQuaternion::from_axis_angle(&unit_axis_local, value);
            }
            DofType::Slide => origin_for_children += axis_world * value,
        }
    }

    (
        own_origin,
        Frame {
            origin: origin_for_children,
            rotation,
        },
    )
}

/// Write the Jacobian of a single keypoint with respect to the state variables
fn write_keypoint_jacobian(
    jacobian: &mut DMatrix<f32>,
    state: &State,
    node_idx: usize,
    pos: Vector3<f32>,
    dof_records_until_now: &[DofRecord],
) {
    let row0 = 3 * node_idx;
    let row1: usize = row0 + 1;
    let row2: usize = row0 + 2;

    // Root translation (state cols 0..3):
    // Moving the root moves every keypoint by the same amount
    jacobian[(row0, 0)] = 1.0;
    jacobian[(row1, 1)] = 1.0;
    jacobian[(row2, 2)] = 1.0;

    // Root rotation (state cols 3..6):
    // Rotate about the root's current position
    let radius = pos - state.root_pos;
    for (i, axis) in [Vector3::x(), Vector3::y(), Vector3::z()]
        .iter()
        .enumerate()
    {
        let d = axis.cross(&radius);
        jacobian[(row0, 3 + i)] = d.x;
        jacobian[(row1, 3 + i)] = d.y;
        jacobian[(row2, 3 + i)] = d.z;
    }

    // Upstream joint dofs. Note that each keypoint is only affected by a few
    // DOFs, so this is rather sparse
    for record in dof_records_until_now {
        let jac = match record.dof_type {
            // Rotating about `axis_world` through `origin_world` moves a
            // point in its orbit: standard angular-velocity cross product.
            DofType::Hinge => record.axis_world.cross(&(pos - record.origin_world)),
            // Sliding along `axis_world` moves every downstream point by the
            // same amount along that direction, regardless of position.
            DofType::Slide => record.axis_world,
        };
        jacobian[(row0, record.state_idx)] = jac.x;
        jacobian[(row1, record.state_idx)] = jac.y;
        jacobian[(row2, record.state_idx)] = jac.z;
    }
}
