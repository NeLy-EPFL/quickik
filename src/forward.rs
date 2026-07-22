//! This module implements forward kinematics, including the tracking of the
//! Jacobian of the keypoint positions with respect to the state variables.

use nalgebra::{DMatrix, Unit, UnitQuaternion, Vector3};

use crate::body_plan::{Joint, KinematicTree, N_ROOT_DOFS};
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
    /// DOF's rotation axis in world coordinates
    axis_world: Vector3<f32>,
    /// Origin of the joint that this DOF belongs to, in world coordinates
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
    /// Records of all DOFs in the kinematic tree in order of their flat indices
    dof_records: Vec<DofRecord>,
}

impl ForwardKinematicsWorkspace {
    pub fn new(kinematic_tree: &KinematicTree) -> Self {
        let n_joints = kinematic_tree.n_joints();
        Self {
            n_joints,
            kpt_positions: vec![Vector3::zeros(); n_joints],
            kpt_jacobian: DMatrix::zeros(3 * n_joints, kinematic_tree.state_dim()),
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
    let frame = evaluate_frame_at_joint(joint, parent_frame, state, &mut workspace.dof_records);

    workspace.kpt_positions[curr_joint_idx] = frame.origin;
    write_keypoint_jacobian(
        &mut workspace.kpt_jacobian,
        state,
        curr_joint_idx,
        frame.origin,
        &workspace.dof_records[..n_records_before],
    );

    for &child_idx in state.kinematic_tree.children_indices(curr_joint_idx) {
        traverse_dfs(workspace, state, child_idx, frame);
    }
    workspace.dof_records.truncate(n_records_before);
}

/// Compute frame of a single joint in world coordinates
fn evaluate_frame_at_joint(
    joint: &Joint,
    parent_frame: Frame,
    state: &State,
    dof_records: &mut Vec<DofRecord>,
) -> Frame {
    // Start with parent frame...
    let origin = parent_frame.origin + parent_frame.rotation * joint.offset_pos;
    let mut rotation = parent_frame.rotation * joint.offset_quat;

    // ... then apply the joint's own DOFs
    for (i, dof) in joint.dofs.iter().enumerate() {
        let axis_local = dof.axis;
        let axis_world = rotation * axis_local;
        let record = DofRecord {
            state_idx: N_ROOT_DOFS + joint.dof_offset + i,
            axis_world,
            origin_world: origin,
        };
        dof_records.push(record);

        let angle = state.dof_angles[joint.dof_offset + i];
        rotation *= UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis_local), angle);
    }

    Frame { origin, rotation }
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

    // Upstream joint dofs:
    // Note that most DOFs do not affect any given keypoint
    for record in dof_records_until_now {
        let radius = pos - record.origin_world;
        let d = record.axis_world.cross(&radius);
        jacobian[(row0, record.state_idx)] = d.x;
        jacobian[(row1, record.state_idx)] = d.y;
        jacobian[(row2, record.state_idx)] = d.z;
    }
}
