//! This module implements forward kinematics, including the tracking of the
//! Jacobian of the keypoint positions with respect to the state variables.

use nalgebra::{DMatrix, Unit, UnitQuaternion, Vector3};

use crate::body_plan::{DofType, Joint, KinematicTree};
use crate::state::State;

#[derive(Clone, Copy, Debug)]
struct Frame {
    origin: Vector3<f32>,
    rotation: UnitQuaternion<f32>,
}

/// A representation of a single DOF's current configuration, including its
/// angle (hinge) or position (slide) and its placement in world coordinates.
struct DofFrame {
    /// DOF's flat index in the state vector, starting from the tree's
    /// [`n_root_dofs`](crate::body_plan::KinematicTree::n_root_dofs)
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
/// all) calls to [`forward_kinematics`].
pub struct ForwardKinematicsWorkspace {
    /// Number of joints (and therefore keypoints) in the kinematic tree
    n_joints: usize,
    /// Keypoint positions in world coordinates
    pub kpt_positions: Vec<Vector3<f32>>,
    /// Stacked per-keypoint Jacobians of shape (`3*n_joints x state_dim`).
    /// Keypoint k's `3 x state_dim` block is rows `3*k .. 3*k+3`.
    pub kpt_jacobian: DMatrix<f32>,
    /// Records of which DOFs can affect each keypoint's position. This enables
    /// sparse Jacobian computation in the solver, since most DOFs do not impact
    /// any given keypoint.
    pub upstream_dof_idxs_by_joint: Vec<Vec<usize>>,
    /// All current body DOFs frames in order of their flat indices
    dof_frames: Vec<DofFrame>,
}

impl ForwardKinematicsWorkspace {
    pub fn new(kinematic_tree: &KinematicTree) -> Self {
        let n_joints = kinematic_tree.n_joints();
        let state_dim = kinematic_tree.state_dim();
        Self {
            n_joints,
            kpt_positions: vec![Vector3::zeros(); n_joints],
            kpt_jacobian: DMatrix::zeros(3 * n_joints, state_dim),
            upstream_dof_idxs_by_joint: std::iter::repeat_with(|| Vec::with_capacity(state_dim))
                .take(n_joints)
                .collect(),
            dof_frames: Vec::with_capacity(kinematic_tree.n_dofs()),
        }
    }
}
pub fn forward_kinematics(workspace: &mut ForwardKinematicsWorkspace, state: &State) {
    debug_assert_eq!(workspace.n_joints, state.kinematic_tree.n_joints());

    // Use `.as_mut_slice().fill(0.0);` instead of of `.fill(0.0)`: the latter
    // is generic while the former is specialized for contiguous storage of
    // specific type and is ~60x faster.
    workspace
        .kpt_positions
        .as_mut_slice()
        .fill(Vector3::zeros());
    workspace.kpt_jacobian.as_mut_slice().fill(0.0);
    workspace.dof_frames.clear();

    let root_frame = Frame {
        origin: state.root_pos,
        rotation: state.root_quat,
    };
    traverse_dfs(workspace, state, state.kinematic_tree.root_idx, root_frame);
}

/// Recursively traverse the kinematic tree in depth-first order, unrolling
/// forward kinematics and recording the Jacobian of each keypoint with respect
/// to the DOF states.
fn traverse_dfs(
    workspace: &mut ForwardKinematicsWorkspace,
    state: &State,
    curr_joint_idx: usize,
    parent_frame: Frame,
) {
    let joint = &state.kinematic_tree.joints[curr_joint_idx];

    let n_records_before = workspace.dof_frames.len();
    let (joint_origin, frame) =
        evaluate_frame_at_joint(joint, parent_frame, state, &mut workspace.dof_frames);

    // A joint's own DOFs (hinge or slide) never move its own keypoint, only its
    // descendants' (see this module's doc comment on `joint_origin` in
    // `evaluate_frame_at_joint`). So the keypoint and its Jacobian use
    // `joint_origin` (computed before this joint's own DOFs) rather than
    // `frame.origin` (which children use, and which does reflect them).
    workspace.kpt_positions[curr_joint_idx] = joint_origin;
    update_jacobian_for_curr_keypoint(
        &mut workspace.kpt_jacobian,
        state,
        curr_joint_idx,
        joint_origin,
        &workspace.dof_frames[..n_records_before],
    );

    // Record which DOFs can affect this keypoint.
    workspace.upstream_dof_idxs_by_joint[curr_joint_idx].clear();
    // If the root is floating, its six DOFs affect all keypoints.
    workspace.upstream_dof_idxs_by_joint[curr_joint_idx]
        .extend(0..state.kinematic_tree.n_root_dofs());
    for i in 0..n_records_before {
        let state_idx = workspace.dof_frames[i].state_idx;
        workspace.upstream_dof_idxs_by_joint[curr_joint_idx].push(state_idx);
    }

    // Recurse to children joints
    for &child_idx in state.kinematic_tree.children_indices(curr_joint_idx) {
        traverse_dfs(workspace, state, child_idx, frame);
    }
    // Revert to the DOF frames until the current point (i.e. discard downstream
    // frames that are only for the children branches).
    workspace.dof_frames.truncate(n_records_before);
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
    dof_records: &mut Vec<DofFrame>,
) -> (Vector3<f32>, Frame) {
    // Start with parent frame...
    let own_origin = parent_frame.origin + parent_frame.rotation * joint.offset_pos;
    let mut rotation = parent_frame.rotation * joint.offset_quat;
    let mut origin_for_children = own_origin;
    let n_root_dofs = state.kinematic_tree.n_root_dofs();

    // Then apply the joint's own DOFs
    for (i, dof) in joint.dofs.iter().enumerate() {
        let axis_local = dof.axis();
        let axis_world = rotation * axis_local;
        let record = DofFrame {
            state_idx: n_root_dofs + joint.dof_startidx + i,
            dof_type: dof.dof_type,
            axis_world,
            origin_world: origin_for_children,
        };
        dof_records.push(record);

        let value = state.dof_values[joint.dof_startidx + i]; // angle or slide pos
        match dof.dof_type {
            DofType::Hinge => {
                // `Dof::axis` is already unit, skip normalization here
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

/// Add entries of the Jacobian for the current keypoint to the Jacobian buffer.
fn update_jacobian_for_curr_keypoint(
    full_jacobian_buf: &mut DMatrix<f32>,
    state: &State,
    joint_idx: usize,
    pos: Vector3<f32>,
    upstream_dof_frames: &[DofFrame],
) {
    let row0 = 3 * joint_idx;
    let row1: usize = row0 + 1;
    let row2: usize = row0 + 2;

    // A fixed-base tree's root isn't a state variable at all, so it
    // contributes no Jacobian columns.
    if state.kinematic_tree.n_root_dofs() > 0 {
        // Root translation (state cols 0..3):
        // Moving the root moves every keypoint by the same amount
        full_jacobian_buf[(row0, 0)] = 1.0;
        full_jacobian_buf[(row1, 1)] = 1.0;
        full_jacobian_buf[(row2, 2)] = 1.0;

        // Root rotation (state cols 3..6): rotate about root's current position
        let radius = pos - state.root_pos;
        for (i, axis) in [Vector3::x(), Vector3::y(), Vector3::z()]
            .iter()
            .enumerate()
        {
            let d = axis.cross(&radius);
            full_jacobian_buf[(row0, 3 + i)] = d.x;
            full_jacobian_buf[(row1, 3 + i)] = d.y;
            full_jacobian_buf[(row2, 3 + i)] = d.z;
        }
    }

    // Upstream joint DOFs. Note that each keypoint is only affected by a few
    // DOFs, so this is rather sparse.
    for frame in upstream_dof_frames {
        let jac = match frame.dof_type {
            // Rotating about `axis_world` through `origin_world` moves a
            // point in its orbit, so this is the "angular velocity" cross
            // product with the rotational axis.
            DofType::Hinge => frame.axis_world.cross(&(pos - frame.origin_world)),
            // Sliding along `axis_world` moves every downstream point by the
            // same amount along that direction, regardless of position.
            DofType::Slide => frame.axis_world,
        };
        full_jacobian_buf[(row0, frame.state_idx)] = jac.x;
        full_jacobian_buf[(row1, frame.state_idx)] = jac.y;
        full_jacobian_buf[(row2, frame.state_idx)] = jac.z;
    }
}
