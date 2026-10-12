//! Forward kinematics, including the Jacobian of each keypoint's position with
//! respect to the state. Public for users who only need FK (e.g. to render a
//! known pose); the solvers use the same code.
//!
//! The tree is traversed as a flat loop over its precomputed parent-before-
//! child order. Each joint's world transform and each DOF's world axis/origin
//! are stored in arrays indexed by joint and by flat DOF index respectively, so
//! a keypoint's Jacobian can read all of its upstream DOFs directly: they all
//! belong to strict ancestors, which were visited earlier.

use std::sync::Arc;

use nalgebra::{DMatrix, Isometry3, Translation3, UnitQuaternion, Vector3};

use crate::body_plan::{DofType, KinematicTree};
use crate::error::{Result, bail};
use crate::state::State;

/// Buffers for [`forward_kinematics`], allocated once per tree and reused
/// across calls.
#[derive(Clone)]
pub struct ForwardKinematicsWorkspace {
    kinematic_tree: Arc<KinematicTree>,
    keypoint_positions: Vec<Vector3<f32>>,
    /// Compact per-keypoint Jacobians: column `i` of keypoint `k`'s is the
    /// derivative w.r.t. state index `tree.upstream_state_indices(k)[i]`. Every
    /// other column of the full Jacobian is zero.
    jacobian_columns: Vec<Vec<Vector3<f32>>>,
    /// World transform of each joint, including its own DOFs (what its
    /// children are placed relative to).
    joint_transforms: Vec<Isometry3<f32>>,
    /// World axis of each DOF, by flat DOF index.
    dof_axes_world: Vec<Vector3<f32>>,
    /// World position of each DOF's joint origin just before that DOF is
    /// applied, by flat DOF index. Only read for hinges.
    dof_origins_world: Vec<Vector3<f32>>,
}

impl ForwardKinematicsWorkspace {
    pub fn new(kinematic_tree: &Arc<KinematicTree>) -> Self {
        let n_joints = kinematic_tree.n_joints();
        let n_dofs = kinematic_tree.n_dofs();
        Self {
            keypoint_positions: vec![Vector3::zeros(); n_joints],
            jacobian_columns: (0..n_joints)
                .map(|k| vec![Vector3::zeros(); kinematic_tree.upstream_state_indices(k).len()])
                .collect(),
            joint_transforms: vec![Isometry3::identity(); n_joints],
            dof_axes_world: vec![Vector3::zeros(); n_dofs],
            dof_origins_world: vec![Vector3::zeros(); n_dofs],
            kinematic_tree: Arc::clone(kinematic_tree),
        }
    }

    pub fn kinematic_tree(&self) -> &Arc<KinematicTree> {
        &self.kinematic_tree
    }

    /// World-frame keypoint positions from the last [`forward_kinematics`]
    /// call, in joint order.
    pub fn keypoint_positions(&self) -> &[Vector3<f32>] {
        &self.keypoint_positions
    }

    /// Dense `(3 * n_joints, state_dim)` Jacobian of
    /// [`keypoint_positions`](Self::keypoint_positions) w.r.t. the state:
    /// keypoint `k`'s position is rows `3k..3k + 3`. Assembled from the
    /// internal compact form on each call.
    pub fn jacobian(&self) -> DMatrix<f32> {
        let tree = &self.kinematic_tree;
        let mut jacobian = DMatrix::zeros(3 * tree.n_joints(), tree.state_dim());
        for k in 0..tree.n_joints() {
            let columns = self.keypoint_jacobian(k);
            let state_indices = tree.upstream_state_indices(k);
            for (column, &state_idx) in columns.iter().zip(state_indices) {
                for axis in 0..3 {
                    jacobian[(3 * k + axis, state_idx)] = column[axis];
                }
            }
        }
        jacobian
    }

    /// Keypoint `joint_idx`'s compact Jacobian: one column per entry of
    /// `tree.upstream_state_indices(joint_idx)`.
    #[inline]
    pub(crate) fn keypoint_jacobian(&self, joint_idx: usize) -> &[Vector3<f32>] {
        &self.jacobian_columns[joint_idx]
    }
}

/// Computes every keypoint's world position for `state` and its Jacobian
/// w.r.t. the state. Results are read from `workspace`. Returns an error if
/// `state` doesn't have as many DOFs as the workspace's tree.
pub fn forward_kinematics(workspace: &mut ForwardKinematicsWorkspace, state: &State) -> Result<()> {
    if state.dof_values.len() != workspace.kinematic_tree.n_dofs() {
        bail!(
            "state has {} DOF values, but the workspace's tree has {} DOFs",
            state.dof_values.len(),
            workspace.kinematic_tree.n_dofs()
        );
    }
    forward_kinematics_unchecked(workspace, state);
    Ok(())
}

/// [`forward_kinematics`] without the argument check, for the solver's hot
/// loop (which checks once per solve instead).
pub(crate) fn forward_kinematics_unchecked(
    workspace: &mut ForwardKinematicsWorkspace,
    state: &State,
) {
    let tree = &workspace.kinematic_tree;
    let root_transform = Isometry3::from_parts(Translation3::from(state.root_pos), state.root_quat);

    for &j in tree.traversal_order() {
        let joint = &tree.joints()[j];
        let parent_transform = match joint.parent {
            None => root_transform,
            Some(p) => workspace.joint_transforms[p],
        };

        // The keypoint sits at the joint origin before the joint's own DOFs are
        // applied: a joint's own DOFs only move its descendants.
        let keypoint_pos =
            parent_transform.translation.vector + parent_transform.rotation * joint.offset_pos;
        let mut rotation = parent_transform.rotation * joint.offset_quat;
        let mut origin = keypoint_pos;
        let dof_start = tree.joint_dof_start_indices()[j];
        for (i, dof) in joint.dofs.iter().enumerate() {
            let d = dof_start + i;
            let axis_world = rotation * dof.axis.into_inner();
            workspace.dof_axes_world[d] = axis_world;
            workspace.dof_origins_world[d] = origin;
            let value = state.dof_values[d];
            match dof.dof_type {
                DofType::Hinge => rotation *= UnitQuaternion::from_axis_angle(&dof.axis, value),
                DofType::Slide => origin += axis_world * value,
            }
        }
        workspace.joint_transforms[j] = Isometry3::from_parts(Translation3::from(origin), rotation);
        workspace.keypoint_positions[j] = keypoint_pos;

        fill_keypoint_jacobian(
            tree,
            state,
            keypoint_pos,
            tree.upstream_state_indices(j),
            &workspace.dof_axes_world,
            &workspace.dof_origins_world,
            &mut workspace.jacobian_columns[j],
        );
    }
}

/// Writes one keypoint's compact Jacobian columns. Upstream state indices are
/// ascending, so the root's columns (if any) come first.
fn fill_keypoint_jacobian(
    tree: &KinematicTree,
    state: &State,
    keypoint_pos: Vector3<f32>,
    state_indices: &[usize],
    dof_axes_world: &[Vector3<f32>],
    dof_origins_world: &[Vector3<f32>],
    columns: &mut [Vector3<f32>],
) {
    let n_root_dofs = tree.n_root_dofs();
    if n_root_dofs > 0 {
        // Root translation moves every keypoint by the same amount.
        columns[0] = Vector3::x();
        columns[1] = Vector3::y();
        columns[2] = Vector3::z();
        // Root rotation (world-frame axis-angle) rotates about the root's
        // position: `e_i x r` for each world axis `e_i`.
        let r = keypoint_pos - state.root_pos;
        columns[3] = Vector3::new(0.0, -r.z, r.y);
        columns[4] = Vector3::new(r.z, 0.0, -r.x);
        columns[5] = Vector3::new(-r.y, r.x, 0.0);
    }
    let dof_columns = columns[n_root_dofs..].iter_mut();
    for (column, &state_idx) in dof_columns.zip(&state_indices[n_root_dofs..]) {
        let d = state_idx - n_root_dofs;
        *column = match tree.dof_types()[d] {
            // Rotating about `axis` through `origin` moves the point along its
            // orbit: the "angular velocity" cross product.
            DofType::Hinge => dof_axes_world[d].cross(&(keypoint_pos - dof_origins_world[d])),
            // Sliding moves every downstream point along the axis.
            DofType::Slide => dof_axes_world[d],
        };
    }
}
