//! This module defines the data structures for the state of the kinematic tree.

use std::sync::Arc;

use nalgebra::{DVector, UnitQuaternion, Vector3};

use crate::body_plan::KinematicTree;
use crate::utils::unit_quat_from_axis_angle_vec;

/// The pose being solved for.
#[derive(Clone, Debug)]
pub struct State {
    /// The kinematic tree that defines the body.
    pub kinematic_tree: Arc<KinematicTree>,
    /// The position of the root joint in world coordinates. Kept at the origin
    /// for a fixed-base tree.
    pub root_pos: Vector3<f32>,
    /// The rotation of the root joint in world coordinates. Kept at identity
    /// for a fixed-base tree.
    pub root_quat: UnitQuaternion<f32>,
    /// The values of all joint DOFs (angles in radians for hinges, positions
    /// for slides), in the tree's joint order, then each joint's DOF order.
    pub dof_values: Vec<f32>,
}

impl State {
    /// Creates a state at the neutral pose: every DOF at its
    /// [`neutral`](crate::body_plan::Dof::neutral) value, root at the origin
    /// with identity rotation.
    pub fn neutral_pose(kinematic_tree: Arc<KinematicTree>) -> Self {
        Self {
            dof_values: kinematic_tree.dof_neutrals().to_vec(),
            kinematic_tree,
            root_pos: Vector3::zeros(),
            root_quat: UnitQuaternion::identity(),
        }
    }

    pub fn state_dim(&self) -> usize {
        self.kinematic_tree.state_dim()
    }

    /// Applies a Gauss-Newton step in place, clamping DOFs to their limits.
    /// `delta.len()` must equal `self.state_dim()`.
    pub(crate) fn apply_delta(&mut self, delta: &DVector<f32>) {
        // Don't check in release builds as this happens in the hot loop.
        debug_assert_eq!(delta.len(), self.state_dim());

        let tree = &self.kinematic_tree;
        let n_root_dofs = tree.n_root_dofs();
        if n_root_dofs > 0 {
            self.root_pos += Vector3::new(delta[0], delta[1], delta[2]);
            // The delta rotation is a world-frame axis-angle vector (small
            // enough not to need anything else), but the rotation itself is
            // stored as a unit quaternion to avoid gimbal lock.
            let d_root_rot = Vector3::new(delta[3], delta[4], delta[5]);
            self.root_quat = unit_quat_from_axis_angle_vec(d_root_rot) * self.root_quat;
        }

        for (i, (value, limits)) in self
            .dof_values
            .iter_mut()
            .zip(tree.dof_limits())
            .enumerate()
        {
            *value += delta[n_root_dofs + i];
            if let Some([min, max]) = *limits {
                *value = value.clamp(min, max);
            }
        }
    }
}
