//! This module defines the data structures for the state of the kinematic tree.

use std::sync::Arc;

use nalgebra::{DVector, UnitQuaternion, Vector3};

use crate::body_plan::KinematicTree;
use crate::utils::unit_quat_from_axis_angle_vec;

/// The pose being solved for.
#[derive(Clone, Debug)]
pub struct State {
    /// Pointer to the kinematic tree that defines the body.
    pub kinematic_tree: Arc<KinematicTree>,
    /// The position of the root joint in world coordinates.
    pub root_pos: Vector3<f32>,
    /// The rotation of the root joint in world coordinates.
    pub root_quat: UnitQuaternion<f32>,
    /// The values of all joint DOFs (angles in radian for hinges, positions for
    /// slides).
    pub dof_values: Vec<f32>,
}

impl State {
    pub fn state_dim(&self) -> usize {
        self.kinematic_tree.state_dim()
    }

    /// Create a new state at neutral pose.
    pub fn neutral_pose(kinematic_tree: Arc<KinematicTree>) -> Self {
        let mut dof_values = vec![0.0; kinematic_tree.n_dofs()];
        for joint in &kinematic_tree.joints {
            for (i, dof) in joint.dofs.iter().enumerate() {
                dof_values[joint.dof_startidx + i] = dof.neutral;
            }
        }
        Self {
            kinematic_tree,
            root_pos: Vector3::zeros(),
            root_quat: UnitQuaternion::identity(),
            dof_values,
        }
    }

    /// Applies a Gauss-Newton step in place.
    /// `delta.len()` must equal `self.state_dim()`.
    pub fn apply_delta(&mut self, delta: &DVector<f32>) {
        // Don't check in release builds as this happens in the hot loop.
        debug_assert_eq!(delta.len(), self.state_dim());

        // Root state -- absent (0 columns) for a fixed-base tree.
        let n_root_dofs = self.kinematic_tree.n_root_dofs();
        if n_root_dofs > 0 {
            let d_root_pos = Vector3::new(delta[0], delta[1], delta[2]);
            self.root_pos += d_root_pos;
            // delta rotation is represented as axis-angle vector (because it's
            // small enough), but the rotation itself is stored as a unit
            // quaternion to avoid the usual gimbal lock issues.
            let d_root_rot = Vector3::new(delta[3], delta[4], delta[5]);
            self.root_quat = unit_quat_from_axis_angle_vec(d_root_rot) * self.root_quat;
        }

        // Body DOF state, clamped to each DOF's angle limits (if any)
        let dofs = self.kinematic_tree.joints.iter().flat_map(|j| &j.dofs);
        for (i, (angle, dof)) in self.dof_values.iter_mut().zip(dofs).enumerate() {
            *angle += delta[n_root_dofs + i];
            if let Some([min, max]) = dof.limits {
                *angle = angle.clamp(min, max);
            }
        }
    }
}
