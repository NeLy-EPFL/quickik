//! This module defines the data structures for the state of the kinematic tree.

use std::sync::Arc;

use nalgebra::{DVector, UnitQuaternion, Vector3};

use crate::body_plan::{KinematicTree, N_ROOT_DOFS};
use crate::utils::unit_quat_from_axis_angle_vec;

/// The pose being solved for.
#[derive(Clone, Debug)]
pub struct State {
    /// Pointer to the kinematic tree that defines the body
    pub kinematic_tree: Arc<KinematicTree>,
    /// The position of the root joint in world coordinates
    pub root_pos: Vector3<f32>,
    /// The rotation of the root joint in world coordinates
    pub root_rot: UnitQuaternion<f32>,
    /// The angles of all joint DOFs
    pub dof_angles: Vec<f32>,
}

impl State {
    pub fn state_dim(&self) -> usize {
        self.kinematic_tree.state_dim()
    }

    /// Create a new state at neutral pose.
    pub fn neutral_pose(kinematic_tree: Arc<KinematicTree>) -> Self {
        let mut dof_angles = vec![0.0; kinematic_tree.n_dofs()];
        for joint in &kinematic_tree.joints {
            for (i, dof) in joint.dofs.iter().enumerate() {
                dof_angles[joint.dof_offset + i] = dof.neutral_angle;
            }
        }
        Self {
            kinematic_tree,
            root_pos: Vector3::zeros(),
            root_rot: UnitQuaternion::identity(),
            dof_angles,
        }
    }

    /// Applies a Gauss-Newton step in place.
    /// `delta.len()` must equal `self.state_dim()`.
    pub fn apply_delta(&mut self, delta: &DVector<f32>) {
        debug_assert_eq!(delta.len(), self.state_dim());

        // Root state
        let d_root_pos = Vector3::new(delta[0], delta[1], delta[2]);
        self.root_pos += d_root_pos;
        let d_root_rot = Vector3::new(delta[3], delta[4], delta[5]);
        self.root_rot = unit_quat_from_axis_angle_vec(d_root_rot) * self.root_rot;

        // Body DOF state
        for (i, angle) in self.dof_angles.iter_mut().enumerate() {
            *angle += delta[N_ROOT_DOFS + i];
        }
    }
}
