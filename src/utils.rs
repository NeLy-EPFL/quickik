//! This module contains various utility functions.

use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};

pub fn quat_from_wxyz([w, x, y, z]: [f32; 4]) -> UnitQuaternion<f32> {
    UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z))
}

// Rotations less than this angle will be treated as zero
const ANGLE_EPSILON: f32 = 1e-12;

/// Converts an axis-angle vector to a unit quaternion.
/// The axis-angle is a single 3D vector where the norm specifies the angle.
pub fn unit_quat_from_axis_angle_vec(v: Vector3<f32>) -> UnitQuaternion<f32> {
    let angle = v.norm();
    if angle < ANGLE_EPSILON {
        UnitQuaternion::identity()
    } else {
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(v), angle)
    }
}
