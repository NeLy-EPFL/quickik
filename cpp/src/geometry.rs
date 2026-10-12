#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    /// A 2D vector, e.g. a pixel position.
    #[derive(Clone, Copy)]
    pub struct Vec2 {
        pub x: f32,
        pub y: f32,
    }

    /// A 3D vector, e.g. a world position.
    #[derive(Clone, Copy)]
    pub struct Vec3 {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    /// A rotation as a unit quaternion, with scalar part `w`.
    #[derive(Clone, Copy)]
    pub struct Quat {
        pub w: f32,
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    /// A 3x3 matrix, given by its rows.
    #[derive(Clone, Copy)]
    pub struct Mat3 {
        pub row0: Vec3,
        pub row1: Vec3,
        pub row2: Vec3,
    }
}

use nalgebra::{Matrix3, UnitQuaternion, Vector2, Vector3};

impl From<ffi::Vec2> for Vector2<f32> {
    fn from(v: ffi::Vec2) -> Self {
        Vector2::new(v.x, v.y)
    }
}

impl From<Vector2<f32>> for ffi::Vec2 {
    fn from(v: Vector2<f32>) -> Self {
        ffi::Vec2 { x: v.x, y: v.y }
    }
}

impl From<ffi::Vec3> for Vector3<f32> {
    fn from(v: ffi::Vec3) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f32>> for ffi::Vec3 {
    fn from(v: Vector3<f32>) -> Self {
        ffi::Vec3 {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<UnitQuaternion<f32>> for ffi::Quat {
    fn from(q: UnitQuaternion<f32>) -> Self {
        ffi::Quat {
            w: q.w,
            x: q.i,
            y: q.j,
            z: q.k,
        }
    }
}

impl From<ffi::Mat3> for Matrix3<f32> {
    fn from(m: ffi::Mat3) -> Self {
        let rows = [m.row0, m.row1, m.row2].map(|row| Vector3::from(row).transpose());
        Matrix3::from_rows(&rows)
    }
}

impl From<Matrix3<f32>> for ffi::Mat3 {
    fn from(m: Matrix3<f32>) -> Self {
        let row = |i: usize| ffi::Vec3 {
            x: m[(i, 0)],
            y: m[(i, 1)],
            z: m[(i, 2)],
        };
        ffi::Mat3 {
            row0: row(0),
            row1: row(1),
            row2: row(2),
        }
    }
}

/// Flattens a slice of 3D vectors into `3 * len` floats, without copying.
pub(crate) fn flatten(vectors: &[Vector3<f32>]) -> &[f32] {
    bytemuck::cast_slice(vectors)
}

/// Copies a matrix into a row-major vector.
pub(crate) fn row_major(matrix: &nalgebra::DMatrix<f32>) -> Vec<f32> {
    matrix.transpose().as_slice().to_vec()
}
