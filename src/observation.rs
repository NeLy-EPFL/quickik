//! This module defines an observation of a single keypoint, e.g. from MoCap
//! data. It also defines mappings (e.g. camera models) from 3D world
//! coordinates to 2D coordinates, which are used when keypoint observations are
//! only available in 2D.

use nalgebra::{Dyn, Matrix, Matrix2x3, Matrix3, Storage, StorageMut, Vector2, Vector3};

/// Observation of a single keypoint, e.g. from MoCap data.
///
/// A [`Position2D`] observation carries no mapper of its own -- the mapper
/// used to project the forward-kinematics position into 2D is specified once
/// when [`Solver`] is constructed.
///
/// [`Position2D`]: KeypointObservation::Position2D
/// [`Solver`]: crate::solver::Solver
#[derive(Clone, Copy, Debug)]
pub enum KeypointObservation {
    /// Not observed this frame (e.g. occluded).
    Missing,
    /// A 3D world position, e.g. triangulated from multiple calibrated
    /// cameras. `weight` is multiplied together with the keypoint's
    /// [`Joint::weight_scaler`] to give this observation's overall weight in
    /// the solve.
    ///
    /// [`Joint::weight_scaler`]: crate::body_plan::Joint::weight_scaler
    Position3D { obs_pos: Vector3<f32>, weight: f32 },
    /// A 2D pixel position from the single calibrated camera (or other
    /// mapper) that the consuming [`Solver`] was constructed with.
    /// `weight` is multiplied together with the keypoint's
    /// [`Joint::weight_scaler`] to give this observation's overall weight in
    /// the solve.
    ///
    /// [`Solver`]: crate::solver::Solver
    /// [`Joint::weight_scaler`]: crate::body_plan::Joint::weight_scaler
    Position2D { obs_pos: Vector2<f32>, weight: f32 },
}

/// Mapping 3D keypoint positions in world coordinates and their Jacobians to
/// 2D coordinates, in whatever format the MoCap data supplies. For example, if
/// the MoCap data provides 2D pixel coordinates from a camera, this might be a
/// camera calibration model. If the MoCap data already provides reprojected 2D
/// physical coordinates, this might simply be a reduced-rank identity mapping.
pub trait Mapper3Dto2D: Copy + std::fmt::Debug {
    /// Maps a 3D world position and its Jacobian to 2D, writing the projected
    /// Jacobian into `jacobian_2d_out` (shape `2 x jacobian_world3d.ncols()`)
    /// in place rather than allocating a new matrix -- this runs once per
    /// keypoint per Gauss-Newton iteration, on the solver's hot path.
    /// `jacobian_world3d`/`jacobian_2d_out` accept any storage (owned or a
    /// view) so callers don't need to copy a view into an owned matrix just
    /// to call this.
    fn project_3d_to_2d<S1, S2>(
        &self,
        pos_world3d: &Vector3<f32>,
        jacobian_world3d: &Matrix<f32, Dyn, Dyn, S1>,
        jacobian_2d_out: &mut Matrix<f32, Dyn, Dyn, S2>,
    ) -> Vector2<f32>
    where
        S1: Storage<f32, Dyn, Dyn>,
        S2: StorageMut<f32, Dyn, Dyn>;
}

/// Placeholder mapper for a [`Solver`] that receives 3D keypoint observations.
///
/// [`Solver`]: crate::solver::Solver
#[derive(Clone, Copy, Debug)]
pub struct NoMapper;

impl Mapper3Dto2D for NoMapper {
    fn project_3d_to_2d<S1, S2>(
        &self,
        _pos_world3d: &Vector3<f32>,
        _jacobian_world3d: &Matrix<f32, Dyn, Dyn, S1>,
        _jacobian_2d_out: &mut Matrix<f32, Dyn, Dyn, S2>,
    ) -> Vector2<f32>
    where
        S1: Storage<f32, Dyn, Dyn>,
        S2: StorageMut<f32, Dyn, Dyn>,
    {
        unreachable!(
            "NoMapper::project_3d_to_2d was called -- a Solver<NoMapper> (no mapper set) was \
             given a Position2D observation"
        )
    }
}

/// A pinhole camera for inverse kinematics from 2D keypoint observations.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Focal length in pixels (x)
    pub fx: f32,
    /// Focal length in pixels (y)
    pub fy: f32,
    /// Principal point (x)
    pub cx: f32,
    /// Principal point (y)
    pub cy: f32,
    /// World-to-camera translation
    pub world2cam_pos: Vector3<f32>,
    /// World-to-camera rotation
    /// (`p_cam = world2cam_rot_mat * p_world + world2cam_pos`)
    pub world2cam_rot_mat: Matrix3<f32>,
}

impl Mapper3Dto2D for Camera {
    /// Projects a world 3D keypoint position and its Jacobian to 2D.
    fn project_3d_to_2d<S1, S2>(
        &self,
        pos_world3d: &Vector3<f32>,
        jacobian_world3d: &Matrix<f32, Dyn, Dyn, S1>,
        jacobian_2d_out: &mut Matrix<f32, Dyn, Dyn, S2>,
    ) -> Vector2<f32>
    where
        S1: Storage<f32, Dyn, Dyn>,
        S2: StorageMut<f32, Dyn, Dyn>,
    {
        // Project position
        let pos_cam3d = self.world2cam_rot_mat * pos_world3d + self.world2cam_pos;
        let pos_cam2d = Vector2::new(
            self.fx * pos_cam3d.x / pos_cam3d.z + self.cx,
            self.fy * pos_cam3d.y / pos_cam3d.z + self.cy,
        );

        // Project Jacobian: d(cam2d)/d(cam3d) and its composition with
        // d(cam3d)/d(world3d) are both fixed 2x3/3x3 products (stack-only,
        // no allocation regardless of state_dim).
        let jac_d_cam2d_d_cam3d = Matrix2x3::new(
            self.fx / pos_cam3d.z,
            0.0,
            -self.fx * pos_cam3d.x / (pos_cam3d.z * pos_cam3d.z),
            0.0,
            self.fy / pos_cam3d.z,
            -self.fy * pos_cam3d.y / (pos_cam3d.z * pos_cam3d.z),
        );
        let jac_d_cam2d_d_world3d = jac_d_cam2d_d_cam3d * self.world2cam_rot_mat;

        // Chain with d(world3d)/d(state) (the caller's dynamic-width
        // Jacobian), writing directly into `jacobian_2d_out` -- done manually
        // rather than via a generic matrix multiply, so this (the only
        // dynamic-width step) never allocates an intermediate product.
        for col in 0..jacobian_world3d.ncols() {
            let jx = jacobian_world3d[(0, col)];
            let jy = jacobian_world3d[(1, col)];
            let jz = jacobian_world3d[(2, col)];
            jacobian_2d_out[(0, col)] = jac_d_cam2d_d_world3d[(0, 0)] * jx
                + jac_d_cam2d_d_world3d[(0, 1)] * jy
                + jac_d_cam2d_d_world3d[(0, 2)] * jz;
            jacobian_2d_out[(1, col)] = jac_d_cam2d_d_world3d[(1, 0)] * jx
                + jac_d_cam2d_d_world3d[(1, 1)] * jy
                + jac_d_cam2d_d_world3d[(1, 2)] * jz;
        }

        pos_cam2d
    }
}

/// A 2D X-Y view of a 3D keypoint, already in world coordinates.
#[derive(Clone, Copy, Debug)]
pub struct XYView;

impl Mapper3Dto2D for XYView {
    fn project_3d_to_2d<S1, S2>(
        &self,
        pos_world3d: &Vector3<f32>,
        jacobian_world3d: &Matrix<f32, Dyn, Dyn, S1>,
        jacobian_2d_out: &mut Matrix<f32, Dyn, Dyn, S2>,
    ) -> Vector2<f32>
    where
        S1: Storage<f32, Dyn, Dyn>,
        S2: StorageMut<f32, Dyn, Dyn>,
    {
        for col in 0..jacobian_world3d.ncols() {
            jacobian_2d_out[(0, col)] = jacobian_world3d[(0, col)];
            jacobian_2d_out[(1, col)] = jacobian_world3d[(1, col)];
        }
        Vector2::new(pos_world3d.x, pos_world3d.y)
    }
}
