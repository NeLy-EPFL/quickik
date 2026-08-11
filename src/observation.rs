//! Keypoint observations, and the projection describing what space a solver's
//! observations live in.

use nalgebra::{DMatrixView, DMatrixViewMut, Matrix2x3, Matrix3, Vector2, Vector3};

/// Observation of a single keypoint, e.g. from MoCap data.
///
/// A [`Position2D`] observation carries no projection of its own: the
/// projection is fixed once (when [`Solver`] is constructed), so every 2D
/// observation a given solver sees lives in the same space.
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
    /// A 2D position in whatever space the consuming [`Solver`]'s
    /// [`Projection`] maps into (e.g. camera pixel coordinates). `weight` is
    /// multiplied together with the keypoint's [`Joint::weight_scaler`] to
    /// give this observation's overall weight in the solve.
    ///
    /// [`Solver`]: crate::solver::Solver
    /// [`Joint::weight_scaler`]: crate::body_plan::Joint::weight_scaler
    Position2D { obs_pos: Vector2<f32>, weight: f32 },
}

/// Defines the space that a solver's keypoint observations live in, and how
/// positions solved by forward kinematics are mapped into it. QuickIK supports
/// three kinds of projection:
/// 
/// - Null projection for observations that are already in 3D world coordinates
///   ([`KeypointObservation::Position3D`]). This projection is created with
///   [`Projection::new_3d`].
/// - Orthographic X-Y projection for [`KeypointObservation::Position2D`]
///   observations. This requires keypoint positions to be in world coordinates,
///   but missing Z. Created with [`Projection::new_ortho_xy`].
/// - Pinhole camera projection for [`KeypointObservation::Position2D`]
///   observations as seen from a pinhole camera. Created with
///   [`Projection::new_pinhole_camera`] using camera parameters.
///
/// Adding a fourth (a distortion model, say) means adding a variant here rather
/// than implementing anything: with only these cases to serve, an open trait
/// bought nothing that the bindings could use, since neither Python nor C++ can
/// supply a projection of their own across the FFI boundary.
#[derive(Clone, Copy, Debug)]
pub struct Projection(Kind);

#[derive(Clone, Copy, Debug)]
enum Kind {
    World3D,
    OrthoXY,
    PinholeCamera {
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        world2cam_pos: Vector3<f32>,
        world2cam_rot_mat: Matrix3<f32>,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Self::new_3d()
    }
}

impl Projection {
    /// Observations are 3D world positions: nothing is ever projected, and
    /// [`project_to_2d`](Self::project_to_2d) must not be called.
    pub fn new_3d() -> Self {
        Projection(Kind::World3D)
    }

    /// An orthographic X-Y view, for keypoints already reprojected into world
    /// X-Y coordinates: drops Z and passes X-Y through unchanged.
    pub fn new_ortho_xy() -> Self {
        Projection(Kind::OrthoXY)
    }

    /// A pinhole camera, for keypoints observed as pixel coordinates.
    ///
    /// `fx`/`fy` are focal lengths in pixels and `cx`/`cy` the principal point.
    /// `world2cam_rot_mat` and `world2cam_pos` give the extrinsics as
    /// `p_cam = world2cam_rot_mat * p_world + world2cam_pos`.
    pub fn new_pinhole_camera(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        world2cam_pos: Vector3<f32>,
        world2cam_rot_mat: Matrix3<f32>,
    ) -> Self {
        Projection(Kind::PinholeCamera {
            fx,
            fy,
            cx,
            cy,
            world2cam_pos,
            world2cam_rot_mat,
        })
    }

    /// Whether observations are 3D world positions (i.e. this is
    /// [`new_3d`](Self::new_3d)), so nothing needs projecting.
    pub fn is_3d(&self) -> bool {
        matches!(self.0, Kind::World3D)
    }

    /// Returns the projected position, and writes `d(pos_2d)/d(state)` into
    /// `jacobian_2d`.
    ///
    /// `jacobian_3d` is `d(pos_3d)/d(state)`, shaped `3 x n`, and `jacobian_2d`
    /// is the `2 x n` output for the same `n`. `n` is whatever the caller
    /// passes -- the solver narrows it to just the state columns a given
    /// keypoint actually depends on -- so the Jacobian is an out-parameter
    /// rather than a return value: this runs once per 2D keypoint per solver
    /// iteration, and returning a matrix of runtime width would allocate on
    /// that path.
    ///
    /// # Panics
    ///
    /// If this is [`new_3d`](Self::new_3d), which has nothing to project.
    /// `Solver::solve` checks [`is_3d`](Self::is_3d) up front and never reaches
    /// here in that case.
    pub fn project_to_2d(
        &self,
        pos_3d: &Vector3<f32>,
        jacobian_3d: &DMatrixView<'_, f32>,
        jacobian_2d: &mut DMatrixViewMut<'_, f32>,
    ) -> Vector2<f32> {
        match self.0 {
            Kind::World3D => panic!(
                "Projection::project_to_2d called on Projection::new_3d(), which projects nothing"
            ),
            Kind::OrthoXY => {
                chain_jacobian(
                    &Matrix2x3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    jacobian_3d,
                    jacobian_2d,
                );
                Vector2::new(pos_3d.x, pos_3d.y)
            }
            Kind::PinholeCamera {
                fx,
                fy,
                cx,
                cy,
                world2cam_pos,
                world2cam_rot_mat,
            } => {
                let pos_cam3d = world2cam_rot_mat * pos_3d + world2cam_pos;
                let inv_z = 1.0 / pos_cam3d.z;

                // d(pos_2d)/d(pos_cam3d), chained back through the
                // world-to-camera rotation to give d(pos_2d)/d(pos_3d).
                let d_pos2d_d_cam3d = Matrix2x3::new(
                    fx * inv_z,
                    0.0,
                    -fx * pos_cam3d.x * inv_z * inv_z,
                    0.0,
                    fy * inv_z,
                    -fy * pos_cam3d.y * inv_z * inv_z,
                );
                chain_jacobian(
                    &(d_pos2d_d_cam3d * world2cam_rot_mat),
                    jacobian_3d,
                    jacobian_2d,
                );

                Vector2::new(fx * pos_cam3d.x * inv_z + cx, fy * pos_cam3d.y * inv_z + cy)
            }
        }
    }
}

/// Chains a projection's `2 x 3` Jacobian onto `jacobian_3d`, writing the
/// `2 x n` result into `jacobian_2d`.
///
/// Written as an explicit loop rather than a matrix product because this is the
/// only step whose width isn't known at compile time: a product would allocate
/// an intermediate, whereas the `2 x 3` factor each variant supplies is
/// stack-only.
fn chain_jacobian(
    d_pos2d_d_pos3d: &Matrix2x3<f32>,
    jacobian_3d: &DMatrixView<'_, f32>,
    jacobian_2d: &mut DMatrixViewMut<'_, f32>,
) {
    for col in 0..jacobian_3d.ncols() {
        let (jx, jy, jz) = (
            jacobian_3d[(0, col)],
            jacobian_3d[(1, col)],
            jacobian_3d[(2, col)],
        );
        jacobian_2d[(0, col)] = d_pos2d_d_pos3d[(0, 0)] * jx
            + d_pos2d_d_pos3d[(0, 1)] * jy
            + d_pos2d_d_pos3d[(0, 2)] * jz;
        jacobian_2d[(1, col)] = d_pos2d_d_pos3d[(1, 0)] * jx
            + d_pos2d_d_pos3d[(1, 1)] * jy
            + d_pos2d_d_pos3d[(1, 2)] * jz;
    }
}
