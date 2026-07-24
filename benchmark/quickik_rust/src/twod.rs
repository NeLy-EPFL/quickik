//! Reprojects the existing 3D fixture targets down to 2D, so the same task
//! can also be benchmarked and correctness-checked from 2D-only observations.
//! Two mappers: a synthetic pinhole [`Camera`], fixed once per body (not per
//! frame, mirroring a real single-camera 2D-tracking setup) so it frames
//! every point ever fed to it, and the trivial [`XYView`] (just drops Z).

use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};
use quickik::observation::{Camera, KeypointObservation, Mapper3Dto2D};

/// Builds a single fixed pinhole camera that frames every point in `points`,
/// from directly below looking straight up (+Z) -- a "bottom view", matching
/// a real under-the-floor camera rig and looking along the same axis that
/// [`XYView`] always drops. Distance from the centroid is chosen so the
/// whole bounding sphere (with margin) stays inside the field of view.
///
/// Uses *normalized* image-plane coordinates (principal point at the origin:
/// `cx = cy = 0`) rather than real pixel coordinates. Pixel coordinates would
/// be O(1e2-1e3), while every other residual in this solve -- 3D positions,
/// and the neutral-pose prior's weight (tuned for model-unit-scale
/// residuals) -- is O(1). A residual that much larger swamps the prior in
/// the normal equations, removing the pull that keeps this
/// otherwise-underconstrained monocular fit out of mirror/local-minima
/// solutions.
///
/// The focal length is set to the working distance (`fx = fy = distance`),
/// not literally `1`: the projection's Jacobian is `d(u)/d(x_cam) = fx / z`,
/// so `fx = distance` keeps that derivative around 1 at the object's own
/// depth -- matching the O(1) Jacobian magnitude of [`XYView`] and the raw
/// 3D case, rather than attenuating it by `1 / distance`. Without this, the
/// same fixed regularization weight ends up with proportionally more pull
/// than intended, since it's added to the normal equations' diagonal as a
/// constant unaffected by the observation's own Jacobian scale.
pub fn synthetic_camera(points: impl Iterator<Item = Vector3<f32>>) -> Camera {
    let pts: Vec<Vector3<f32>> = points.collect();
    assert!(!pts.is_empty(), "synthetic_camera needs at least one point");
    let centroid = pts.iter().sum::<Vector3<f32>>() / pts.len() as f32;
    let radius = pts
        .iter()
        .map(|p| (p - centroid).norm())
        .fold(0.0f32, f32::max);

    const FOV_DEG: f32 = 60.0;
    // Headroom so the bounding sphere doesn't sit right at the frame edge.
    const MARGIN: f32 = 1.5;

    let half_fov = (FOV_DEG / 2.0f32).to_radians();
    let distance = radius * MARGIN / half_fov.sin();

    let forward = Vector3::z();
    let cam_pos = centroid - forward * distance;
    let world_up = Vector3::y(); // orthogonal to `forward` by construction
    let right = forward.cross(&world_up).normalize();
    let up = right.cross(&forward);
    let world2cam_rot_mat =
        Matrix3::from_rows(&[right.transpose(), up.transpose(), forward.transpose()]);
    let world2cam_pos = -(world2cam_rot_mat * cam_pos);

    Camera {
        fx: distance,
        fy: distance,
        cx: 0.0,
        cy: 0.0,
        world2cam_pos,
        world2cam_rot_mat,
    }
}

/// Projects a single 3D position through `camera`, discarding the projected
/// Jacobian output -- only needed here to build a 2D observation from a known
/// 3D point, not to solve anything.
fn project_position(camera: &Camera, pos: Vector3<f32>) -> Vector2<f32> {
    let mut jacobian_2d_placeholder = DMatrix::zeros(2, 1);
    camera.project_3d_to_2d(&pos, &DMatrix::zeros(3, 1), &mut jacobian_2d_placeholder)
}

/// `target_ego` reprojected through `camera` into `Position2D` observations,
/// with `Missing` prepended for the free-floating root (same convention as
/// [`crate::correctness::build_observations`]).
pub fn observations_2d_camera(target_ego: &[[f32; 3]], camera: &Camera) -> Vec<KeypointObservation> {
    let mut obs = Vec::with_capacity(target_ego.len() + 1);
    obs.push(KeypointObservation::Missing);
    obs.extend(target_ego.iter().map(|&[x, y, z]| KeypointObservation::Position2D {
        obs_pos: project_position(camera, Vector3::new(x, y, z)),
        weight: 1.0,
    }));
    obs
}

/// `target_ego` reprojected via [`quickik::observation::XYView`] (its x/y
/// coordinates, unchanged) into `Position2D` observations, with `Missing`
/// prepended for the root.
pub fn observations_2d_xyview(target_ego: &[[f32; 3]]) -> Vec<KeypointObservation> {
    let mut obs = Vec::with_capacity(target_ego.len() + 1);
    obs.push(KeypointObservation::Missing);
    obs.extend(
        target_ego
            .iter()
            .map(|&[x, y, _z]| KeypointObservation::Position2D {
                obs_pos: Vector2::new(x, y),
                weight: 1.0,
            }),
    );
    obs
}
