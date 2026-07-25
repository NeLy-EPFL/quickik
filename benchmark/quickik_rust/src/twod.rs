//! Reprojects the existing 3D fixture targets down to 2D (via
//! [`XYView`](quickik::observation::XYView), which just drops Z), so the
//! same task can also be benchmarked and correctness-checked from 2D-only
//! observations.

use nalgebra::Vector2;
use quickik::observation::KeypointObservation;

/// `target_ego` reprojected via [`quickik::observation::XYView`] (its x/y
/// coordinates, unchanged) into `Position2D` observations, with `Missing`
/// prepended for the free-floating root (same convention as
/// [`crate::correctness::build_observations`]).
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
