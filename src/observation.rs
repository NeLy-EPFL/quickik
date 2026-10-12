//! Keypoint observations and the 2D projections they may carry.
//!
//! Define each projection (e.g. each camera) once as an `Arc<Projection>` and
//! give every observation from it a clone of that `Arc`, i.e. a shared pointer
//! to the same projection rather than a copy of it.

use std::sync::Arc;

use nalgebra::{Matrix2x3, Matrix3, Vector2, Vector3};

use crate::error::{Result, bail};

// =============================================================================
//  Projections
// =============================================================================

/// A pinhole camera, for keypoints observed as pixel coordinates:
/// `p_cam = world2cam_rot_mat * p_world + world2cam_pos`, then
/// `uv = (fx * x / z + cx, fy * y / z + cy)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeCamera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub world2cam_pos: Vector3<f32>,
    pub world2cam_rot_mat: Matrix3<f32>,
}

impl PinholeCamera {
    pub fn new(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        world2cam_pos: Vector3<f32>,
        world2cam_rot_mat: Matrix3<f32>,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            world2cam_pos,
            world2cam_rot_mat,
        }
    }

    /// Returns the projected position and its derivative w.r.t. `pos_3d`.
    pub fn project(&self, pos_3d: &Vector3<f32>) -> (Vector2<f32>, Matrix2x3<f32>) {
        let p = self.world2cam_rot_mat * pos_3d + self.world2cam_pos;
        let inv_z = 1.0 / p.z;
        let uv = Vector2::new(
            self.fx * p.x * inv_z + self.cx,
            self.fy * p.y * inv_z + self.cy,
        );
        #[rustfmt::skip]
        let d_uv_d_cam = Matrix2x3::new(
            self.fx * inv_z, 0.0,             -self.fx * p.x * inv_z * inv_z,
            0.0,             self.fy * inv_z, -self.fy * p.y * inv_z * inv_z,
        );
        (uv, d_uv_d_cam * self.world2cam_rot_mat)
    }
}

/// An orthographic view along world Z: `uv = scale * (x, y) + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrthographicViewXY {
    pub scale: f32,
    pub offset: Vector2<f32>,
}

impl OrthographicViewXY {
    pub fn new(scale: f32, offset: Vector2<f32>) -> Self {
        Self { scale, offset }
    }

    /// Returns the projected position and its derivative w.r.t. `pos_3d`.
    pub fn project(&self, pos_3d: &Vector3<f32>) -> (Vector2<f32>, Matrix2x3<f32>) {
        let uv = self.scale * Vector2::new(pos_3d.x, pos_3d.y) + self.offset;
        let s = self.scale;
        (uv, Matrix2x3::new(s, 0.0, 0.0, 0.0, s, 0.0))
    }
}

/// Maps a 3D world position to 2D, e.g. camera pixel coordinates. Build one
/// from a specific projection with `.into()`, e.g.
/// `let projection: Projection = PinholeCamera::new(..).into();`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    PinholeCamera(PinholeCamera),
    OrthographicViewXY(OrthographicViewXY),
}

impl Projection {
    /// Returns the projected position and its derivative w.r.t. `pos_3d`.
    ///
    /// `#[inline(always)]` because this runs once per 2D observation per
    /// solver iteration, and without LTO (e.g. in a downstream crate built with
    /// default release settings) LLVM declines to inline it, which measured ~7%
    /// slower on 2D benchmarks with the previous version of this function.
    #[inline(always)]
    pub fn project(&self, pos_3d: &Vector3<f32>) -> (Vector2<f32>, Matrix2x3<f32>) {
        match self {
            Projection::PinholeCamera(camera) => camera.project(pos_3d),
            Projection::OrthographicViewXY(view) => view.project(pos_3d),
        }
    }
}

impl From<PinholeCamera> for Projection {
    fn from(camera: PinholeCamera) -> Self {
        Projection::PinholeCamera(camera)
    }
}

impl From<OrthographicViewXY> for Projection {
    fn from(view: OrthographicViewXY) -> Self {
        Projection::OrthographicViewXY(view)
    }
}

// =============================================================================
//  Observations
// =============================================================================

/// What was observed: a 3D world position, or a 2D position together with the
/// projection it lives in.
#[derive(Clone, Debug)]
pub enum ObservedPosition {
    Position3D(Vector3<f32>),
    Position2D(Vector2<f32>, Arc<Projection>),
}

/// One observation of one joint's keypoint.
///
/// A frame's observations are a flat list in any order. A joint may appear any
/// number of times (e.g. once per camera) or not at all (e.g. occluded): leave
/// missing keypoints out. An observation with weight 0 contributes nothing to
/// the solve but still gets its rows in
/// [`SolverResult::jacobian`](crate::solver::SolverResult::jacobian).
#[derive(Clone, Debug)]
pub struct KeypointObservation {
    pub joint_idx: usize,
    pub position: ObservedPosition,
    /// Multiplied by the joint's
    /// [`weight_multiplier`](crate::body_plan::Joint::weight_multiplier) to
    /// give the final weight.
    pub weight: f32,
}

impl KeypointObservation {
    /// A 3D world position, e.g. triangulated from several cameras.
    pub fn position_3d(joint_idx: usize, pos: Vector3<f32>, weight: f32) -> Self {
        Self {
            joint_idx,
            position: ObservedPosition::Position3D(pos),
            weight,
        }
    }

    /// A 2D position in the space `projection` maps into.
    pub fn position_2d(
        joint_idx: usize,
        pos: Vector2<f32>,
        projection: Arc<Projection>,
        weight: f32,
    ) -> Self {
        Self {
            joint_idx,
            position: ObservedPosition::Position2D(pos, projection),
            weight,
        }
    }

    /// Number of residual (and Jacobian) rows this observation contributes.
    pub(crate) fn n_rows(&self) -> usize {
        match self.position {
            ObservedPosition::Position3D(_) => 3,
            ObservedPosition::Position2D(..) => 2,
        }
    }
}

// =============================================================================
//  Sequences
// =============================================================================

/// Many frames of observations, e.g. a recording (for
/// [`SequentialSolver`](crate::sequential_solver::SequentialSolver)) or a
/// batch of independent items (for
/// [`BatchedSolver`](crate::batched_solver::BatchedSolver)). Immutable.
#[derive(Clone, Debug)]
pub struct ObservationSequence {
    frames: Vec<Vec<KeypointObservation>>,
}

impl ObservationSequence {
    /// One list of observations per frame. Frames may differ in which and how
    /// many observations they have (e.g. moving cameras).
    pub fn from_lists(frames: Vec<Vec<KeypointObservation>>) -> Self {
        Self { frames }
    }

    /// Same as [`from_lists`](Self::from_lists), with all frames concatenated:
    /// frame `i` is the next `frame_lengths[i]` observations. Mainly for the
    /// C++ bindings, which can't pass nested lists. Returns an error if
    /// `frame_lengths` doesn't add up to `observations.len()`.
    pub fn from_flat(
        observations: Vec<KeypointObservation>,
        frame_lengths: &[usize],
    ) -> Result<Self> {
        let total: usize = frame_lengths.iter().sum();
        if total != observations.len() {
            bail!(
                "frame_lengths add up to {total}, but {} observations were given",
                observations.len()
            );
        }
        let mut observations = observations.into_iter();
        let frames = frame_lengths
            .iter()
            .map(|&len| observations.by_ref().take(len).collect())
            .collect();
        Ok(Self { frames })
    }

    /// Builds frames with a fixed layout from flat row-major arrays: per frame,
    /// the 3D block (all joints, in joint order) if given, then one 2D block
    /// per view in `projections` order.
    ///
    /// - `pos_3d`: `(n_frames, n_joints, 3)`, with `weights_3d`:
    ///   `(n_frames, n_joints)`. Either both or neither.
    /// - `pos_2d`: `(n_frames, n_views, n_joints, 2)`, with `weights_2d`:
    ///   `(n_frames, n_views, n_joints)`, where `n_views = projections.len()`.
    ///   Required if and only if `projections` is non-empty.
    ///
    /// Every entry is kept, so every frame's Jacobian has the same rows. A
    /// weight that isn't positive (including NaN) is stored as 0 with a zeroed
    /// position, i.e. a missing keypoint that can't poison the solve.
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(
        n_frames: usize,
        n_joints: usize,
        pos_3d: Option<&[f32]>,
        weights_3d: Option<&[f32]>,
        projections: &[Arc<Projection>],
        pos_2d: Option<&[f32]>,
        weights_2d: Option<&[f32]>,
    ) -> Result<Self> {
        let n_views = projections.len();
        let data_3d = match (pos_3d, weights_3d) {
            (None, None) => None,
            (Some(pos), Some(weights)) => {
                check_len("pos_3d", pos, n_frames * n_joints * 3)?;
                check_len("weights_3d", weights, n_frames * n_joints)?;
                Some((pos, weights))
            }
            _ => bail!("pos_3d and weights_3d must be given together"),
        };
        let data_2d = match (pos_2d, weights_2d) {
            (None, None) if n_views == 0 => None,
            (Some(pos), Some(weights)) if n_views > 0 => {
                check_len("pos_2d", pos, n_frames * n_views * n_joints * 2)?;
                check_len("weights_2d", weights, n_frames * n_views * n_joints)?;
                Some((pos, weights))
            }
            _ => {
                bail!("pos_2d and weights_2d must be given if and only if projections is non-empty")
            }
        };

        let mut frames = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let mut frame = Vec::new();
            if let Some((pos, weights)) = data_3d {
                for j in 0..n_joints {
                    let i = f * n_joints + j;
                    let (p, w) = match weights[i] > 0.0 {
                        true => (
                            Vector3::new(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]),
                            weights[i],
                        ),
                        false => (Vector3::zeros(), 0.0),
                    };
                    frame.push(KeypointObservation::position_3d(j, p, w));
                }
            }
            if let Some((pos, weights)) = data_2d {
                for (v, projection) in projections.iter().enumerate() {
                    for j in 0..n_joints {
                        let i = (f * n_views + v) * n_joints + j;
                        let (uv, w) = match weights[i] > 0.0 {
                            true => (Vector2::new(pos[2 * i], pos[2 * i + 1]), weights[i]),
                            false => (Vector2::zeros(), 0.0),
                        };
                        let projection = Arc::clone(projection);
                        frame.push(KeypointObservation::position_2d(j, uv, projection, w));
                    }
                }
            }
            frames.push(frame);
        }
        Ok(Self::from_lists(frames))
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Each frame's observations.
    pub fn frames(&self) -> &[Vec<KeypointObservation>] {
        &self.frames
    }
}

fn check_len(name: &str, data: &[f32], expected: usize) -> Result<()> {
    if data.len() != expected {
        bail!("{name} has {} elements, expected {expected}", data.len());
    }
    Ok(())
}
