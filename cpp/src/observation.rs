#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    use std::sync::Arc;

    use quickik_core::observation;

    use crate::error::ffi::Error;
    use crate::geometry::ffi::{Mat3, Vec2, Vec3};

    /// A pinhole camera, for keypoints observed as pixel coordinates:
    /// `p_cam = world2cam_rot_mat * p_world + world2cam_pos`, then
    /// `uv = (fx * x / z + cx, fy * y / z + cy)`.
    pub struct PinholeCamera {
        pub fx: f32,
        pub fy: f32,
        pub cx: f32,
        pub cy: f32,
        pub world2cam_pos: Vec3,
        pub world2cam_rot_mat: Mat3,
    }

    /// An orthographic view along world Z: `uv = scale * (x, y) + offset`.
    pub struct OrthographicViewXY {
        pub scale: f32,
        pub offset: Vec2,
    }

    /// Maps a 3D world position to 2D, e.g. camera pixel coordinates. Create
    /// one per camera and pass it to every observation from that camera.
    #[diplomat::opaque]
    pub struct Projection(pub(crate) Arc<observation::Projection>);

    impl Projection {
        pub fn from_pinhole_camera(camera: PinholeCamera) -> Box<Projection> {
            let camera = observation::PinholeCamera::new(
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.world2cam_pos.into(),
                camera.world2cam_rot_mat.into(),
            );
            Box::new(Projection(Arc::new(camera.into())))
        }

        pub fn from_orthographic_view_xy(view: OrthographicViewXY) -> Box<Projection> {
            let view = observation::OrthographicViewXY::new(view.scale, view.offset.into());
            Box::new(Projection(Arc::new(view.into())))
        }

        /// The camera, or none if this is another kind of projection.
        pub fn as_pinhole_camera(&self) -> Option<PinholeCamera> {
            match self.0.as_ref() {
                observation::Projection::PinholeCamera(camera) => Some(PinholeCamera {
                    fx: camera.fx,
                    fy: camera.fy,
                    cx: camera.cx,
                    cy: camera.cy,
                    world2cam_pos: camera.world2cam_pos.into(),
                    world2cam_rot_mat: camera.world2cam_rot_mat.into(),
                }),
                _ => None,
            }
        }

        /// The view, or none if this is another kind of projection.
        pub fn as_orthographic_view_xy(&self) -> Option<OrthographicViewXY> {
            match self.0.as_ref() {
                observation::Projection::OrthographicViewXY(view) => Some(OrthographicViewXY {
                    scale: view.scale,
                    offset: view.offset.into(),
                }),
                _ => None,
            }
        }

        /// The projected position of `pos_3d`.
        pub fn project(&self, pos_3d: Vec3) -> Vec2 {
            self.0.project(&pos_3d.into()).0.into()
        }
    }

    /// One frame's keypoint observations, in any order. A joint may be
    /// observed any number of times (e.g. once per camera) or not at all (e.g.
    /// occluded). Rust and Python use a plain list of `KeypointObservation`s
    /// instead.
    #[diplomat::opaque_mut]
    pub struct KeypointObservationList(pub(crate) Vec<observation::KeypointObservation>);

    impl KeypointObservationList {
        pub fn create() -> Box<KeypointObservationList> {
            Box::new(KeypointObservationList(Vec::new()))
        }

        /// Adds an observed 3D world position of joint `joint_idx`'s keypoint,
        /// e.g. triangulated from several cameras. `weight` is multiplied by
        /// the joint's `weight_multiplier` to give the final weight.
        pub fn add_position_3d(&mut self, joint_idx: usize, pos: Vec3, weight: f32) {
            let keypoint =
                observation::KeypointObservation::position_3d(joint_idx, pos.into(), weight);
            self.0.push(keypoint);
        }

        /// Adds an observed 2D position of joint `joint_idx`'s keypoint, in the
        /// space `projection` maps into. `weight` is as in `add_position_3d`.
        pub fn add_position_2d(
            &mut self,
            joint_idx: usize,
            pos: Vec2,
            projection: &Projection,
            weight: f32,
        ) {
            let projection = Arc::clone(&projection.0);
            let keypoint = observation::KeypointObservation::position_2d(
                joint_idx,
                pos.into(),
                projection,
                weight,
            );
            self.0.push(keypoint);
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }

        pub fn clear(&mut self) {
            self.0.clear();
        }
    }

    /// Many frames of observations, e.g. a recording, for
    /// `SequentialSolver::solve` and `solve_parallel`. Immutable.
    #[diplomat::opaque]
    pub struct ObservationSequence(pub(crate) observation::ObservationSequence);

    impl ObservationSequence {
        /// All frames' observations concatenated in `observations`: frame `i`
        /// is the next `frame_lengths[i]` observations. Returns an error if
        /// `frame_lengths` doesn't add up to `observations.len()`.
        pub fn from_flat(
            observations: &KeypointObservationList,
            frame_lengths: &[usize],
        ) -> Result<Box<ObservationSequence>, Box<Error>> {
            let sequence =
                observation::ObservationSequence::from_flat(observations.0.clone(), frame_lengths)?;
            Ok(Box::new(ObservationSequence(sequence)))
        }

        /// Builds frames with a fixed layout from flat row-major arrays: per
        /// frame, the 3D block (all joints, in joint order) if given, then one
        /// 2D block per view in `projections` order.
        ///
        /// - `pos_3d`: `(n_frames, n_joints, 3)`, with `weights_3d`:
        ///   `(n_frames, n_joints)`. Either both or neither.
        /// - `pos_2d`: `(n_frames, n_views, n_joints, 2)`, with `weights_2d`:
        ///   `(n_frames, n_views, n_joints)`, where `n_views` is the number of
        ///   `projections`. Required if and only if `projections` is non-empty.
        ///
        /// Every entry is kept, so every frame's Jacobian has the same rows. A
        /// weight that isn't positive (including NaN) marks a missing
        /// keypoint. Returns an error if an array has the wrong length.
        #[allow(clippy::too_many_arguments)]
        pub fn from_arrays(
            n_frames: usize,
            n_joints: usize,
            pos_3d: Option<&[f32]>,
            weights_3d: Option<&[f32]>,
            projections: &[&Projection],
            pos_2d: Option<&[f32]>,
            weights_2d: Option<&[f32]>,
        ) -> Result<Box<ObservationSequence>, Box<Error>> {
            let projections: Vec<_> = projections.iter().map(|p| Arc::clone(&p.0)).collect();
            let sequence = observation::ObservationSequence::from_arrays(
                n_frames,
                n_joints,
                pos_3d,
                weights_3d,
                &projections,
                pos_2d,
                weights_2d,
            )?;
            Ok(Box::new(ObservationSequence(sequence)))
        }

        /// Number of frames.
        pub fn len(&self) -> usize {
            self.0.len()
        }
    }
}
