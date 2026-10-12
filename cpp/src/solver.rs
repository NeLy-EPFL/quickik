#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    use quickik_core::solver;

    use crate::body_plan::ffi::KinematicTree;
    use crate::error::ffi::Error;
    use crate::geometry::ffi::Vec3;
    use crate::geometry::{flatten, row_major};
    use crate::observation::ffi::KeypointObservationList;
    use crate::state::ffi::State;

    /// Iteration settings (fixed at creation) plus priors (added with the
    /// `add_*` methods, each at most once). No priors by default.
    ///
    /// Copied into each solver on creation. Checked there too: root priors on
    /// a fixed-base tree, and smoothness priors in `SingleFrameSolver`, are
    /// errors.
    #[diplomat::opaque_mut]
    pub struct SolverConfig(pub(crate) solver::SolverConfig);

    impl SolverConfig {
        /// - `n_iterations`: maximum number of Gauss-Newton steps per frame.
        /// - `position_tolerance`, `angle_tolerance`: stop early once a step's
        ///   largest root-position component is at most `position_tolerance`
        ///   and its largest angle component (root rotation and every DOF) is
        ///   at most `angle_tolerance` (radians). Set both to 0 to always run
        ///   `n_iterations` steps.
        /// - `damping`: Levenberg-Marquardt damping, relative to the normal
        ///   equations' diagonal (floored at 1). Only for numerical stability,
        ///   so keep it very small.
        pub fn create(
            #[diplomat::attr(auto, default_value = 10)] n_iterations: usize,
            #[diplomat::attr(auto, default_value = 1e-3)] position_tolerance: f32,
            #[diplomat::attr(auto, default_value = 1e-3)] angle_tolerance: f32,
            #[diplomat::attr(auto, default_value = 1e-6)] damping: f32,
        ) -> Box<SolverConfig> {
            let config = solver::SolverConfig::new(
                n_iterations,
                position_tolerance,
                angle_tolerance,
                damping,
            );
            Box::new(SolverConfig(config))
        }

        pub fn n_iterations(&self) -> usize {
            self.0.n_iterations()
        }

        pub fn position_tolerance(&self) -> f32 {
            self.0.position_tolerance()
        }

        pub fn angle_tolerance(&self) -> f32 {
            self.0.angle_tolerance()
        }

        pub fn damping(&self) -> f32 {
            self.0.damping()
        }

        /// Pulls each DOF toward its neutral value, with per-DOF weight
        /// `weight * weight_multiplier`. Improves robustness to missing or
        /// noisy keypoints, but biases the solution toward the neutral pose.
        pub fn add_dof_neutral_prior(&mut self, weight: f32) -> Result<(), Box<Error>> {
            Ok(self.0.add_dof_neutral_prior(weight)?)
        }

        /// Pulls the root's local `body_up` axis toward the world-frame
        /// `world_up` (e.g. +z, or a tilted vector on a slope), leaving
        /// rotation about the up axis free. Both vectors are normalized.
        pub fn add_root_upright_prior(
            &mut self,
            weight: f32,
            body_up: Vec3,
            world_up: Vec3,
        ) -> Result<(), Box<Error>> {
            Ok(self
                .0
                .add_root_upright_prior(weight, body_up.into(), world_up.into())?)
        }

        /// Pulls each DOF toward its value in the previous frame, with per-DOF
        /// weight `weight * weight_multiplier`. Inactive on a sequence's first
        /// frame.
        pub fn add_dof_smoothness_prior(&mut self, weight: f32) -> Result<(), Box<Error>> {
            Ok(self.0.add_dof_smoothness_prior(weight)?)
        }

        /// Pulls the root position and rotation toward the previous frame's,
        /// with separate weights since their units differ (length vs radians).
        /// Inactive on a sequence's first frame.
        pub fn add_root_smoothness_prior(
            &mut self,
            position_weight: f32,
            rotation_weight: f32,
        ) -> Result<(), Box<Error>> {
            Ok(self
                .0
                .add_root_smoothness_prior(position_weight, rotation_weight)?)
        }
    }

    /// One frame's converged pose and, optionally, its linearization.
    #[diplomat::opaque]
    pub struct SolverResult {
        pub(crate) state: State,
        pub(crate) keypoint_pos: Option<Vec<nalgebra::Vector3<f32>>>,
        /// Row-major copies of the core result's matrices.
        pub(crate) jacobian: Option<Vec<f32>>,
        pub(crate) cholesky_l: Option<Vec<f32>>,
    }

    impl SolverResult {
        pub fn state<'a>(&'a self) -> &'a State {
            &self.state
        }

        /// `n_joints * 3`: world-frame keypoint positions at `state()`, in
        /// joint order. None unless the solve had `with_fk = true`.
        pub fn keypoint_pos<'a>(&'a self) -> Option<&'a [f32]> {
            self.keypoint_pos.as_deref().map(flatten)
        }

        /// Row-major `(n_rows, state_dim)` residual Jacobian: one row block
        /// per observation, in the frame's input order, 3 rows for a 3D
        /// observation and 2 for a 2D one (already mapped through its
        /// projection). Weight-0 observations get rows too. Linearized at the
        /// pose just before the last iteration's update, which is small at
        /// convergence. None unless the solve had `with_grad = true`.
        pub fn jacobian<'a>(&'a self) -> Option<&'a [f32]> {
            self.jacobian.as_deref()
        }

        /// Row-major `(state_dim, state_dim)` lower-triangular Cholesky factor
        /// `L` of the normal-equations matrix (priors and damping included) at
        /// the same linearization as `jacobian()`. None unless the solve had
        /// `with_grad = true` and that matrix was positive-definite.
        pub fn cholesky_l<'a>(&'a self) -> Option<&'a [f32]> {
            self.cholesky_l.as_deref()
        }
    }

    /// Per-frame results of a sequence solve. Supports range-based `for`.
    #[diplomat::opaque]
    pub struct SolverResultList(pub(crate) Vec<SolverResult>);

    impl SolverResultList {
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// The result of frame `i`, or null if out of range.
        #[diplomat::attr(auto, indexer)]
        pub fn get<'a>(&'a self, i: usize) -> Option<&'a SolverResult> {
            self.0.get(i)
        }

        #[diplomat::attr(auto, iterable)]
        pub fn iter<'a>(&'a self) -> Box<SolverResultIterator<'a>> {
            Box::new(SolverResultIterator(self.0.iter()))
        }
    }

    /// Iterates over a `SolverResultList`.
    #[diplomat::opaque_mut]
    pub struct SolverResultIterator<'a>(std::slice::Iter<'a, SolverResult>);

    impl<'a> SolverResultIterator<'a> {
        #[diplomat::attr(auto, iterator)]
        pub fn next(&mut self) -> Option<&'a SolverResult> {
            self.0.next()
        }
    }

    /// Solves one frame at a time, with the caller managing the state.
    #[diplomat::opaque_mut]
    pub struct SingleFrameSolver(pub(crate) solver::SingleFrameSolver);

    impl SingleFrameSolver {
        /// Returns an error if `config` has smoothness priors (there is no
        /// previous frame) or root priors on a fixed-base tree.
        pub fn create(
            tree: &KinematicTree,
            config: &SolverConfig,
        ) -> Result<Box<SingleFrameSolver>, Box<Error>> {
            let solver = solver::SingleFrameSolver::new(&tree.0, &config.0)?;
            Ok(Box::new(SingleFrameSolver(solver)))
        }

        /// Refines `state` in place, starting from its current value, and
        /// returns the result. `with_grad` and `with_fk` request
        /// `SolverResult::jacobian`/`SolverResult::cholesky_l` and
        /// `SolverResult::keypoint_pos` respectively; each costs a little
        /// extra.
        pub fn solve(
            &mut self,
            state: &mut State,
            observations: &KeypointObservationList,
            with_grad: bool,
            with_fk: bool,
        ) -> Result<Box<SolverResult>, Box<Error>> {
            let result = self
                .0
                .solve(&mut state.0, &observations.0, with_grad, with_fk)?;
            Ok(Box::new(result.into()))
        }
    }

    impl From<solver::SolverResult> for SolverResult {
        fn from(result: solver::SolverResult) -> Self {
            SolverResult {
                state: State(result.state),
                keypoint_pos: result.keypoint_pos,
                jacobian: result.jacobian.as_ref().map(row_major),
                cholesky_l: result.cholesky_l.as_ref().map(|c| row_major(&c.l())),
            }
        }
    }
}

/// Converts a sequence solve's results.
pub(crate) fn result_list(
    results: Vec<quickik_core::solver::SolverResult>,
) -> Box<ffi::SolverResultList> {
    Box::new(ffi::SolverResultList(
        results.into_iter().map(Into::into).collect(),
    ))
}
