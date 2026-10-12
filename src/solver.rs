//! Solver configuration, per-frame results, the Gauss-Newton core shared by all
//! solvers, and [`SingleFrameSolver`].

use std::sync::Arc;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dyn, SVector, Vector2, Vector3};

use crate::body_plan::KinematicTree;
use crate::error::{Result, bail};
use crate::forward::{ForwardKinematicsWorkspace, forward_kinematics_unchecked};
use crate::observation::{KeypointObservation, ObservationSequence, ObservedPosition};
use crate::priors::{
    DofNeutralPrior, DofSmoothnessPrior, Priors, RootSmoothnessPrior, RootUprightPrior,
};
use crate::state::State;

// =============================================================================
//  Configuration
// =============================================================================

/// Iteration settings (fixed at construction) plus priors (added with the
/// `add_*` methods, each at most once). No priors by default.
///
/// Copied into each solver on construction. Checked there too: root priors
/// on a fixed-base tree, and smoothness priors in [`SingleFrameSolver`] or
/// [`BatchedSolver`](crate::batched_solver::BatchedSolver), are errors.
#[derive(Clone, Debug)]
pub struct SolverConfig {
    n_iterations: usize,
    position_tolerance: f32,
    angle_tolerance: f32,
    damping: f32,
    pub(crate) priors: Priors,
}

impl Default for SolverConfig {
    /// `n_iterations = 10`, `position_tolerance = 1e-3`,
    /// `angle_tolerance = 1e-3`, `damping = 1e-6`, no priors.
    fn default() -> Self {
        Self::new(10, 1e-3, 1e-3, 1e-6)
    }
}

impl SolverConfig {
    /// - `n_iterations`: maximum number of Gauss-Newton steps per frame.
    /// - `position_tolerance`, `angle_tolerance`: stop early once a step's
    ///   largest root-position component is at most `position_tolerance` *and*
    ///   its largest angle component (root rotation and every DOF) is at most
    ///   `angle_tolerance` (radians). Set both to 0 to always run
    ///   `n_iterations` steps.
    /// - `damping`: Levenberg-Marquardt damping, relative to the normal
    ///   equations' diagonal (floored at 1). Only for numerical stability, so
    ///   keep it very small (e.g. 1e-6).
    pub fn new(
        n_iterations: usize,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    ) -> Self {
        Self {
            n_iterations,
            position_tolerance,
            angle_tolerance,
            damping,
            priors: Priors::default(),
        }
    }

    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }

    pub fn position_tolerance(&self) -> f32 {
        self.position_tolerance
    }

    pub fn angle_tolerance(&self) -> f32 {
        self.angle_tolerance
    }

    pub fn damping(&self) -> f32 {
        self.damping
    }

    /// Pulls each DOF toward its [`neutral`](crate::body_plan::Dof::neutral)
    /// value, with per-DOF weight `weight * Dof::weight_multiplier`. Improves
    /// robustness to missing or noisy keypoints, but biases the solution
    /// toward the neutral pose.
    pub fn add_dof_neutral_prior(&mut self, weight: f32) -> Result<()> {
        check_weight(weight)?;
        let prior = DofNeutralPrior { weight };
        set_once(&mut self.priors.dof_neutral, prior, "DOF neutral")
    }

    /// Pulls the root's local `body_up` axis toward the world-frame `world_up`
    /// (e.g. +z, or a tilted vector on a slope), leaving rotation about the up
    /// axis free. Both vectors are normalized.
    pub fn add_root_upright_prior(
        &mut self,
        weight: f32,
        body_up: Vector3<f32>,
        world_up: Vector3<f32>,
    ) -> Result<()> {
        check_weight(weight)?;
        let (Some(body_up), Some(world_up)) =
            (body_up.try_normalize(0.0), world_up.try_normalize(0.0))
        else {
            bail!("body_up and world_up must be nonzero");
        };
        let prior = RootUprightPrior {
            weight,
            body_up,
            world_up,
        };
        set_once(&mut self.priors.root_upright, prior, "root upright")
    }

    /// Pulls each DOF toward its value in the previous frame, with per-DOF
    /// weight `weight * Dof::weight_multiplier`. Inactive on a sequence's first
    /// frame.
    pub fn add_dof_smoothness_prior(&mut self, weight: f32) -> Result<()> {
        check_weight(weight)?;
        let prior = DofSmoothnessPrior { weight };
        set_once(&mut self.priors.dof_smoothness, prior, "DOF smoothness")
    }

    /// Pulls the root position and rotation toward the previous frame's, with
    /// separate weights since their units differ (length vs radians). Inactive
    /// on a sequence's first frame.
    pub fn add_root_smoothness_prior(
        &mut self,
        position_weight: f32,
        rotation_weight: f32,
    ) -> Result<()> {
        check_weight(position_weight)?;
        check_weight(rotation_weight)?;
        let prior = RootSmoothnessPrior {
            position_weight,
            rotation_weight,
        };
        set_once(&mut self.priors.root_smoothness, prior, "root smoothness")
    }
}

fn check_weight(weight: f32) -> Result<()> {
    if !(weight.is_finite() && weight >= 0.0) {
        bail!("prior weights must be finite and non-negative, got {weight}");
    }
    Ok(())
}

fn set_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<()> {
    if slot.is_some() {
        bail!("a {name} prior was already added");
    }
    *slot = Some(value);
    Ok(())
}

// =============================================================================
//  Results
// =============================================================================

/// One frame's converged pose and, optionally, its linearization.
#[derive(Clone, Debug)]
pub struct SolverResult {
    pub state: State,
    /// World-frame keypoint positions at `state`, in joint order. `Some` if
    /// and only if the solve had `with_fk: true`.
    pub keypoint_pos: Option<Vec<Vector3<f32>>>,
    /// Residual Jacobian `(n_rows, state_dim)`: one row block per observation,
    /// in the frame's input order, 3 rows for a 3D observation and 2 for a 2D
    /// one (already mapped through its projection). Weight-0 observations get
    /// rows too. Linearized at the pose just before the last iteration's
    /// update, which is small at convergence. `Some` if and only if the solve
    /// had `with_grad: true`.
    pub jacobian: Option<DMatrix<f32>>,
    /// Cholesky factorization of the normal-equations matrix (priors and
    /// damping included) at the same linearization as `jacobian`. `Some` if
    /// and only if the solve had `with_grad: true` and that matrix was
    /// positive-definite.
    pub cholesky_l: Option<Cholesky<f32, Dyn>>,
}

// =============================================================================
//  Gauss-Newton core
// =============================================================================

/// The Gauss-Newton solver shared by all public solvers: one per thread, since
/// it owns its scratch buffers.
#[derive(Clone)]
pub(crate) struct SolverCore {
    tree: Arc<KinematicTree>,
    config: SolverConfig,
    fk: ForwardKinematicsWorkspace,
    jtj: DMatrix<f32>,
    jtr: DVector<f32>,
    delta: DVector<f32>,
    /// Scratch for one 2D observation's projected compact Jacobian.
    jacobian_2d: Vec<Vector2<f32>>,
}

impl SolverCore {
    pub fn new(tree: &Arc<KinematicTree>, config: &SolverConfig) -> Result<Self> {
        if tree.fixed_base() && config.priors.has_root() {
            bail!("root priors can't be used with a fixed-base tree");
        }
        let state_dim = tree.state_dim();
        Ok(Self {
            tree: Arc::clone(tree),
            config: config.clone(),
            fk: ForwardKinematicsWorkspace::new(tree),
            jtj: DMatrix::zeros(state_dim, state_dim),
            jtr: DVector::zeros(state_dim),
            delta: DVector::zeros(state_dim),
            jacobian_2d: vec![Vector2::zeros(); tree.max_n_upstream()],
        })
    }

    pub fn tree(&self) -> &Arc<KinematicTree> {
        &self.tree
    }

    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    pub fn check_state(&self, state: &State) -> Result<()> {
        if state.dof_values.len() != self.tree.n_dofs() {
            bail!(
                "state has {} DOF values, but the solver's tree has {} DOFs",
                state.dof_values.len(),
                self.tree.n_dofs()
            );
        }
        Ok(())
    }

    pub fn check_observations(&self, observations: &[KeypointObservation]) -> Result<()> {
        let max_joint_idx = observations.iter().map(|o| o.joint_idx).max();
        if let Some(idx) = max_joint_idx
            && idx >= self.tree.n_joints()
        {
            bail!(
                "observation of joint {idx}, but the tree has {} joints",
                self.tree.n_joints()
            );
        }
        Ok(())
    }

    pub fn check_sequence(&self, sequence: &ObservationSequence) -> Result<()> {
        for frame in sequence.frames() {
            self.check_observations(frame)?;
        }
        Ok(())
    }

    /// Runs up to `n_iterations` Gauss-Newton steps on `state` in place.
    /// Arguments must already be checked against the tree.
    pub fn solve_frame(
        &mut self,
        state: &mut State,
        prev_state: Option<&State>,
        observations: &[KeypointObservation],
        with_grad: bool,
        with_fk: bool,
    ) -> SolverResult {
        let n_iterations = self.config.n_iterations;
        let mut jacobian = None;
        let mut cholesky_l = None;
        for iteration_idx in 0..n_iterations {
            forward_kinematics_unchecked(&mut self.fk, state);
            self.build_normal_equations(state, prev_state, observations);
            let is_positive_definite = self.solve_normal_equations();

            let is_converged = self.has_converged();
            if with_grad && (is_converged || iteration_idx == n_iterations - 1) {
                jacobian = Some(self.residual_jacobian(observations));
                // After an in-place decomposition, `jtj` holds the factor `L`
                cholesky_l = is_positive_definite.then(|| Cholesky::pack_dirty(self.jtj.clone()));
            }

            state.apply_delta(&self.delta);
            if is_converged {
                break;
            }
        }
        // Only reachable with `n_iterations == 0`: linearize at the input pose.
        if with_grad && jacobian.is_none() {
            forward_kinematics_unchecked(&mut self.fk, state);
            jacobian = Some(self.residual_jacobian(observations));
        }
        // `keypoint_pos` must match the returned (updated) state, which the
        // last in-loop FK call predates.
        if with_fk {
            forward_kinematics_unchecked(&mut self.fk, state);
        }
        SolverResult {
            state: state.clone(),
            keypoint_pos: with_fk.then(|| self.fk.keypoint_positions().to_vec()),
            jacobian,
            cholesky_l,
        }
    }

    fn build_normal_equations(
        &mut self,
        state: &State,
        prev_state: Option<&State>,
        observations: &[KeypointObservation],
    ) {
        // `fill` on the contiguous slice is much faster than the generic
        // `Matrix::fill`.
        self.jtj.as_mut_slice().fill(0.0);
        self.jtr.as_mut_slice().fill(0.0);

        let tree = &*self.tree;
        for obs in observations {
            let j = obs.joint_idx;
            let weight = obs.weight * tree.joint_weight_multipliers()[j];
            if weight == 0.0 {
                continue;
            }
            let state_indices = tree.upstream_state_indices(j);
            let jacobian_3d = self.fk.keypoint_jacobian(j);
            let keypoint_pos = self.fk.keypoint_positions()[j];
            match &obs.position {
                ObservedPosition::Position3D(obs_pos) => {
                    let residual = obs_pos - keypoint_pos;
                    let (jtj, jtr) = (&mut self.jtj, &mut self.jtr);
                    accumulate(jacobian_3d, &residual, weight, state_indices, jtj, jtr);
                }
                ObservedPosition::Position2D(obs_pos, projection) => {
                    let (pos_2d, d_pos2d_d_pos3d) = projection.project(&keypoint_pos);
                    let jacobian_2d = &mut self.jacobian_2d[..jacobian_3d.len()];
                    for (column_2d, column_3d) in jacobian_2d.iter_mut().zip(jacobian_3d) {
                        *column_2d = d_pos2d_d_pos3d * column_3d;
                    }
                    let residual = obs_pos - pos_2d;
                    let (jtj, jtr) = (&mut self.jtj, &mut self.jtr);
                    accumulate(jacobian_2d, &residual, weight, state_indices, jtj, jtr);
                }
            }
        }

        let priors = &self.config.priors;
        priors.accumulate(tree, state, prev_state, &mut self.jtj, &mut self.jtr);

        // Relative Levenberg-Marquardt damping. `max(1.0)` both scales damping
        // up when the diagonal is large (e.g. pixel units), so it isn't
        // swamped, and floors it when a DOF is weakly or not at all
        // constrained, so it doesn't vanish when it's needed most.
        let damping = self.config.damping;
        for i in 0..self.jtj.nrows() {
            self.jtj[(i, i)] += damping * self.jtj[(i, i)].max(1.0);
        }
    }

    /// Solves the normal equations into `self.delta` via an in-place Cholesky
    /// decomposition, leaving the factor `L` in `self.jtj`. Returns whether
    /// `jtj` was positive-definite; if not, `self.delta` is zero (no update).
    fn solve_normal_equations(&mut self) -> bool {
        // `Cholesky::new` takes the matrix by value, so move `jtj` out and put
        // the factor back afterwards (both without allocating).
        let jtj = std::mem::replace(&mut self.jtj, DMatrix::zeros(0, 0));
        match Cholesky::new(jtj) {
            Some(cholesky) => {
                self.delta.copy_from(&self.jtr);
                cholesky.solve_mut(&mut self.delta);
                self.jtj = cholesky.unpack();
                true
            }
            None => {
                // Can happen e.g. when nothing is observed and no prior
                // constrains the state.
                let state_dim = self.tree.state_dim();
                self.jtj = DMatrix::zeros(state_dim, state_dim);
                self.delta.as_mut_slice().fill(0.0);
                false
            }
        }
    }

    fn has_converged(&self) -> bool {
        // `delta[..n_position_dofs]` is the root position (empty for a fixed
        // base); the rest is root rotation and every DOF.
        let n_position_dofs = self.tree.n_root_dofs().min(3);
        let max_abs = |values: &[f32]| values.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let (position, angles) = self.delta.as_slice().split_at(n_position_dofs);
        max_abs(position) <= self.config.position_tolerance
            && max_abs(angles) <= self.config.angle_tolerance
    }

    /// The dense residual Jacobian at the current FK evaluation: one row block
    /// per observation, in input order, including weight-0 ones.
    fn residual_jacobian(&self, observations: &[KeypointObservation]) -> DMatrix<f32> {
        let n_rows = observations.iter().map(|o| o.n_rows()).sum();
        let mut jacobian = DMatrix::zeros(n_rows, self.tree.state_dim());
        let mut row = 0;
        for obs in observations {
            let j = obs.joint_idx;
            let state_indices = self.tree.upstream_state_indices(j);
            let columns = self.fk.keypoint_jacobian(j).iter().zip(state_indices);
            match &obs.position {
                ObservedPosition::Position3D(_) => {
                    for (column, &s) in columns {
                        for axis in 0..3 {
                            jacobian[(row + axis, s)] = column[axis];
                        }
                    }
                }
                ObservedPosition::Position2D(_, projection) => {
                    let keypoint_pos = self.fk.keypoint_positions()[j];
                    let (_, d_pos2d_d_pos3d) = projection.project(&keypoint_pos);
                    for (column, &s) in columns {
                        let column_2d = d_pos2d_d_pos3d * column;
                        for axis in 0..2 {
                            jacobian[(row + axis, s)] = column_2d[axis];
                        }
                    }
                }
            }
            row += obs.n_rows();
        }
        jacobian
    }
}

/// Adds one observation's `w J^T J` (lower triangle only) and `w J^T r` into
/// the normal equations. `columns` is the observation's compact Jacobian, one
/// column per (ascending) state index in `state_indices`.
#[inline]
fn accumulate<const R: usize>(
    columns: &[SVector<f32, R>],
    residual: &SVector<f32, R>,
    weight: f32,
    state_indices: &[usize],
    jtj: &mut DMatrix<f32>,
    jtr: &mut DVector<f32>,
) {
    let dim = jtj.nrows();
    let jtj = jtj.as_mut_slice();
    for (b, (column_b, &gb)) in columns.iter().zip(state_indices).enumerate() {
        let weighted_b = column_b * weight;
        jtr[gb] += weighted_b.dot(residual);
        // Column `gb` of the column-major `jtj`, rows `ga >= gb` since the
        // state indices are ascending.
        let jtj_column = &mut jtj[gb * dim..(gb + 1) * dim];
        for (column_a, &ga) in columns[b..].iter().zip(&state_indices[b..]) {
            jtj_column[ga] += weighted_b.dot(column_a);
        }
    }
}

// =============================================================================
//  Single-frame solver
// =============================================================================

/// Solves one frame at a time, with the caller managing the state.
pub struct SingleFrameSolver {
    core: SolverCore,
}

impl SingleFrameSolver {
    /// Returns an error if `config` has smoothness priors (there is no
    /// previous frame) or root priors on a fixed-base tree.
    pub fn new(kinematic_tree: &Arc<KinematicTree>, config: &SolverConfig) -> Result<Self> {
        if config.priors.has_temporal() {
            bail!("SingleFrameSolver doesn't support smoothness priors (no previous frame)");
        }
        Ok(Self {
            core: SolverCore::new(kinematic_tree, config)?,
        })
    }

    pub fn kinematic_tree(&self) -> &Arc<KinematicTree> {
        self.core.tree()
    }

    pub fn config(&self) -> &SolverConfig {
        self.core.config()
    }

    /// Refines `state` in place, starting from its current value, and returns
    /// the result. `with_grad` and `with_fk` request
    /// [`SolverResult::jacobian`]/[`SolverResult::cholesky_l`] and
    /// [`SolverResult::keypoint_pos`] respectively; each costs a little extra.
    pub fn solve(
        &mut self,
        state: &mut State,
        observations: &[KeypointObservation],
        with_grad: bool,
        with_fk: bool,
    ) -> Result<SolverResult> {
        self.core.check_state(state)?;
        self.core.check_observations(observations)?;
        Ok(self
            .core
            .solve_frame(state, None, observations, with_grad, with_fk))
    }
}
