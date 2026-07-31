//! This module implements a Gauss-Newton solver for inverse kinematics.

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, Dyn, Vector3};

use crate::body_plan::KinematicTree;
use crate::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use crate::observation::{KeypointObservation, Mapper3Dto2D, NoMapper};
use crate::state::State;

/// The converged pose and (optionally) linearization from one [`Solver::solve`]
/// call, or one item of a [`SequenceSolver`](crate::sequential_solver::SequenceSolver)
/// sequence.
pub struct SolverResult {
    /// The converged pose (`dof_angles`, `root_pos`, `root_rot`).
    pub state: State,
    /// World-space keypoint positions, in `KinematicTree`'s joint order.
    /// `Some` if and only if `solve` was called with `with_fk: true`.
    pub keypoint_pos: Option<Vec<Vector3<f32>>>,
    /// The keypoint-position Jacobian at (approximately) the converged pose:
    /// see [`solve`](Solver::solve)'s docs for exactly which pose. `Some` if
    /// and only if `solve` was called with `with_grad: true`.
    pub jacobian: Option<DMatrix<f32>>,
    /// Cholesky factorization of the normal-equations matrix (`jtj`, i.e.
    /// `jacobian^T @ weights @ jacobian` plus damping and the neutral-pose
    /// prior) at the same linearization as `jacobian`. `Some` if and only if
    /// `with_grad` was `true` *and* that linearization's normal equations were
    /// positive-definite (gradients can't be computed from this solve
    /// otherwise).
    pub cholesky_l: Option<Cholesky<f32, Dyn>>,
}

/// The inverse kinematics solver.
///
/// Generic over the mapper `M` used to project 3D positions and Jacobians to
/// 2D for [`Position2D`] observations. Set to [`NoMapper`] if observations
/// are given in 3D (default). `mapper` is fixed once upon construction (no
/// setter), so each solver can only accept one type of observation.
///
/// The other configuration fields (`n_iterations`, `neutral_weight`,
/// `position_tolerance`, `angle_tolerance`, `damping`) are plain public
/// fields, freely retunable between calls.
///
/// [`Position2D`]: crate::observation::KeypointObservation::Position2D
pub struct Solver<M: Mapper3Dto2D = NoMapper> {
    workspace: ForwardKinematicsWorkspace,
    /// Cached from the kinematic tree at construction time: `0` for a
    /// fixed-base tree, [`N_ROOT_DOFS`](crate::body_plan::N_ROOT_DOFS)
    /// otherwise.
    n_root_dofs: usize,
    neutral_joint_angles: Vec<f32>,
    /// Per-DOF [`Dof::weight_scaler`](crate::body_plan::Dof::weight_scaler),
    /// same indexing as `neutral_joint_angles`.
    dof_weight_scalers: Vec<f32>,
    /// Per-keypoint [`Joint::weight_scaler`](crate::body_plan::Joint::weight_scaler),
    /// one per joint/keypoint in tree order.
    joint_weight_scalers: Vec<f32>,
    jtj: DMatrix<f32>,
    jtr: DVector<f32>,
    /// Gauss-Newton update step, reused across iterations and solved into in
    /// place by `Cholesky::solve_mut` to avoid a per-iteration allocation.
    delta: DVector<f32>,
    /// Per-keypoint Jacobian buffer in compact form: shape is 3 x state_dim,
    /// but nonzero columns are moved to the left, and the rest is ignored.
    jacobian_buffer: DMatrix<f32>,
    /// Per-keypoint projected-2D-Jacobian scratch buffer (shape 2 x
    /// state_dim), same compact-column convention as `jacobian_buffer`. Only
    /// written/read for `Position2D` observations.
    jacobian_2d_buffer: DMatrix<f32>,
    /// Snapshot of `workspace.kpt_jacobian` from the last iteration run with
    /// `with_grad: true`, reused (via `copy_from`) across calls to avoid a
    /// per-call allocation; only meaningful right after such a call, and only
    /// actually read when building that call's [`SolverResult::jacobian`].
    last_jacobian: DMatrix<f32>,
    /// Raw Cholesky factor L (lower-triangular, `jtj = L L^T`) from the last
    /// iteration run with `with_grad: true`, updated in place via `copy_from`
    /// (no allocation). Stored as a plain matrix rather than a [`Cholesky`]
    /// because `Cholesky` has no in-place update API of its own; wrapped into
    /// one on demand (one allocation) when building [`SolverResult::cholesky_l`].
    last_cholesky_l: DMatrix<f32>,
    /// Whether `last_cholesky_l` is from a positive-definite iteration.
    last_cholesky_valid: bool,
    /// Fixed at construction; see [`mapper`](Self::mapper) for why there's no
    /// setter.
    mapper: M,
    /// Fixed number of Gauss-Newton steps per solve.
    pub n_iterations: usize,
    /// Weight of Tikhonov regularization term pulling every joint angle toward
    /// the neutral pose, multiplied together with each DOF's own
    /// [`Dof::weight_scaler`]. This regularization term improves robustness
    /// when keypoints are missing or noisy, but can also bias the solution away
    /// from the true pose.
    ///
    /// [`Dof::weight_scaler`]: crate::body_plan::Dof::weight_scaler
    pub neutral_weight: f32,
    /// Stop iterating early once an update step's largest root-position
    /// component drops below this value, *and* the largest angle update drops
    /// below [`angle_tolerance`](Self::angle_tolerance). In other words,
    /// `n_iterations` acts as a maximum cap rather than a fixed step count.
    /// This is useful for warm-started frames, which may converge much sooner.
    /// Set to 0 to disable early termination.
    pub position_tolerance: f32,
    /// See [`position_tolerance`](Self::position_tolerance). Specified in
    /// radians.
    pub angle_tolerance: f32,
    /// Levenberg-Marquardt damping added to the normal equations' diagonal.
    /// This term is used only to improve numerical stability and should be set
    /// to a very small number (e.g. 1e-6).
    pub damping: f32,
}

impl<M: Mapper3Dto2D> Solver<M> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kinematic_tree: &KinematicTree,
        mapper: M,
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    ) -> Self {
        // Populate neutral joint angles, per-DOF weight scalers, and
        // per-joint weight scalers
        let mut neutral_joint_angles = vec![0.0; kinematic_tree.n_dofs()];
        let mut dof_weight_scalers = vec![1.0; kinematic_tree.n_dofs()];
        let mut joint_weight_scalers = Vec::with_capacity(kinematic_tree.n_joints());
        for joint in &kinematic_tree.joints {
            joint_weight_scalers.push(joint.weight_scaler);
            for (i, dof) in joint.dofs.iter().enumerate() {
                neutral_joint_angles[joint.dof_offset + i] = dof.neutral;
                dof_weight_scalers[joint.dof_offset + i] = dof.weight_scaler;
            }
        }

        // Create workspace and preallocate buffers for normal equations
        let state_dim = kinematic_tree.state_dim();
        Self {
            workspace: ForwardKinematicsWorkspace::new(kinematic_tree),
            n_root_dofs: kinematic_tree.n_root_dofs(),
            neutral_joint_angles,
            dof_weight_scalers,
            joint_weight_scalers,
            jtj: DMatrix::zeros(state_dim, state_dim),
            jtr: DVector::zeros(state_dim),
            delta: DVector::zeros(state_dim),
            jacobian_buffer: DMatrix::zeros(3, state_dim),
            jacobian_2d_buffer: DMatrix::zeros(2, state_dim),
            last_jacobian: DMatrix::zeros(3 * kinematic_tree.n_joints(), state_dim),
            last_cholesky_l: DMatrix::zeros(state_dim, state_dim),
            last_cholesky_valid: false,
            mapper,
            n_iterations,
            neutral_weight,
            position_tolerance,
            angle_tolerance,
            damping,
        }
    }

    /// Fixed at construction; there's no setter, mirroring `M` being fixed at
    /// compile time for this `Solver<M>`.
    pub fn mapper(&self) -> M {
        self.mapper
    }

    /// Runs up to `self.n_iterations` Gauss-Newton steps in place on `state`,
    /// given observations for all keypoints (although the observation type
    /// may be [`Missing`] for some), and reports the converged pose.
    ///
    /// `with_grad`/`with_fk` gate [`SolverResult::jacobian`]/
    /// [`SolverResult::cholesky_l`] and [`SolverResult::keypoint_pos`]
    /// respectively; each costs a little extra work, so only request what
    /// you'll use. The Jacobian/Cholesky factor are linearized at the pose
    /// from just *before* the last iteration's own update step (since that
    /// update is small at convergence, a close approximation of the
    /// converged pose); `keypoint_pos` is always exactly at the converged
    /// pose, one extra forward-kinematics evaluation after the loop.
    ///
    /// [`Missing`]: crate::observation::KeypointObservation::Missing
    pub fn solve(
        &mut self,
        state: &mut State,
        observations: &[KeypointObservation],
        with_grad: bool,
        with_fk: bool,
    ) -> SolverResult {
        self.solve_impl(state, observations, with_grad, with_fk);
        SolverResult {
            state: state.clone(),
            keypoint_pos: with_fk.then(|| self.workspace.kpt_positions.clone()),
            jacobian: with_grad.then(|| self.last_jacobian.clone()),
            cholesky_l: (with_grad && self.last_cholesky_valid)
                .then(|| Cholesky::pack_dirty(self.last_cholesky_l.clone())),
        }
    }

    fn solve_impl(
        &mut self,
        state: &mut State,
        observations: &[KeypointObservation],
        with_grad: bool,
        with_fk: bool,
    ) {
        assert_eq!(
            observations.len(),
            state.kinematic_tree.n_joints(),
            "observations.len() must equal kinematic_tree.n_joints()"
        );

        let state_dim = state.state_dim();
        for iteration_idx in 0..self.n_iterations {
            evaluate_fwdkin(&mut self.workspace, state);

            // See the matching comment in forward.rs: `Matrix::fill` is ~60x
            // slower than filling the underlying contiguous storage directly.
            self.jtj.as_mut_slice().fill(0.0);
            self.jtr.as_mut_slice().fill(0.0);
            for (k, obs) in observations.iter().enumerate() {
                if matches!(obs, KeypointObservation::Missing) {
                    continue;
                }
                let relevant_idxs = &self.workspace.relevant_dof_idxs_by_joint[k];
                // Gather this keypoint's nonzero Jacobian columns (root's
                // N_ROOT_DOFS plus its own ancestor DOFs). Everywhere else is 0.
                for (col, &state_idx) in relevant_idxs.iter().enumerate() {
                    for row in 0..3 {
                        self.jacobian_buffer[(row, col)] =
                            self.workspace.kpt_jacobian[(3 * k + row, state_idx)];
                    }
                }
                accumulate_keypoint_residual(
                    obs,
                    &self.mapper,
                    &self.workspace.kpt_positions[k],
                    &self.jacobian_buffer,
                    &mut self.jacobian_2d_buffer,
                    self.joint_weight_scalers[k],
                    relevant_idxs,
                    &mut self.jtj,
                    &mut self.jtr,
                );
            }

            accumulate_neutral_pose_prior(
                state,
                &self.neutral_joint_angles,
                &self.dof_weight_scalers,
                self.neutral_weight,
                &mut self.jtj,
                &mut self.jtr,
            );

            // Levenberg-Marquardt-style relative damping to improve numerical
            // stability. `max(1.0)` does two independent jobs, not just one:
            // - Scales damping up when J^T J's diagonal is large (e.g.
            //   coordinates in real pixel units, which can run into the
            //   thousands), so a tiny fixed `damping` isn't swamped and left
            //   numerically meaningless.
            // - Floors damping when a DOF's diagonal entry is small or near
            //   zero (weakly- or entirely-unconstrained DOFs, which is routine
            //   and unrelated to coordinate units/scale: it happens whenever few
            //   keypoints constrain a DOF, and is exercised directly by any
            //   config with `neutral_weight: 0.0`), so damping doesn't vanish right
            //   when it's most needed to keep the Cholesky decomposition
            //   well-conditioned.
            for i in 0..state_dim {
                self.jtj[(i, i)] += self.damping * self.jtj[(i, i)].max(1.0);
            }

            let chol_valid = self.solve_normal_equations(state_dim);

            // Whether this is the last iteration `solve_impl` will run: either
            // it's the last one `n_iterations` allows, or `is_converged` is
            // about to `break` the loop below anyway. Deferring the J/L
            // snapshot until this is known (rather than re-snapshotting on
            // every iteration and letting all but the last get overwritten)
            // means it only runs once per call, regardless of how many
            // iterations that call takes.
            let is_converged = self.has_converged(&self.delta);
            let is_last_iteration = is_converged || iteration_idx == self.n_iterations - 1;
            if with_grad && is_last_iteration {
                self.last_jacobian.copy_from(&self.workspace.kpt_jacobian);
                self.last_cholesky_valid = chol_valid;
                if chol_valid {
                    self.last_cholesky_l.copy_from(&self.jtj);
                }
            }

            // Apply state update
            state.apply_delta(&self.delta);
            if is_converged {
                break;
            }
        }

        // `evaluate_fwdkin` above runs at the *start* of each iteration
        // (before that iteration's `state.apply_delta`), so after the loop,
        // `self.workspace.kpt_positions` reflects the pose *before* the last
        // update was applied, one step stale relative to the `state`
        // actually returned to the caller. Run it once more, unconditionally
        // (including when `n_iterations == 0`), so `keypoint_pos` matches the
        // returned `state` exactly. This doesn't touch `last_jacobian`/
        // `last_cholesky_l`, which are already snapshotted above, before this
        // extra call. Skipped when `with_fk` isn't requested, since it's
        // otherwise pure waste (see this crate's own measurements: ~3.5-10%
        // of a solve() call, depending on how many iterations actually run).
        if with_fk {
            evaluate_fwdkin(&mut self.workspace, state);
        }
    }

    /// Solves the current `jtj`/`jtr` normal equations into `self.delta` via
    /// Cholesky decomposition. Returns whether `jtj` was positive-definite;
    /// if not, `self.delta` is set to all zeros (no update this iteration --
    /// this can happen when no keypoint is observed, or when the root is
    /// underconstrained and already matches the targets exactly).
    fn solve_normal_equations(&mut self, state_dim: usize) -> bool {
        // The Cholesky decomposer requires owning the matrix by value
        // (because it runs memory-optimized in-place operations). However,
        // self.jtj is passed via self which is given by mutable reference,
        // so Rust doesn't allow us to move it out of self.
        // Solution: Replace self.jtj with an empty placeholder matrix, move
        // the real jtj matrix into the Cholesky decomposer, and then move
        // the result back to self.jtj.
        // Because Cholesky does in-place math, the owned "jtj" matrix
        // contains garbage value after decomposition. But this is fine
        // because jtj is zeroed at the start of each solver iteration.
        let jtj_owned = std::mem::replace(&mut self.jtj, DMatrix::zeros(0, 0));
        match Cholesky::new(jtj_owned) {
            Some(chol) => {
                self.delta.copy_from(&self.jtr);
                chol.solve_mut(&mut self.delta);
                // `unpack` doesn't allocate: it just hands back the same
                // owned matrix (moved into `Cholesky::new` above) with its
                // upper triangle zeroed.
                self.jtj = chol.unpack();
                true
            }
            None => {
                // Not positive-definite (numerically unstable): no update
                // this iteration. This can happen when no keypoint is
                // observed (even if some are observed, the root might be
                // underconstrained and matches the targets exactly).
                self.jtj = DMatrix::zeros(state_dim, state_dim);
                self.delta.as_mut_slice().fill(0.0);
                false
            }
        }
    }

    fn has_converged(&self, delta: &DVector<f32>) -> bool {
        // Positions: delta[0..n_root_position_dofs] is root position (empty
        // for a fixed-base tree, n_root_dofs == 0, since it has no root
        // position state at all).
        let n_root_position_dofs = self.n_root_dofs.min(3);
        let max_abs_position_delta = delta
            .rows(0, n_root_position_dofs)
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        // Angles: the rest, root rotation (if any) plus every DOF angle.
        let max_abs_angle_delta = delta
            .rows(n_root_position_dofs, delta.len() - n_root_position_dofs)
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        max_abs_position_delta <= self.position_tolerance
            && max_abs_angle_delta <= self.angle_tolerance
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_keypoint_residual<M: Mapper3Dto2D>(
    obs: &KeypointObservation,
    mapper: &M,
    fwdkin_pos3d: &Vector3<f32>,
    jacobian_3d: &DMatrix<f32>,
    jacobian_2d_buffer: &mut DMatrix<f32>,
    joint_weight_scaler: f32,
    relevant_idxs: &[usize],
    jtj: &mut DMatrix<f32>,
    jtr: &mut DVector<f32>,
) {
    // Only this keypoint's own n_relevant_dofs nonzero columns of jacobian_3d
    // are read. The result is scattered into `jtj`/`jtr` at the DOFs' global
    // state indices.
    let n_relevant_dofs = relevant_idxs.len();
    match *obs {
        KeypointObservation::Missing => {}
        KeypointObservation::Position3D { obs_pos, weight } => {
            let weight = weight * joint_weight_scaler;
            let residual = obs_pos - fwdkin_pos3d;
            let jacobian_3d_view = jacobian_3d.columns(0, n_relevant_dofs);
            // i/j = index in compact views; gi/gj = global state indices
            for (i, &gi) in relevant_idxs.iter().enumerate() {
                let col_i = jacobian_3d_view.column(i);
                for (j, &gj) in relevant_idxs.iter().enumerate() {
                    jtj[(gi, gj)] += col_i.dot(&jacobian_3d_view.column(j)) * weight;
                }
                jtr[gi] += col_i.dot(&residual) * weight;
            }
        }
        KeypointObservation::Position2D { obs_pos, weight } => {
            let weight = weight * joint_weight_scaler;
            // Same sparse accumulation as the Position3D case above: the
            // mapper writes its projected Jacobian into a view of the
            // preallocated `jacobian_2d_buffer` (no allocation), and jtj/jtr
            // are accumulated via direct dot products rather than a full
            // matrix multiply. `mapper` (e.g. `NoMapper`) panics on its own
            // if this `Solver` wasn't actually constructed with a real
            // mapper; see `Mapper3Dto2D`'s docs.
            let jacobian_3d_view = jacobian_3d.columns(0, n_relevant_dofs);
            let mut jacobian_2d_view = jacobian_2d_buffer.columns_mut(0, n_relevant_dofs);
            let fwdkin_pos2d =
                mapper.project_3d_to_2d(fwdkin_pos3d, &jacobian_3d_view, &mut jacobian_2d_view);
            let residual = obs_pos - fwdkin_pos2d;
            for (i, &gi) in relevant_idxs.iter().enumerate() {
                let col_i = jacobian_2d_view.column(i);
                for (j, &gj) in relevant_idxs.iter().enumerate() {
                    jtj[(gi, gj)] += col_i.dot(&jacobian_2d_view.column(j)) * weight;
                }
                jtr[gi] += col_i.dot(&residual) * weight;
            }
        }
    }
}

fn accumulate_neutral_pose_prior(
    state: &State,
    neutral_joint_angles: &[f32],
    dof_weight_scalers: &[f32],
    weight: f32,
    jtj: &mut DMatrix<f32>,
    jtr: &mut DVector<f32>,
) {
    if weight == 0.0 {
        return;
    }
    let n_root_dofs = state.kinematic_tree.n_root_dofs();
    for (i, ((&curr_angle, &neutral_angle), &dof_weight_scaler)) in (state.dof_angles)
        .iter()
        .zip(neutral_joint_angles)
        .zip(dof_weight_scalers)
        .enumerate()
    {
        let weight = weight * dof_weight_scaler;
        let state_idx = n_root_dofs + i;
        jtj[(state_idx, state_idx)] += weight; // only contributor is self
        jtr[state_idx] += weight * (neutral_angle - curr_angle);
    }
}
