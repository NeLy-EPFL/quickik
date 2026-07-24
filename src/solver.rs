//! This module implements a Gauss-Newton solver for inverse kinematics.

use nalgebra::{DMatrix, DVector, Vector3};

use crate::body_plan::{KinematicTree, N_ROOT_DOFS};
use crate::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use crate::observation::{KeypointObservation, Mapper3Dto2D, NoMapper};
use crate::state::State;

/// Configuration for the inverse kinematics solver.
#[derive(Clone, Copy, Debug)]
pub struct SolverConfig<M: Mapper3Dto2D = NoMapper> {
    /// Fixed number of Gauss-Newton steps per solve.
    pub n_iterations: usize,
    /// Levenberg-Marquardt damping added to the normal equations' diagonal.
    /// This term is used only to improve numerical stability and should be set
    /// to a very small number (e.g. 1e-6).
    pub damping: f32,
    /// Tikhonov weight pulling every joint angle toward the neutral pose,
    /// multiplied together with each DOF's own [`Dof::weight_scaler`]. This
    /// regularization term improves robustness when keypoints are missing or
    /// noisy, but can also bias the solution away from the true pose.
    ///
    /// [`Dof::weight_scaler`]: crate::body_plan::Dof::weight_scaler
    pub weight: f32,
    /// Stop iterating early once an update step's largest root-position
    /// component drops below this value, *and* the largest angle update drops
    /// below [`angle_tolerance`](Self::angle_tolerance). In other words,
    /// `n_iterations` acts as a maximum cap rather than a fixed step count.
    /// This is useful for warm-started frames, which may converge much sooner.
    /// Set to 0 to disable early termination.
    ///
    /// [`Missing`]: crate::observation::KeypointObservation::Missing
    pub position_tolerance: f32,
    /// See [`position_tolerance`](Self::position_tolerance). Specified in
    /// radians.
    pub angle_tolerance: f32,
    /// Mapper used to project every [`Position2D`] observation. `None` if
    /// keypoint observations will be provided in 3D.
    ///
    /// [`Position2D`]: crate::observation::KeypointObservation::Position2D
    pub mapper: Option<M>,
}

impl<M: Mapper3Dto2D> Default for SolverConfig<M> {
    fn default() -> Self {
        SolverConfig {
            n_iterations: 10,
            damping: 1e-6,
            weight: 1e-3,
            position_tolerance: 1e-3,
            angle_tolerance: 1e-3,
            mapper: None,
        }
    }
}

/// The inverse kinematics solver.
///
/// Generic over the mapper `M` used to project 3D positions and Jacobians to
/// 2D for [`Position2D`] observations. Set to [`NoMapper`] if observations
/// are given in 3D (default). The mapper is fixed once upon construction, so
/// each solver can only accept one type of observation.
///
/// [`Position2D`]: crate::observation::KeypointObservation::Position2D
pub struct Solver<M: Mapper3Dto2D = NoMapper> {
    workspace: ForwardKinematicsWorkspace,
    neutral_joint_angles: Vec<f32>,
    /// Per-DOF [`Dof::weight_scaler`](crate::body_plan::Dof::weight_scaler),
    /// same indexing as `neutral_joint_angles`.
    dof_weight_scalers: Vec<f32>,
    /// Per-keypoint [`Joint::weight_scaler`](crate::body_plan::Joint::weight_scaler),
    /// one per joint/keypoint in tree order.
    joint_weight_scalers: Vec<f32>,
    jtj: DMatrix<f32>,
    jtr: DVector<f32>,
    /// Per-keypoint Jacobian buffer in compact form: shape is 3 x state_dim,
    /// but nonzero columns are moved to the left, and the rest is ignored.
    jacobian_buffer: DMatrix<f32>,
    /// Per-keypoint projected-2D-Jacobian scratch buffer (shape 2 x
    /// state_dim), same compact-column convention as `jacobian_buffer`. Only
    /// written/read for `Position2D` observations.
    jacobian_2d_buffer: DMatrix<f32>,
    pub config: SolverConfig<M>,
}

impl<M: Mapper3Dto2D> Solver<M> {
    pub fn new(kinematic_tree: &KinematicTree, config: SolverConfig<M>) -> Self {
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
            neutral_joint_angles,
            dof_weight_scalers,
            joint_weight_scalers,
            jtj: DMatrix::zeros(state_dim, state_dim),
            jtr: DVector::zeros(state_dim),
            jacobian_buffer: DMatrix::zeros(3, state_dim),
            jacobian_2d_buffer: DMatrix::zeros(2, state_dim),
            config,
        }
    }

    /// Runs `self.config.n_iterations` Gauss-Newton steps in place on
    /// `state`, given observations for all  keypoints (although the observation
    /// type may be [`Missing`] for some).
    ///
    /// [`Missing`]: crate::observation::KeypointObservation::Missing
    pub fn solve(&mut self, state: &mut State, observations: &[KeypointObservation]) {
        debug_assert_eq!(observations.len(), state.kinematic_tree.n_joints());

        let state_dim = state.state_dim();
        for _ in 0..self.config.n_iterations {
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
                    self.config.mapper.as_ref(),
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
                self.config.weight,
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
            //   zero (weakly- or entirely-unconstrained DOFs -- routine, and
            //   unrelated to coordinate units/scale: it happens whenever few
            //   keypoints constrain a DOF, and is exercised directly by any
            //   config with `weight: 0.0`), so damping doesn't vanish right
            //   when it's most needed to keep the Cholesky decomposition
            //   well-conditioned.
            for i in 0..state_dim {
                self.jtj[(i, i)] += self.config.damping * self.jtj[(i, i)].max(1.0);
            }

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
            let delta = match nalgebra::linalg::Cholesky::new(jtj_owned) {
                Some(chol) => {
                    let delta = chol.solve(&self.jtr);
                    self.jtj = chol.unpack();
                    delta
                }
                None => {
                    // Not positive-definite (numerically unstable): no
                    // update this iteration. This can happen when no keypoint
                    // is observed (even if some are observed, the root might
                    // be underconstrained and matches the targets exactly).
                    self.jtj = DMatrix::zeros(state_dim, state_dim);
                    DVector::zeros(state_dim)
                }
            };
            state.apply_delta(&delta);

            if self.has_converged(&delta) {
                break;
            }
        }
    }

    fn has_converged(&self, delta: &DVector<f32>) -> bool {
        // Positions: delta[0..3] is root position
        let max_abs_position_delta = delta
            .rows(0, 3)
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        // Angles: delta[3..6] is root rotation, the rest are DOF angles
        let max_abs_angle_delta = delta
            .rows(3, delta.len() - 3)
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        max_abs_position_delta <= self.config.position_tolerance
            && max_abs_angle_delta <= self.config.angle_tolerance
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_keypoint_residual<M: Mapper3Dto2D>(
    obs: &KeypointObservation,
    mapper: Option<&M>,
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
            let mapper = mapper
                .expect("Position2D observation given to a Solver constructed with mapper: None");
            // Same sparse accumulation as the Position3D case above: the
            // mapper writes its projected Jacobian into a view of the
            // preallocated `jacobian_2d_buffer` (no allocation), and jtj/jtr
            // are accumulated via direct dot products rather than a full
            // matrix multiply.
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
    for (i, ((&curr_angle, &neutral_angle), &dof_weight_scaler)) in (state.dof_angles)
        .iter()
        .zip(neutral_joint_angles)
        .zip(dof_weight_scalers)
        .enumerate()
    {
        let weight = weight * dof_weight_scaler;
        let state_idx = N_ROOT_DOFS + i;
        jtj[(state_idx, state_idx)] += weight; // only contributor is self
        jtr[state_idx] += weight * (neutral_angle - curr_angle);
    }
}
