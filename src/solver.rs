//! This module implements a Gauss-Newton solver for inverse kinematics.

use nalgebra::{DMatrix, DVector, Vector3};

use crate::body_plan::{KinematicTree, N_ROOT_DOFS};
use crate::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use crate::observation::KeypointObservation;
use crate::state::State;

/// Configuration for the inverse kinematics solver.
#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    /// Fixed number of Gauss-Newton steps per solve.
    pub n_iterations: usize,
    /// Levenberg-Marquardt damping added to the normal equations' diagonal.
    /// This term is used only to improve numerical stability and should be set
    /// to a very small number (e.g. 1e-6).
    pub damping: f32,
    /// Tikhonov weight pulling every joint angle toward the neutral pose.
    /// This regularization term improves robustness when keypoints are missing
    /// or noisy, but can also bias the solution away from the true pose.
    pub neutral_pose_weight: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            n_iterations: 10,
            damping: 1e-6,
            neutral_pose_weight: 1e-3,
        }
    }
}

/// The inverse kinematics solver.
pub struct Solver {
    workspace: ForwardKinematicsWorkspace,
    neutral_joint_angles: Vec<f32>,
    jtj: DMatrix<f32>,
    jtr: DVector<f32>,
}

impl Solver {
    pub fn new(kinematic_tree: &KinematicTree) -> Self {
        // Populate neutral joint angles
        let mut neutral_joint_angles = vec![0.0; kinematic_tree.n_dofs()];
        for joint in &kinematic_tree.joints {
            for (i, dof) in joint.dofs.iter().enumerate() {
                neutral_joint_angles[joint.dof_offset + i] = dof.neutral_angle;
            }
        }

        // Create workspace and preallocate buffers for normal equations
        let state_dim = kinematic_tree.state_dim();
        Self {
            workspace: ForwardKinematicsWorkspace::new(kinematic_tree),
            neutral_joint_angles,
            jtj: DMatrix::zeros(state_dim, state_dim),
            jtr: DVector::zeros(state_dim),
        }
    }

    /// Runs `config.n_iterations` Gauss-Newton steps in place on `state`,
    /// given one observation per keypoint (`observations.len() ==
    /// state.kinematic_tree.n_joints()`, in `state.kinematic_tree.joints` order). Warm
    /// start by reusing the same `state` across frames rather than
    /// constructing a new one -- and reuse this same `Solver` too, since it
    /// must have been built from that `state`'s own bodyplan.
    pub fn solve(
        &mut self,
        state: &mut State,
        observations: &[KeypointObservation],
        config: &SolverConfig,
    ) {
        debug_assert_eq!(observations.len(), state.kinematic_tree.n_joints());

        let state_dim = state.state_dim();
        for _ in 0..config.n_iterations {
            evaluate_fwdkin(&mut self.workspace, state);

            self.jtj.fill(0.0);
            self.jtr.fill(0.0);
            for (k, obs) in observations.iter().enumerate() {
                if matches!(obs, KeypointObservation::Missing) {
                    continue;
                }
                let jacobian_3d = self.workspace.kpt_jacobian.rows(3 * k, 3).into_owned();
                accumulate_keypoint_residual(
                    obs,
                    &self.workspace.kpt_positions[k],
                    &jacobian_3d,
                    &mut self.jtj,
                    &mut self.jtr,
                );
            }

            accumulate_neutral_pose_prior(
                state,
                &self.neutral_joint_angles,
                config.neutral_pose_weight,
                &mut self.jtj,
                &mut self.jtr,
            );

            // Levenberg-Marquardt-style relative damping to improve numerical
            // stability. If the existing values in J^T J are large (i.e. larger
            // than 1, for example when coordinates are pixel coordinates and
            // can go up to thousands), the damping term is scaled up to match.
            for i in 0..state_dim {
                self.jtj[(i, i)] += config.damping * self.jtj[(i, i)].max(1.0);
            }

            // Update state
            let delta = nalgebra::linalg::Cholesky::new(self.jtj.clone())
                .map(|chol| chol.solve(&self.jtr))
                // no update if all keypoints are missing or if numerically unstable
                .unwrap_or_else(|| DVector::zeros(state_dim));
            state.apply_delta(&delta);
        }
    }
}

fn accumulate_keypoint_residual(
    obs: &KeypointObservation,
    fwdkin_pos3d: &Vector3<f32>,
    jacobian_3d: &DMatrix<f32>,
    jtj: &mut DMatrix<f32>,
    jtr: &mut DVector<f32>,
) {
    match *obs {
        KeypointObservation::Missing => {}
        KeypointObservation::Position3D { obs_pos, weight } => {
            let residual = obs_pos - fwdkin_pos3d;
            *jtj += jacobian_3d.transpose() * jacobian_3d * weight;
            *jtr += jacobian_3d.transpose() * residual * weight;
        }
        KeypointObservation::Position2D {
            obs_pos,
            weight,
            mapper,
        } => {
            let (fwdkin_pos2d, jacobian_2d) = mapper.project_3d_to_2d(fwdkin_pos3d, jacobian_3d);
            let residual = obs_pos - fwdkin_pos2d;
            *jtj += jacobian_2d.transpose() * &jacobian_2d * weight;
            *jtr += jacobian_2d.transpose() * residual * weight;
        }
    }
}

fn accumulate_neutral_pose_prior(
    state: &State,
    neutral_joint_angles: &[f32],
    weight: f32,
    jtj: &mut DMatrix<f32>,
    jtr: &mut DVector<f32>,
) {
    if weight == 0.0 {
        return;
    }
    for (i, (&curr_angle, &neutral_angle)) in (state.dof_angles)
        .iter()
        .zip(neutral_joint_angles)
        .enumerate()
    {
        let state_idx = N_ROOT_DOFS + i;
        jtj[(state_idx, state_idx)] += weight; // only contributor is self
        jtr[state_idx] += weight * (neutral_angle - curr_angle);
    }
}
