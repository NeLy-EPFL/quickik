//! Crate-private: the priors a [`SolverConfig`](crate::solver::SolverConfig)
//! holds, and their contributions to the Gauss-Newton normal equations.
//!
//! Every prior adds `J^T W J` to `jtj` (lower triangle only, which is all the
//! Cholesky decomposition reads) and `J^T W r` to `jtr`, where `r` is target
//! minus current value and `J` is the derivative of the current value w.r.t.
//! the state. Root rotation deltas are world-frame axis-angle vectors (see
//! `State::apply_delta`), and root priors are only allowed on free-floating
//! trees, whose state starts with 3 root position and 3 root rotation entries.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use crate::body_plan::KinematicTree;
use crate::state::State;

/// Pulls each DOF toward its neutral value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DofNeutralPrior {
    pub weight: f32,
}

/// Pulls the root's local `body_up` axis toward `world_up`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RootUprightPrior {
    pub weight: f32,
    /// Unit vector in the root's local frame.
    pub body_up: Vector3<f32>,
    /// Unit vector in the world frame.
    pub world_up: Vector3<f32>,
}

/// Pulls each DOF toward its value in the previous frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DofSmoothnessPrior {
    pub weight: f32,
}

/// Pulls the root pose toward the previous frame's.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RootSmoothnessPrior {
    pub position_weight: f32,
    pub rotation_weight: f32,
}

/// At most one prior of each type; `None` means absent.
#[derive(Clone, Debug, Default)]
pub(crate) struct Priors {
    pub dof_neutral: Option<DofNeutralPrior>,
    pub root_upright: Option<RootUprightPrior>,
    pub dof_smoothness: Option<DofSmoothnessPrior>,
    pub root_smoothness: Option<RootSmoothnessPrior>,
}

impl Priors {
    /// Whether any prior needs the previous frame's state.
    pub fn has_temporal(&self) -> bool {
        self.dof_smoothness.is_some() || self.root_smoothness.is_some()
    }

    /// Whether any prior acts on the root, which a fixed-base tree doesn't
    /// have as a state variable.
    pub fn has_root(&self) -> bool {
        self.root_upright.is_some() || self.root_smoothness.is_some()
    }

    /// Adds every prior's contribution. Smoothness priors are skipped if there
    /// is no previous frame (e.g. on a sequence's first frame).
    pub fn accumulate(
        &self,
        tree: &KinematicTree,
        state: &State,
        prev_state: Option<&State>,
        jtj: &mut DMatrix<f32>,
        jtr: &mut DVector<f32>,
    ) {
        if let Some(prior) = &self.dof_neutral {
            prior.accumulate(tree, state, jtj, jtr);
        }
        if let Some(prior) = &self.root_upright {
            prior.accumulate(state, jtj, jtr);
        }
        if let Some(prev_state) = prev_state {
            if let Some(prior) = &self.dof_smoothness {
                prior.accumulate(tree, state, prev_state, jtj, jtr);
            }
            if let Some(prior) = &self.root_smoothness {
                prior.accumulate(state, prev_state, jtj, jtr);
            }
        }
    }
}

impl DofNeutralPrior {
    fn accumulate(
        &self,
        tree: &KinematicTree,
        state: &State,
        jtj: &mut DMatrix<f32>,
        jtr: &mut DVector<f32>,
    ) {
        let targets = tree.dof_neutrals();
        accumulate_dof_targets(tree, &state.dof_values, targets, self.weight, jtj, jtr);
    }
}

impl DofSmoothnessPrior {
    fn accumulate(
        &self,
        tree: &KinematicTree,
        state: &State,
        prev_state: &State,
        jtj: &mut DMatrix<f32>,
        jtr: &mut DVector<f32>,
    ) {
        let targets = &prev_state.dof_values;
        accumulate_dof_targets(tree, &state.dof_values, targets, self.weight, jtj, jtr);
    }
}

impl RootUprightPrior {
    fn accumulate(&self, state: &State, jtj: &mut DMatrix<f32>, jtr: &mut DVector<f32>) {
        // Current up direction `u = R * body_up`. Rotating by a small
        // world-frame `delta` moves it by `delta x u = -[u]x delta`, so
        // `J = -[u]x`, `J^T J = |u|^2 I - u u^T`, and `J^T r = u x r`.
        // Rotation about `u` itself is left free.
        let u = state.root_quat * self.body_up;
        let r = self.world_up - u;
        let jtj_block = Matrix3::identity() * u.norm_squared() - u * u.transpose();
        let jtr_block = u.cross(&r);
        for a in 0..3 {
            jtr[3 + a] += self.weight * jtr_block[a];
            for b in 0..=a {
                jtj[(3 + a, 3 + b)] += self.weight * jtj_block[(a, b)];
            }
        }
    }
}

impl RootSmoothnessPrior {
    fn accumulate(
        &self,
        state: &State,
        prev_state: &State,
        jtj: &mut DMatrix<f32>,
        jtr: &mut DVector<f32>,
    ) {
        let d_pos = prev_state.root_pos - state.root_pos;
        // Residual: the rotation taking the current orientation to the
        // previous one, as a world-frame axis-angle vector. Applying a small
        // delta shrinks it by approximately that delta, so `J = I`, as for the
        // position.
        let d_rot = (prev_state.root_quat * state.root_quat.inverse()).scaled_axis();
        for a in 0..3 {
            jtj[(a, a)] += self.position_weight;
            jtr[a] += self.position_weight * d_pos[a];
            jtj[(3 + a, 3 + a)] += self.rotation_weight;
            jtr[3 + a] += self.rotation_weight * d_rot[a];
        }
    }
}

/// Pulls every DOF toward `targets`, with per-DOF weight
/// `weight * Dof::weight_multiplier`. Shared by both DOF priors, which differ
/// only in their targets. Only touches the diagonal.
fn accumulate_dof_targets(
    tree: &KinematicTree,
    values: &[f32],
    targets: &[f32],
    weight: f32,
    jtj: &mut DMatrix<f32>,
    jtr: &mut DVector<f32>,
) {
    let n_root_dofs = tree.n_root_dofs();
    let multipliers = tree.dof_weight_multipliers();
    for (d, ((&value, &target), &multiplier)) in
        values.iter().zip(targets).zip(multipliers).enumerate()
    {
        let w = weight * multiplier;
        let s = n_root_dofs + d;
        jtj[(s, s)] += w;
        jtr[s] += w * (target - value);
    }
}
