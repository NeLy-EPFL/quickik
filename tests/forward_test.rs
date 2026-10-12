mod common;

use std::sync::Arc;

use nalgebra::{DMatrix, UnitQuaternion, Vector3};
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, forward_kinematics};
use quickik::state::State;

use common::{branching_tree, keypoint_positions, some_pose};

/// Perturbs `state` along state index `s` by `eps`, following the solver's
/// state parametrization (world-frame axis-angle for root rotation).
fn perturbed(state: &State, s: usize, eps: f32) -> State {
    let mut state = state.clone();
    let n_root_dofs = state.kinematic_tree.n_root_dofs();
    if s < n_root_dofs.min(3) {
        state.root_pos[s] += eps;
    } else if s < n_root_dofs {
        let mut axis_angle = Vector3::zeros();
        axis_angle[s - 3] = eps;
        state.root_quat = UnitQuaternion::from_scaled_axis(axis_angle) * state.root_quat;
    } else {
        state.dof_values[s - n_root_dofs] += eps;
    }
    state
}

fn finite_difference_jacobian(state: &State) -> DMatrix<f32> {
    let tree = &state.kinematic_tree;
    let eps = 1e-3;
    let mut jacobian = DMatrix::zeros(3 * tree.n_joints(), tree.state_dim());
    for s in 0..tree.state_dim() {
        let plus = keypoint_positions(&perturbed(state, s, eps));
        let minus = keypoint_positions(&perturbed(state, s, -eps));
        for k in 0..tree.n_joints() {
            let column = (plus[k] - minus[k]) / (2.0 * eps);
            jacobian.fixed_view_mut::<3, 1>(3 * k, s).copy_from(&column);
        }
    }
    jacobian
}

#[test]
fn jacobian_matches_finite_differences() {
    for fixed_base in [false, true] {
        let tree = branching_tree(fixed_base);
        let state = some_pose(&tree);
        let mut workspace = ForwardKinematicsWorkspace::new(&tree);
        forward_kinematics(&mut workspace, &state).unwrap();
        let analytic = workspace.jacobian();
        let numeric = finite_difference_jacobian(&state);
        let max_err = (analytic - numeric).abs().max();
        assert!(
            max_err < 1e-3,
            "fixed_base={fixed_base}: max error {max_err}"
        );
    }
}

#[test]
fn joint_order_does_not_affect_results() {
    // Same tree with joints listed children-first, so the root isn't joint 0
    // and parents come after their children.
    let tree = branching_tree(false);
    let order = [5, 3, 2, 1, 4, 0]; // new index i holds old joint order[i]
    let new_index = |old: usize| order.iter().position(|&o| o == old).unwrap();
    let joints = order
        .iter()
        .map(|&old| {
            let mut joint = tree.joints()[old].clone();
            joint.parent = joint.parent.map(new_index);
            joint
        })
        .collect();
    let shuffled = Arc::new(KinematicTree::new(joints, false).unwrap());
    assert_eq!(shuffled.root_idx(), 5);

    let state = some_pose(&tree);
    let mut shuffled_state = State::neutral_pose(Arc::clone(&shuffled));
    shuffled_state.root_pos = state.root_pos;
    shuffled_state.root_quat = state.root_quat;
    for (new, &old) in order.iter().enumerate() {
        let n = tree.joints()[old].dofs.len();
        let old_start = tree.joint_dof_start_indices()[old];
        let new_start = shuffled.joint_dof_start_indices()[new];
        shuffled_state.dof_values[new_start..new_start + n]
            .copy_from_slice(&state.dof_values[old_start..old_start + n]);
    }

    let positions = keypoint_positions(&state);
    let shuffled_positions = keypoint_positions(&shuffled_state);
    for (new, &old) in order.iter().enumerate() {
        assert!((positions[old] - shuffled_positions[new]).norm() < 1e-6);
    }
}

#[test]
fn rejects_state_of_wrong_size() {
    let tree = branching_tree(false);
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    let mut state = some_pose(&tree);
    state.dof_values.pop();
    assert!(forward_kinematics(&mut workspace, &state).is_err());
}
