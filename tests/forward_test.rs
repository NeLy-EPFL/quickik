mod common;

use nalgebra::Vector3;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::state::State;

#[test]
fn neutral_pose_positions() {
    let tree = common::two_joint_chain();
    let state = State::neutral_pose(tree.clone());
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    assert!((workspace.kpt_positions[0] - Vector3::new(0.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((workspace.kpt_positions[1] - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((workspace.kpt_positions[2] - Vector3::new(2.0, 0.0, 0.0)).norm() < 1e-6);
}

#[test]
fn bent_pose_positions() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    // Rotate joint1 by 90 degrees about Z: joint2 should swing to (1, 1, 0).
    state.dof_angles[0] = std::f32::consts::FRAC_PI_2;
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    assert!((workspace.kpt_positions[1] - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-5);
    assert!((workspace.kpt_positions[2] - Vector3::new(1.0, 1.0, 0.0)).norm() < 1e-5);
}

/// Verifies `relevant_dof_idxs_by_joint` tracks exactly which state indices each
/// keypoint's Jacobian row can have a nonzero entry in: root's N_ROOT_DOFS
/// always, plus each ancestor joint's own DOF in root-to-parent order
/// (never a joint's *own* DOF, since rotating a joint only moves its
/// descendants -- see the module's doc comment).
#[test]
fn active_indices_track_ancestor_dofs_not_own_dof() {
    let tree = common::two_joint_chain();
    let state = State::neutral_pose(tree.clone());
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    // root: only N_ROOT_DOFS, no ancestors.
    assert_eq!(
        workspace.relevant_dof_idxs_by_joint[0],
        vec![0, 1, 2, 3, 4, 5]
    );
    // joint1: only N_ROOT_DOFS -- its own DOF (state index 6) never affects
    // its own keypoint.
    assert_eq!(
        workspace.relevant_dof_idxs_by_joint[1],
        vec![0, 1, 2, 3, 4, 5]
    );
    // joint2: N_ROOT_DOFS plus joint1's DOF (its ancestor), not its own.
    assert_eq!(
        workspace.relevant_dof_idxs_by_joint[2],
        vec![0, 1, 2, 3, 4, 5, 6]
    );
    // tip: N_ROOT_DOFS plus both upstream joints' DOFs.
    assert_eq!(
        workspace.relevant_dof_idxs_by_joint[3],
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
}

/// On a tree with two independent single-DOF branches sharing no keypoints,
/// a keypoint's active indices must include only its *own* branch's DOF --
/// this is the sparsity the accumulation step in `solver.rs` relies on to
/// skip work for keypoints unaffected by a given DOF.
#[test]
fn active_indices_exclude_other_branches_dofs() {
    let tree = common::two_independent_single_dof_branches();
    let state = State::neutral_pose(tree.clone());
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    // branch_a_tip (index 2): N_ROOT_DOFS plus branch_a_joint's DOF (index
    // 0's flattened state index, 6) -- never branch_b_joint's (state index
    // 7), even though both are only one hop from the root.
    assert_eq!(
        workspace.relevant_dof_idxs_by_joint[2],
        vec![0, 1, 2, 3, 4, 5, 6]
    );
    // branch_b_tip (index 4): symmetric, only branch_b_joint's DOF (7).
    assert_eq!(
        workspace.relevant_dof_idxs_by_joint[4],
        vec![0, 1, 2, 3, 4, 5, 7]
    );
}

/// Cross-check the analytical Jacobian against central finite differences
/// across every state variable, catching any sign or indexing errors in
/// `write_keypoint_jacobian`.
#[test]
fn jacobian_matches_finite_differences() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles[0] = 0.3;
    state.dof_angles[1] = -0.2;

    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);
    let analytical_jacobian = workspace.kpt_jacobian.clone();
    let baseline_positions = workspace.kpt_positions.clone();

    let eps = 1e-4;
    for var in 0..state.state_dim() {
        let mut delta = nalgebra::DVector::zeros(state.state_dim());
        delta[var] = eps;
        let mut perturbed = state.clone();
        perturbed.apply_delta(&delta);

        evaluate_fwdkin(&mut workspace, &perturbed);
        for (k, (numerical_position, baseline_position)) in workspace
            .kpt_positions
            .iter()
            .zip(&baseline_positions)
            .enumerate()
        {
            let numerical_d = (numerical_position - baseline_position) / eps;
            let analytical_d = analytical_jacobian.fixed_view::<3, 1>(3 * k, var);
            assert!(
                (numerical_d - analytical_d).norm() < 1e-2,
                "keypoint {k}, state var {var}: analytical {analytical_d:?} vs numerical {numerical_d:?}"
            );
        }
    }
}
