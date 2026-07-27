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
/// descendants; see the module's doc comment).
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
/// a keypoint's active indices must include only its *own* branch's DOF:
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
fn assert_jacobian_matches_finite_differences(
    tree: &std::sync::Arc<quickik::body_plan::KinematicTree>,
    dof_values: &[f32],
) {
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles.copy_from_slice(dof_values);

    let mut workspace = ForwardKinematicsWorkspace::new(tree);
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

#[test]
fn jacobian_matches_finite_differences() {
    assert_jacobian_matches_finite_differences(&common::two_joint_chain(), &[0.3, -0.2]);
}

#[test]
fn jacobian_matches_finite_differences_with_slide_dof() {
    assert_jacobian_matches_finite_differences(&common::slide_joint_chain(), &[0.4]);
}

#[test]
fn jacobian_matches_finite_differences_with_hinge_then_slide() {
    assert_jacobian_matches_finite_differences(
        &common::hinge_then_slide_chain(),
        &[std::f32::consts::FRAC_PI_4, 0.4],
    );
}

#[test]
fn jacobian_matches_finite_differences_with_hinge_and_slide_on_same_joint() {
    assert_jacobian_matches_finite_differences(
        &common::joint_with_hinge_and_slide(),
        &[std::f32::consts::FRAC_PI_4, 0.4],
    );
}

/// A slide DOF translates its joint's frame along its (world-rotated) axis,
/// but, like a hinge DOF's rotation, never moves its own joint's tracked
/// keypoint, only its descendants'.
#[test]
fn slide_dof_moves_only_descendants() {
    let tree = common::slide_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles[0] = 0.5;
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    assert!((workspace.kpt_positions[0] - Vector3::new(0.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((workspace.kpt_positions[1] - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((workspace.kpt_positions[2] - Vector3::new(2.5, 0.0, 0.0)).norm() < 1e-5);
}

/// A downstream slide's world-frame axis follows an upstream hinge's
/// rotation, so rotating the hinge also swings the slide's translation
/// direction, and therefore every keypoint past it.
#[test]
fn hinge_then_slide_positions() {
    let tree = common::hinge_then_slide_chain();
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles[0] = std::f32::consts::FRAC_PI_2; // hinge_joint: 90 deg about Z
    state.dof_angles[1] = 0.3; // slide_joint: slide by 0.3 along its local X
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    assert!((workspace.kpt_positions[1] - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-5);
    assert!((workspace.kpt_positions[2] - Vector3::new(1.0, 1.0, 0.0)).norm() < 1e-5);
    assert!((workspace.kpt_positions[3] - Vector3::new(1.0, 2.3, 0.0)).norm() < 1e-5);
}

/// On a fixed-base tree, the root contributes no state at all, so a
/// keypoint's active indices are just its ancestors' own DOFs (never a
/// leading `0..N_ROOT_DOFS` block, unlike the free-floating case in
/// `active_indices_track_ancestor_dofs_not_own_dof` above).
#[test]
fn fixed_base_tree_has_no_root_dofs_in_active_indices() {
    let tree = common::fixed_base_two_joint_chain();
    let state = State::neutral_pose(tree.clone());
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    assert_eq!(workspace.relevant_dof_idxs_by_joint[0], Vec::<usize>::new());
    assert_eq!(workspace.relevant_dof_idxs_by_joint[1], Vec::<usize>::new());
    assert_eq!(workspace.relevant_dof_idxs_by_joint[2], vec![0]);
    assert_eq!(workspace.relevant_dof_idxs_by_joint[3], vec![0, 1]);
}

/// A fixed-base tree's keypoints move exactly like its free-floating
/// counterpart's when the root state is left at `neutral_pose`'s default
/// (zero position, identity rotation): `fixed_base` only removes the
/// root's own state variables, it doesn't change where the root sits.
#[test]
fn fixed_base_tree_neutral_pose_matches_free_floating_counterpart() {
    let tree = common::fixed_base_two_joint_chain();
    let state = State::neutral_pose(tree.clone());
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    assert!((workspace.kpt_positions[0] - Vector3::new(0.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((workspace.kpt_positions[1] - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((workspace.kpt_positions[2] - Vector3::new(2.0, 0.0, 0.0)).norm() < 1e-6);
}

#[test]
fn jacobian_matches_finite_differences_fixed_base() {
    assert_jacobian_matches_finite_differences(&common::fixed_base_two_joint_chain(), &[0.3, -0.2]);
}

/// Hand-derived expected position and Jacobian for a single joint carrying a
/// hinge DOF then a slide DOF, at theta = pi/2, d = 0.5:
/// tip = (1 + (d+1) cos(theta), (d+1) sin(theta), 0) = (1, 1.5, 0);
/// d(tip)/d(theta) = (-(d+1) sin(theta), (d+1) cos(theta), 0) = (-1.5, 0, 0);
/// d(tip)/d(d) = (cos(theta), sin(theta), 0) = (0, 1, 0).
/// This directly exercises the cross-term where an earlier hinge within the
/// same joint rotates a later slide's translation axis.
#[test]
fn hinge_and_slide_on_same_joint_cross_term() {
    let tree = common::joint_with_hinge_and_slide();
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles[0] = std::f32::consts::FRAC_PI_2;
    state.dof_angles[1] = 0.5;
    let mut workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut workspace, &state);

    // joint1's own keypoint is unaffected by either of its own DOFs.
    assert!((workspace.kpt_positions[1] - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-6);
    assert!((workspace.kpt_positions[2] - Vector3::new(1.0, 1.5, 0.0)).norm() < 1e-5);

    let theta_col = workspace.kpt_jacobian.fixed_view::<3, 1>(3 * 2, 6);
    assert!((theta_col - Vector3::new(-1.5, 0.0, 0.0)).norm() < 1e-4);
    let d_col = workspace.kpt_jacobian.fixed_view::<3, 1>(3 * 2, 7);
    assert!((d_col - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-4);
}
