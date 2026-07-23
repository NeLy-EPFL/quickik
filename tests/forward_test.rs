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
