mod common;

use nalgebra::DVector;
use quickik::state::State;

#[test]
fn apply_delta_clamps_limited_dofs_but_not_unbounded_ones() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());

    // dof_angles[0] (joint1) is unbounded, dof_angles[1] (joint2) is limited
    // to [-0.5, 0.5]. Push both far past 0.5.
    let mut delta = DVector::zeros(state.state_dim());
    delta[6] = 10.0;
    delta[7] = 10.0;
    state.apply_delta(&delta);

    assert!((state.dof_angles[0] - 10.0).abs() < 1e-6);
    assert!((state.dof_angles[1] - 0.5).abs() < 1e-6);

    // Push the limited DOF back down past its lower bound too.
    let mut delta = DVector::zeros(state.state_dim());
    delta[7] = -10.0;
    state.apply_delta(&delta);
    assert!((state.dof_angles[1] - (-0.5)).abs() < 1e-6);
}

#[test]
fn apply_delta_updates_root_position_and_rotation() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());

    let mut delta = DVector::zeros(state.state_dim());
    delta[0] = 1.0;
    delta[1] = 2.0;
    delta[2] = 3.0;
    state.apply_delta(&delta);

    assert!((state.root_pos - nalgebra::Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-6);
}
