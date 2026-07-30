mod common;

use std::sync::Arc;

use nalgebra::Vector3;
use quickik::batched_solver::BatchedSolver;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::observation::{KeypointObservation, NoMapper};
use quickik::solver::Solver;
use quickik::state::State;

const N_ITERATIONS: usize = 10;
const POSITION_TOLERANCE: f32 = 1e-3;
const ANGLE_TOLERANCE: f32 = 1e-3;
const DAMPING: f32 = 1e-6;

fn keypoints_at(tree: &Arc<KinematicTree>, angles: &[f32]) -> Vec<Vector3<f32>> {
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles.copy_from_slice(angles);
    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    evaluate_fwdkin(&mut workspace, &state);
    workspace.kpt_positions.clone()
}

fn observations_for(tree: &Arc<KinematicTree>, angles: &[f32]) -> Vec<KeypointObservation> {
    keypoints_at(tree, angles)
        .into_iter()
        .map(|obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect()
}

fn joint_names(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

#[test]
fn solve_matches_sequential_solve_with_grad() {
    let tree = common::two_joint_chain();

    // A permutation of the tree's own joint order ("root", "joint1",
    // "joint2", "tip"), so this test actually exercises name-based
    // remapping rather than happening to pass only for the identity order.
    let keypoints_order = joint_names(&["tip", "root", "joint2", "joint1"]);
    // external position i <- internal joint index order_joint_indices[i].
    let order_joint_indices = [3usize, 0, 2, 1];

    let targets: Vec<[f32; 2]> = vec![[0.4, 0.3], [-0.2, 0.1], [0.3, -0.4], [0.15, 0.25]];

    // Ground truth: solve each item sequentially, from the neutral pose,
    // with its own `Solver` and tree-order observations directly.
    let mut expected_dof_angles = Vec::new();
    let mut expected_root_pos = Vec::new();
    let mut expected_root_rot = Vec::new();
    let mut expected_ok = Vec::new();
    let mut expected_jacobians = Vec::new();
    for angles in &targets {
        let mut state = State::neutral_pose(tree.clone());
        let mut solver: Solver = Solver::new(
            &tree,
            NoMapper,
            N_ITERATIONS,
            0.0,
            POSITION_TOLERANCE,
            ANGLE_TOLERANCE,
            DAMPING,
        );
        let result = solver.solve(&mut state, &observations_for(&tree, angles), true, false);
        expected_ok.push(result.cholesky_l.is_some());
        expected_jacobians.push(result.jacobian.unwrap());
        expected_dof_angles.push(result.dof_angles);
        expected_root_pos.push(result.root_pos);
        expected_root_rot.push(result.root_rot);
    }

    // Same observations, but permuted into `keypoints_order` -- this is what
    // BatchedSolver::solve actually receives.
    let observations_array: Vec<Vec<KeypointObservation>> = targets
        .iter()
        .map(|angles| {
            let internal_order = observations_for(&tree, angles);
            order_joint_indices
                .iter()
                .map(|&joint_idx| internal_order[joint_idx])
                .collect()
        })
        .collect();

    let batched_solver = BatchedSolver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
        keypoints_order,
    );
    let result = batched_solver.solve(&observations_array, true, false);

    assert_eq!(result.joint_angles.len(), expected_dof_angles.len());
    let cholesky_l = result.cholesky_l.unwrap();
    let jacobian = result.jacobian.unwrap();
    assert!(result.keypoint_pos.is_none(), "with_fk was false");
    for i in 0..expected_dof_angles.len() {
        assert_eq!(
            cholesky_l[i].is_some(),
            expected_ok[i],
            "item {i}: PD-ness should match the sequential solve"
        );
        assert_eq!(
            result.joint_angles[i], expected_dof_angles[i],
            "item {i}: converged joint angles should match the sequential solve exactly"
        );
        assert_eq!(
            result.base_pos[i], expected_root_pos[i],
            "item {i}: converged base_pos should match the sequential solve exactly"
        );
        assert_eq!(
            result.base_quat[i], expected_root_rot[i],
            "item {i}: converged base_quat should match the sequential solve exactly"
        );
        assert_eq!(
            jacobian[i], expected_jacobians[i],
            "item {i}: jacobian should match the sequential solve exactly"
        );
    }
}

#[test]
fn with_grad_false_and_with_fk_false_leaves_optional_fields_none() {
    let tree = common::two_joint_chain();
    let keypoints_order = joint_names(&["root", "joint1", "joint2", "tip"]);
    let batched_solver = BatchedSolver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        1e-3,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
        keypoints_order,
    );

    let observations_array = vec![
        observations_for(&tree, &[0.4, 0.3]),
        observations_for(&tree, &[-0.1, 0.2]),
    ];
    let result = batched_solver.solve(&observations_array, false, false);

    assert_eq!(result.joint_angles.len(), 2);
    assert!(result.keypoint_pos.is_none());
    assert!(result.jacobian.is_none());
    assert!(result.cholesky_l.is_none());
    // joint_angles/base_pos/base_quat are always populated regardless.
    assert!((result.joint_angles[0][0] - 0.4).abs() < 1e-2);
}

#[test]
fn with_fk_true_reports_keypoint_positions() {
    let tree = common::two_joint_chain();
    let keypoints_order = joint_names(&["root", "joint1", "joint2", "tip"]);
    let batched_solver = BatchedSolver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
        keypoints_order,
    );

    let target_angles = [0.4, 0.3];
    let observations_array = vec![observations_for(&tree, &target_angles)];
    let result = batched_solver.solve(&observations_array, false, true);

    let keypoint_pos = result.keypoint_pos.unwrap();
    let expected = keypoints_at(&tree, &target_angles);
    assert_eq!(keypoint_pos[0].len(), expected.len());
    for (a, e) in keypoint_pos[0].iter().zip(&expected) {
        assert!((a - e).norm() < 1e-2, "actual={a:?} expected={e:?}");
    }
}

#[test]
#[should_panic(expected = "unknown joint name")]
fn new_rejects_unknown_joint_name() {
    let tree = common::two_joint_chain();
    let keypoints_order = joint_names(&["root", "joint1", "joint2", "nonexistent"]);
    BatchedSolver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        1e-3,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
        keypoints_order,
    );
}

#[test]
#[should_panic(expected = "listed more than once")]
fn new_rejects_duplicate_joint_name() {
    let tree = common::two_joint_chain();
    let keypoints_order = joint_names(&["root", "joint1", "joint1", "tip"]);
    BatchedSolver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        1e-3,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
        keypoints_order,
    );
}

#[test]
#[should_panic(expected = "fixed-base tree")]
fn new_rejects_fixed_base_tree() {
    let tree = common::fixed_base_two_joint_chain();
    let keypoints_order = joint_names(&["root", "joint1", "joint2", "tip"]);
    BatchedSolver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        1e-3,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
        keypoints_order,
    );
}
