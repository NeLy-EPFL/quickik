mod common;

use std::sync::Arc;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::high_level::{
    ParallelSolveConfig, SequenceSolver, solve_batch_with_grad, solve_sequence_segmented_parallel,
};
use quickik::observation::KeypointObservation;
use quickik::solver::{Solver, SolverConfig};
use quickik::state::State;

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

#[test]
fn solve_frame_warm_starts_from_previous_pose() {
    let tree = common::two_joint_chain();
    let config = SolverConfig {
        n_iterations: 1,
        neutral_weight: 0.0,
        ..SolverConfig::default()
    };
    let target = observations_for(&tree, &[0.4, 0.3]);

    // Cold start: a single Gauss-Newton iteration from the neutral pose.
    let mut cold: SequenceSolver = SequenceSolver::new(tree.clone(), config);
    cold.solve_frame(&target);
    let cold_error = (cold.state.dof_angles[0] - 0.4).abs();

    // Warm start: two consecutive frames with the same (static) target -- the
    // second call starts from the first call's near-answer instead of from
    // neutral, so it should end up closer after the same one iteration.
    let mut warm: SequenceSolver = SequenceSolver::new(tree.clone(), config);
    warm.solve_frame(&target);
    warm.solve_frame(&target);
    let warm_error = (warm.state.dof_angles[0] - 0.4).abs();

    assert!(
        warm_error < cold_error,
        "warm-started second frame ({warm_error}) should be closer to the target \
         than a single cold-start iteration ({cold_error})"
    );
}

#[test]
fn solve_sequence_returns_one_state_per_frame() {
    let tree = common::two_joint_chain();
    let mut seq_solver: SequenceSolver = SequenceSolver::new(tree.clone(), SolverConfig::default());

    let sequence = vec![
        observations_for(&tree, &[0.1, 0.05]),
        observations_for(&tree, &[0.2, 0.1]),
        observations_for(&tree, &[0.3, 0.15]),
    ];
    let states = seq_solver.solve_sequence(&sequence);

    assert_eq!(states.len(), 3);
    assert!((states[2].dof_angles[0] - 0.3).abs() < 1e-2);
    assert!((states[2].dof_angles[1] - 0.15).abs() < 1e-2);
}

#[test]
fn sequence_solver_last_fk_positions_matches_the_converged_state() {
    let tree = common::two_joint_chain();
    let mut seq_solver: SequenceSolver = SequenceSolver::new(tree.clone(), SolverConfig::default());
    seq_solver.solve_frame(&observations_for(&tree, &[0.4, 0.3]));

    let mut expected_workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut expected_workspace, &seq_solver.state);

    let actual = seq_solver.last_fk_positions();
    assert_eq!(actual.len(), expected_workspace.kpt_positions.len());
    for (a, e) in actual.iter().zip(&expected_workspace.kpt_positions) {
        assert!((a - e).norm() < 1e-5, "actual={a:?} expected={e:?}");
    }
}

#[test]
fn solve_sequence_with_fk_matches_solve_sequence_states_and_fk_positions() {
    let tree = common::two_joint_chain();
    let sequence = vec![
        observations_for(&tree, &[0.1, 0.05]),
        observations_for(&tree, &[0.2, 0.1]),
        observations_for(&tree, &[0.3, 0.15]),
    ];

    let mut plain_solver: SequenceSolver =
        SequenceSolver::new(tree.clone(), SolverConfig::default());
    let plain_states = plain_solver.solve_sequence(&sequence);

    let mut fk_solver: SequenceSolver = SequenceSolver::new(tree.clone(), SolverConfig::default());
    let results = fk_solver.solve_sequence_with_fk(&sequence);

    assert_eq!(results.len(), plain_states.len());
    for ((state, fk), plain_state) in results.iter().zip(&plain_states) {
        assert_eq!(state.dof_angles, plain_state.dof_angles);

        let mut expected_workspace = ForwardKinematicsWorkspace::new(&tree);
        evaluate_fwdkin(&mut expected_workspace, state);
        assert_eq!(fk.len(), expected_workspace.kpt_positions.len());
        for (a, e) in fk.iter().zip(&expected_workspace.kpt_positions) {
            assert!((a - e).norm() < 1e-5, "actual={a:?} expected={e:?}");
        }
    }
}

#[test]
fn solve_sequence_segmented_parallel_reconstructs_smooth_trajectory() {
    let tree = common::two_joint_chain();
    let n_frames = 40;
    // A smooth trajectory well within joint2's [-0.5, 0.5] limit, so limit
    // clamping doesn't confound the closeness check below.
    let true_angles: Vec<[f32; 2]> = (0..n_frames)
        .map(|t| {
            let a = 0.3 * (t as f32 * 0.15).sin();
            [a, a * 0.5]
        })
        .collect();
    let sequence: Vec<Vec<KeypointObservation>> = true_angles
        .iter()
        .map(|angles| observations_for(&tree, angles))
        .collect();

    let parallel_config = ParallelSolveConfig {
        segment_len: 10,
        overlap_len: 3,
        overlap_tolerance: 0.05,
        n_workers: -1,
    };
    let config: SolverConfig = SolverConfig::default();
    let states = solve_sequence_segmented_parallel(&tree, config, &sequence, parallel_config);

    assert_eq!(states.len(), n_frames);
    for (state, angles) in states.iter().zip(&true_angles) {
        assert!(
            (state.dof_angles[0] - angles[0]).abs() < 1e-2,
            "dof0 mismatch: got {}, want {}",
            state.dof_angles[0],
            angles[0]
        );
        assert!(
            (state.dof_angles[1] - angles[1]).abs() < 1e-2,
            "dof1 mismatch: got {}, want {}",
            state.dof_angles[1],
            angles[1]
        );
    }
}

#[test]
fn parallel_solve_config_for_recording_reconstructs_smooth_trajectory() {
    let tree = common::two_joint_chain();
    let n_frames = 40;
    // A smooth trajectory well within joint2's [-0.5, 0.5] limit, so limit
    // clamping doesn't confound the closeness check below.
    let true_angles: Vec<[f32; 2]> = (0..n_frames)
        .map(|t| {
            let a = 0.3 * (t as f32 * 0.15).sin();
            [a, a * 0.5]
        })
        .collect();
    let sequence: Vec<Vec<KeypointObservation>> = true_angles
        .iter()
        .map(|angles| observations_for(&tree, angles))
        .collect();

    let parallel_config = ParallelSolveConfig::for_recording(n_frames);
    let config: SolverConfig = SolverConfig::default();
    let states = solve_sequence_segmented_parallel(&tree, config, &sequence, parallel_config);

    assert_eq!(states.len(), n_frames);
    for (state, angles) in states.iter().zip(&true_angles) {
        assert!((state.dof_angles[0] - angles[0]).abs() < 1e-2);
        assert!((state.dof_angles[1] - angles[1]).abs() < 1e-2);
    }
}

#[test]
fn solve_sequence_segmented_parallel_honors_explicit_n_workers() {
    let tree = common::two_joint_chain();
    let n_frames = 40;
    let true_angles: Vec<[f32; 2]> = (0..n_frames)
        .map(|t| {
            let a = 0.3 * (t as f32 * 0.15).sin();
            [a, a * 0.5]
        })
        .collect();
    let sequence: Vec<Vec<KeypointObservation>> = true_angles
        .iter()
        .map(|angles| observations_for(&tree, angles))
        .collect();

    // n_workers: 1 forces every segment through the same chunk on a single
    // spawned thread, exercising a different code path than the default
    // (-1, i.e. all available cores) used elsewhere in this file.
    let parallel_config = ParallelSolveConfig {
        segment_len: 10,
        overlap_len: 3,
        overlap_tolerance: 0.05,
        n_workers: 1,
    };
    let config: SolverConfig = SolverConfig::default();
    let states = solve_sequence_segmented_parallel(&tree, config, &sequence, parallel_config);

    assert_eq!(states.len(), n_frames);
    for (state, angles) in states.iter().zip(&true_angles) {
        assert!((state.dof_angles[0] - angles[0]).abs() < 1e-2);
        assert!((state.dof_angles[1] - angles[1]).abs() < 1e-2);
    }
}

#[test]
fn solve_batch_with_grad_matches_sequential_solve_with_grad() {
    let tree = common::two_joint_chain();
    let config = SolverConfig {
        neutral_weight: 0.0,
        ..SolverConfig::default()
    };

    // A permutation of the tree's own joint order ("root", "joint1",
    // "joint2", "tip"), so this test actually exercises name-based
    // remapping rather than happening to pass only for the identity order.
    let keypoints_order: Vec<String> = ["tip", "root", "joint2", "joint1"]
        .into_iter()
        .map(String::from)
        .collect();
    // external position i <- internal joint index order_joint_indices[i].
    let order_joint_indices = [3usize, 0, 2, 1];

    let targets: Vec<[f32; 2]> = vec![[0.4, 0.3], [-0.2, 0.1], [0.3, -0.4], [0.15, 0.25]];

    // Ground truth: solve each item sequentially, from the neutral pose,
    // with its own `Solver` and tree-order observations directly.
    let mut expected_states: Vec<State> = Vec::new();
    let mut expected_ok: Vec<bool> = Vec::new();
    let mut expected_jacobians = Vec::new();
    for angles in &targets {
        let mut state = State::neutral_pose(tree.clone());
        let mut solver: Solver = Solver::new(&tree, config);
        expected_ok.push(solver.solve_with_grad(&mut state, &observations_for(&tree, angles)));
        expected_jacobians.push(solver.last_jacobian().clone());
        expected_states.push(state);
    }

    // Same observations, but permuted into `keypoints_order` -- this is what
    // solve_batch_with_grad actually receives.
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

    let result = solve_batch_with_grad(&tree, config, &keypoints_order, &observations_array);

    assert_eq!(result.joint_angles.len(), expected_states.len());
    for i in 0..expected_states.len() {
        assert_eq!(
            result.cholesky_l[i].is_some(),
            expected_ok[i],
            "item {i}: PD-ness should match the sequential solve"
        );
        assert_eq!(
            result.joint_angles[i], expected_states[i].dof_angles,
            "item {i}: converged joint angles should match the sequential solve exactly"
        );
        assert_eq!(
            result.base_pos[i], expected_states[i].root_pos,
            "item {i}: converged base_pos should match the sequential solve exactly"
        );
        assert_eq!(
            result.base_quat[i], expected_states[i].root_rot,
            "item {i}: converged base_quat should match the sequential solve exactly"
        );
        assert_eq!(
            result.jacobian[i], expected_jacobians[i],
            "item {i}: last_jacobian should match the sequential solve exactly"
        );
    }
}

#[test]
#[should_panic(expected = "unknown joint name")]
fn solve_batch_with_grad_rejects_unknown_joint_name() {
    let tree = common::two_joint_chain();
    let keypoints_order: Vec<String> = ["root", "joint1", "joint2", "nonexistent"]
        .into_iter()
        .map(String::from)
        .collect();
    let observations_array = vec![vec![KeypointObservation::Missing; tree.n_joints()]];
    let config: SolverConfig = SolverConfig::default();
    solve_batch_with_grad(&tree, config, &keypoints_order, &observations_array);
}

#[test]
#[should_panic(expected = "listed more than once")]
fn solve_batch_with_grad_rejects_duplicate_joint_name() {
    let tree = common::two_joint_chain();
    let keypoints_order: Vec<String> = ["root", "joint1", "joint1", "tip"]
        .into_iter()
        .map(String::from)
        .collect();
    let observations_array = vec![vec![KeypointObservation::Missing; tree.n_joints()]];
    let config: SolverConfig = SolverConfig::default();
    solve_batch_with_grad(&tree, config, &keypoints_order, &observations_array);
}

#[test]
#[should_panic(expected = "fixed-base tree")]
fn solve_batch_with_grad_rejects_fixed_base_tree() {
    let tree = common::fixed_base_two_joint_chain();
    let keypoints_order: Vec<String> = ["root", "joint1", "joint2", "tip"]
        .into_iter()
        .map(String::from)
        .collect();
    let observations_array = vec![vec![KeypointObservation::Missing; tree.n_joints()]];
    let config: SolverConfig = SolverConfig::default();
    solve_batch_with_grad(&tree, config, &keypoints_order, &observations_array);
}

#[test]
fn solve_sequence_segmented_parallel_handles_empty_input() {
    let tree = common::two_joint_chain();
    let parallel_config = ParallelSolveConfig {
        segment_len: 10,
        overlap_len: 3,
        overlap_tolerance: 0.01,
        n_workers: -1,
    };
    let config: SolverConfig = SolverConfig::default();
    let result = solve_sequence_segmented_parallel(&tree, config, &[], parallel_config);
    assert!(result.is_empty());
}
