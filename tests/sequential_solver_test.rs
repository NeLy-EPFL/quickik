mod common;

use std::sync::Arc;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::observation::{KeypointObservation, NoMapper};
use quickik::sequential_solver::SequenceSolver;
use quickik::state::State;

const N_ITERATIONS: usize = 10;
const NEUTRAL_WEIGHT: f32 = 1e-3;
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

fn new_seq_solver(
    tree: &Arc<KinematicTree>,
    n_iterations: usize,
    neutral_weight: f32,
) -> SequenceSolver {
    SequenceSolver::new(
        tree,
        NoMapper,
        n_iterations,
        neutral_weight,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    )
}

#[test]
fn solve_always_warm_starts_across_separate_calls() {
    let tree = common::two_joint_chain();
    let target = observations_for(&tree, &[0.4, 0.3]);

    // Cold start: a single Gauss-Newton iteration from the neutral pose.
    let mut cold = new_seq_solver(&tree, 1, 0.0);
    let cold_result = cold.solve(std::slice::from_ref(&target), false, false);
    let cold_error = (cold_result[0].dof_angles[0] - 0.4).abs();

    // Warm start: two separate `.solve()` calls with the same (static)
    // target -- the second call continues from the first call's near-answer
    // instead of from neutral (SequenceSolver always warm-starts across
    // calls, for its whole lifetime), so it should end up closer after the
    // same one iteration.
    let mut warm = new_seq_solver(&tree, 1, 0.0);
    warm.solve(std::slice::from_ref(&target), false, false);
    let warm_results = warm.solve(std::slice::from_ref(&target), false, false);
    let warm_error = (warm_results[0].dof_angles[0] - 0.4).abs();

    assert!(
        warm_error < cold_error,
        "warm-started second call ({warm_error}) should be closer to the target than a single \
         cold-start iteration ({cold_error})"
    );
}

#[test]
fn solve_returns_one_result_per_frame() {
    let tree = common::two_joint_chain();
    let mut seq_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);

    let sequence = vec![
        observations_for(&tree, &[0.1, 0.05]),
        observations_for(&tree, &[0.2, 0.1]),
        observations_for(&tree, &[0.3, 0.15]),
    ];
    let results = seq_solver.solve(&sequence, false, false);

    assert_eq!(results.len(), 3);
    assert!((results[2].dof_angles[0] - 0.3).abs() < 1e-2);
    assert!((results[2].dof_angles[1] - 0.15).abs() < 1e-2);
}

#[test]
fn keypoint_pos_matches_the_converged_state() {
    let tree = common::two_joint_chain();
    let mut seq_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    let target = observations_for(&tree, &[0.4, 0.3]);
    let results = seq_solver.solve(std::slice::from_ref(&target), false, true);
    let result = &results[0];

    let mut expected_state = State::neutral_pose(tree.clone());
    expected_state
        .dof_angles
        .copy_from_slice(&result.dof_angles);
    expected_state.root_pos = result.root_pos;
    expected_state.root_rot = result.root_rot;
    let mut expected_workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut expected_workspace, &expected_state);

    let actual = result.keypoint_pos.as_ref().unwrap();
    assert_eq!(actual.len(), expected_workspace.kpt_positions.len());
    for (a, e) in actual.iter().zip(&expected_workspace.kpt_positions) {
        assert!((a - e).norm() < 1e-5, "actual={a:?} expected={e:?}");
    }
}

#[test]
fn with_fk_true_does_not_change_the_converged_trajectory() {
    let tree = common::two_joint_chain();
    let sequence = vec![
        observations_for(&tree, &[0.1, 0.05]),
        observations_for(&tree, &[0.2, 0.1]),
        observations_for(&tree, &[0.3, 0.15]),
    ];

    let mut plain_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    let plain_results = plain_solver.solve(&sequence, false, false);

    let mut fk_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    let fk_results = fk_solver.solve(&sequence, false, true);

    assert_eq!(fk_results.len(), plain_results.len());
    for (fk_result, plain_result) in fk_results.iter().zip(&plain_results) {
        assert_eq!(fk_result.dof_angles, plain_result.dof_angles);
        assert!(fk_result.keypoint_pos.is_some());
        assert!(plain_result.keypoint_pos.is_none());
    }
}

fn sine_trajectory(
    tree: &Arc<KinematicTree>,
    n_frames: usize,
) -> (Vec<Vec<KeypointObservation>>, Vec<[f32; 2]>) {
    // A smooth trajectory well within joint2's [-0.5, 0.5] limit, so limit
    // clamping doesn't confound the closeness checks below.
    let true_angles: Vec<[f32; 2]> = (0..n_frames)
        .map(|t| {
            let a = 0.3 * (t as f32 * 0.15).sin();
            [a, a * 0.5]
        })
        .collect();
    let sequence = true_angles
        .iter()
        .map(|angles| observations_for(tree, angles))
        .collect();
    (sequence, true_angles)
}

#[test]
fn solve_segments_parallel_reconstructs_smooth_trajectory() {
    let tree = common::two_joint_chain();
    let n_frames = 40;
    let (sequence, true_angles) = sine_trajectory(&tree, n_frames);

    let seq_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    // A handful of workers (not tied to this machine's core count), giving
    // segments of ~10 frames each -- large enough for each segment's own
    // internal warm-starting to settle onto the trajectory.
    let results = seq_solver.solve_segments_parallel(&sequence, 4, false, false);

    assert_eq!(results.len(), n_frames);
    for (result, angles) in results.iter().zip(&true_angles) {
        assert!(
            (result.dof_angles[0] - angles[0]).abs() < 1e-2,
            "dof0 mismatch: got {}, want {}",
            result.dof_angles[0],
            angles[0]
        );
        assert!(
            (result.dof_angles[1] - angles[1]).abs() < 1e-2,
            "dof1 mismatch: got {}, want {}",
            result.dof_angles[1],
            angles[1]
        );
    }
}

#[test]
fn solve_segments_parallel_with_one_worker_matches_plain_solve_exactly() {
    // n_workers: 1 forces the whole sequence through a single segment, cold
    // started once and warm-started throughout -- bit-for-bit the same
    // computation a fresh SequenceSolver's plain `solve` would do over the
    // same sequence.
    let tree = common::two_joint_chain();
    let (sequence, _) = sine_trajectory(&tree, 40);

    let mut plain_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    let plain_results = plain_solver.solve(&sequence, false, false);

    let parallel_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    let parallel_results = parallel_solver.solve_segments_parallel(&sequence, 1, false, false);

    assert_eq!(parallel_results.len(), plain_results.len());
    for (parallel_result, plain_result) in parallel_results.iter().zip(&plain_results) {
        assert_eq!(parallel_result.dof_angles, plain_result.dof_angles);
        assert_eq!(parallel_result.root_pos, plain_result.root_pos);
        assert_eq!(parallel_result.root_rot, plain_result.root_rot);
    }
}

#[test]
fn solve_segments_parallel_handles_empty_input() {
    let tree = common::two_joint_chain();
    let seq_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    let results = seq_solver.solve_segments_parallel(&[], -1, false, false);
    assert!(results.is_empty());
}

#[test]
fn solve_segments_parallel_does_not_affect_the_solver_s_own_running_state() {
    // solve_segments_parallel is a self-contained bulk operation: it must
    // never read or write this object's own warm-started `solve` state.
    let tree = common::two_joint_chain();
    let mut seq_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);

    let target_a = observations_for(&tree, &[0.4, 0.3]);
    seq_solver.solve(std::slice::from_ref(&target_a), false, false);

    // A completely unrelated bulk sequence, run through solve_segments_parallel.
    let (unrelated_sequence, _) = sine_trajectory(&tree, 10);
    seq_solver.solve_segments_parallel(&unrelated_sequence, 2, false, false);

    // The object's own running state should still warm-start from target_a's
    // converged pose, unaffected by the parallel call above.
    let mut reference_solver = new_seq_solver(&tree, N_ITERATIONS, NEUTRAL_WEIGHT);
    reference_solver.solve(std::slice::from_ref(&target_a), false, false);
    let target_b = observations_for(&tree, &[0.41, 0.31]);
    let expected = reference_solver.solve(std::slice::from_ref(&target_b), false, false);
    let actual = seq_solver.solve(std::slice::from_ref(&target_b), false, false);

    assert_eq!(actual[0].dof_angles, expected[0].dof_angles);
}
