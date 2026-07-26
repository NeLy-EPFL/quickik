mod common;

use std::sync::Arc;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::high_level::{ParallelSolveConfig, SequenceSolver, solve_sequence_segmented_parallel};
use quickik::observation::KeypointObservation;
use quickik::solver::SolverConfig;
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
