mod common;

use std::sync::Arc;

use nalgebra::{Vector2, Vector3};
use quickik::observation::{
    KeypointObservation, ObservationSequence, OrthographicViewXY, Projection,
};
use quickik::sequential_solver::SequentialSolver;
use quickik::solver::SolverConfig;
use quickik::state::State;

use common::{branching_tree, max_abs_diff, observations_3d, some_pose};

/// A smooth motion: `some_pose` scaled by a factor ramping from 0.5 to 1.5.
fn motion(tree: &Arc<quickik::body_plan::KinematicTree>, n_frames: usize) -> Vec<State> {
    let base = some_pose(tree);
    (0..n_frames)
        .map(|i| {
            let s = 0.5 + i as f32 / n_frames as f32;
            let mut state = base.clone();
            state.dof_values.iter_mut().for_each(|v| *v *= s);
            state.root_pos *= s;
            state
        })
        .collect()
}

fn config() -> SolverConfig {
    let mut config = SolverConfig::default();
    config.add_dof_neutral_prior(1e-4).unwrap();
    config
}

#[test]
fn step_and_solve_agree() {
    let tree = branching_tree(false);
    let frames: Vec<_> = motion(&tree, 10).iter().map(observations_3d).collect();

    let mut by_step = SequentialSolver::new(&tree, &config(), None).unwrap();
    let stepped: Vec<_> = frames
        .iter()
        .map(|f| by_step.step(f, false, false).unwrap())
        .collect();
    let mut by_solve = SequentialSolver::new(&tree, &config(), None).unwrap();
    let solved = by_solve
        .solve(
            &ObservationSequence::from_lists(frames.clone()),
            false,
            false,
        )
        .unwrap();

    for (a, b) in stepped.iter().zip(&solved) {
        assert_eq!(a.state.dof_values, b.state.dof_values);
    }
    assert_eq!(by_step.state().dof_values, by_solve.state().dof_values);
}

#[test]
fn smoothness_priors_pull_toward_previous_frame() {
    let tree = branching_tree(false);
    let first = some_pose(&tree);
    let mut second = first.clone();
    second.dof_values.iter_mut().for_each(|v| *v += 0.3);
    second.root_pos += Vector3::new(0.5, 0.0, 0.0);
    // Weakly observed second frame.
    let weak: Vec<_> = observations_3d(&second)
        .into_iter()
        .map(|mut o| {
            o.weight = 1e-3;
            o
        })
        .collect();

    let solve_second = |config: &SolverConfig| {
        let mut solver = SequentialSolver::new(&tree, config, None).unwrap();
        solver.step(&observations_3d(&first), false, false).unwrap();
        solver.step(&weak, false, false).unwrap().state
    };
    let mut smooth = SolverConfig::new(30, 0.0, 0.0, 1e-6);
    smooth.add_dof_smoothness_prior(1.0).unwrap();
    smooth.add_root_smoothness_prior(1.0, 1.0).unwrap();
    let with_prior = solve_second(&smooth);
    let without_prior = solve_second(&SolverConfig::new(30, 0.0, 0.0, 1e-6));

    // Without the prior, the weak observations are fit exactly; with it, the
    // result stays close to the first frame.
    assert!(max_abs_diff(&without_prior.dof_values, &second.dof_values) < 1e-3);
    assert!(max_abs_diff(&with_prior.dof_values, &first.dof_values) < 0.05);
    assert!((with_prior.root_pos - first.root_pos).norm() < 0.05);
}

#[test]
fn reset_forgets_previous_frame() {
    let tree = branching_tree(false);
    let mut config = SolverConfig::new(30, 0.0, 0.0, 1e-6);
    config.add_dof_smoothness_prior(1e3).unwrap();
    let mut solver = SequentialSolver::new(&tree, &config, None).unwrap();
    solver
        .step(
            &observations_3d(&State::neutral_pose(Arc::clone(&tree))),
            false,
            false,
        )
        .unwrap();
    solver.reset(None).unwrap();
    // First frame after the reset: smoothness inactive, so it's fit exactly.
    let target = some_pose(&tree);
    let result = solver
        .step(&observations_3d(&target), false, false)
        .unwrap();
    assert!(max_abs_diff(&result.state.dof_values, &target.dof_values) < 1e-3);
}

#[test]
fn solve_parallel_matches_sequential_solve() {
    let tree = branching_tree(false);
    let truth = motion(&tree, 40);
    let frames: Vec<_> = truth.iter().map(observations_3d).collect();
    let sequence = ObservationSequence::from_lists(frames.clone());

    let solver = SequentialSolver::new(&tree, &config(), None).unwrap();
    for n_overlap in [0, 5, 100] {
        let results = solver
            .solve_parallel(&sequence, 4, n_overlap, false, true)
            .unwrap();
        assert_eq!(results.len(), truth.len());
        for (result, expected) in results.iter().zip(&truth) {
            assert!(max_abs_diff(&result.state.dof_values, &expected.dof_values) < 1e-2);
            assert!(result.keypoint_pos.is_some());
        }
    }
    // The running state is untouched.
    assert_eq!(
        solver.state().dof_values,
        State::neutral_pose(Arc::clone(&tree)).dof_values
    );
}

#[test]
fn from_arrays_matches_from_lists() {
    let tree = branching_tree(false);
    let n_joints = tree.n_joints();
    let truth = motion(&tree, 3);
    let camera = Projection::from(OrthographicViewXY::new(2.0, Vector2::new(1.0, -1.0)));

    let mut pos_3d = Vec::new();
    let mut weights_3d = Vec::new();
    let mut pos_2d = Vec::new();
    let mut weights_2d = Vec::new();
    let mut lists = Vec::new();
    let shared_camera = Arc::new(camera);
    for (f, state) in truth.iter().enumerate() {
        let mut list = Vec::new();
        let positions = common::keypoint_positions(state);
        for (j, p) in positions.iter().enumerate() {
            // Joint 2 is missing in frame 1 (NaN weight in the arrays).
            let w = if f == 1 && j == 2 { f32::NAN } else { 1.0 };
            pos_3d.extend(p.iter());
            weights_3d.push(w);
            list.push(KeypointObservation::position_3d(
                j,
                *p,
                if w.is_nan() { 0.0 } else { w },
            ));
        }
        for (j, p) in positions.iter().enumerate() {
            let uv = 2.0 * Vector2::new(p.x, p.y) + Vector2::new(1.0, -1.0);
            pos_2d.extend(uv.iter());
            weights_2d.push(0.5);
            list.push(KeypointObservation::position_2d(
                j,
                uv,
                Arc::clone(&shared_camera),
                0.5,
            ));
        }
        lists.push(list);
    }
    let from_arrays = ObservationSequence::from_arrays(
        truth.len(),
        n_joints,
        Some(&pos_3d),
        Some(&weights_3d),
        &[Arc::clone(&shared_camera)],
        Some(&pos_2d),
        Some(&weights_2d),
    )
    .unwrap();
    let from_lists = ObservationSequence::from_lists(lists.clone());

    let solve = |sequence: &ObservationSequence| {
        let mut solver = SequentialSolver::new(&tree, &config(), None).unwrap();
        solver.solve(sequence, true, false).unwrap()
    };
    for (a, b) in solve(&from_arrays).iter().zip(&solve(&from_lists)) {
        assert_eq!(a.state.dof_values, b.state.dof_values);
        let jacobian = a.jacobian.as_ref().unwrap();
        assert_eq!(jacobian.nrows(), 3 * n_joints + 2 * n_joints);
        assert_eq!(jacobian, b.jacobian.as_ref().unwrap());
    }

    let bad = ObservationSequence::from_arrays(
        2,
        n_joints,
        Some(&pos_3d),
        Some(&weights_3d),
        &[],
        None,
        None,
    );
    assert!(bad.is_err(), "wrong array length was accepted");
}
