mod common;

use nalgebra::{DMatrix, Matrix3, Vector3};
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::observation::{Camera, KeypointObservation, Mapper3Dto2D, NoMapper, XYView};
use quickik::solver::Solver;
use quickik::state::State;

/// Defaults matching the old `SolverConfig::default()`.
const N_ITERATIONS: usize = 10;
const NEUTRAL_WEIGHT: f32 = 1e-3;
const POSITION_TOLERANCE: f32 = 1e-3;
const ANGLE_TOLERANCE: f32 = 1e-3;
const DAMPING: f32 = 1e-6;

fn keypoints_at(
    tree: &std::sync::Arc<quickik::body_plan::KinematicTree>,
    angles: &[f32],
) -> Vec<Vector3<f32>> {
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles.copy_from_slice(angles);
    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    evaluate_fwdkin(&mut workspace, &state);
    workspace.kpt_positions.clone()
}

#[test]
fn recovers_pose_from_3d_observations() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);

    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

    let mut state = State::neutral_pose(tree.clone());
    // No mapper needed: every observation is Position3D. Disable the
    // neutral-pose prior so an exactly-reachable target is recovered exactly.
    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, false);

    assert!((result.state.dof_angles[0] - 0.4).abs() < 1e-3);
    assert!((result.state.dof_angles[1] - 0.3).abs() < 1e-3);
}

/// A fixed-base tree's root has no state to fit, so the solver should recover
/// the same DOF angles as the free-floating case above while leaving
/// `root_pos`/`root_rot` untouched at their `neutral_pose` default.
#[test]
fn recovers_pose_on_fixed_base_tree_without_moving_root() {
    let tree = common::fixed_base_two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);

    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

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
    let result = solver.solve(&mut state, &observations, false, false);

    assert!((result.state.dof_angles[0] - 0.4).abs() < 1e-3);
    assert!((result.state.dof_angles[1] - 0.3).abs() < 1e-3);
    assert_eq!(result.state.root_pos, Vector3::zeros());
    assert_eq!(result.state.root_rot, nalgebra::UnitQuaternion::identity());
}

#[test]
fn recovers_pose_with_slide_dof_from_3d_observations() {
    let tree = common::hinge_then_slide_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);

    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

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
    let result = solver.solve(&mut state, &observations, false, false);

    assert!((result.state.dof_angles[0] - 0.4).abs() < 1e-3);
    assert!((result.state.dof_angles[1] - 0.3).abs() < 1e-3);
}

#[test]
fn recovers_pose_from_xyview_observations() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.35, -0.25]);

    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|pos| KeypointObservation::Position2D {
            obs_pos: nalgebra::Vector2::new(pos.x, pos.y),
            weight: 1.0,
        })
        .collect();

    let mut state = State::neutral_pose(tree.clone());
    let mut solver = Solver::new(
        &tree,
        XYView,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, false);

    assert!((result.state.dof_angles[0] - 0.35).abs() < 1e-3);
    assert!((result.state.dof_angles[1] - (-0.25)).abs() < 1e-3);
}

#[test]
fn recovers_pose_from_camera_observations() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.2, 0.15]);

    let camera = Camera {
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
        world2cam_pos: Vector3::new(0.0, 0.0, 5.0),
        world2cam_rot_mat: Matrix3::identity(),
    };
    // The Jacobian argument only affects the projected-Jacobian output, not
    // the projected position, so placeholder shapes are fine here.
    let jac_placeholder = nalgebra::DMatrix::<f32>::zeros(3, 3);
    let mut jac2d_placeholder = nalgebra::DMatrix::<f32>::zeros(2, 3);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|pos| {
            let obs_pos = camera.project_3d_to_2d(pos, &jac_placeholder, &mut jac2d_placeholder);
            KeypointObservation::Position2D {
                obs_pos,
                weight: 1.0,
            }
        })
        .collect();

    let mut state = State::neutral_pose(tree.clone());
    let mut solver = Solver::new(
        &tree,
        camera,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, false);

    assert!((result.state.dof_angles[0] - 0.2).abs() < 1e-3);
    assert!((result.state.dof_angles[1] - 0.15).abs() < 1e-3);
}

#[test]
fn missing_observations_leave_state_at_neutral_prior() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    let observations = vec![KeypointObservation::Missing; tree.n_joints()];

    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, false);

    for &angle in &result.state.dof_angles {
        assert!(angle.abs() < 1e-6, "expected no drift, got {angle}");
    }
}

#[test]
fn solver_fields_can_be_tuned_between_solve_calls() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    let observations = vec![KeypointObservation::Missing; tree.n_joints()];

    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    solver.solve(&mut state, &observations, false, false);
    // Mutate the field directly in place -- no need to reconstruct the Solver
    // (and its preallocated buffers) to change per-call numerics between
    // frames.
    solver.n_iterations = 3;
    solver.solve(&mut state, &observations, false, false);

    assert_eq!(solver.n_iterations, 3);
}

#[test]
fn solve_respects_joint_limits() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());

    // Pin root_rot via joint1 and dof1 via joint2, both at their neutral
    // (dof1 = 0) positions, then ask the `tip` keypoint (downstream of
    // joint2, which is limited to [-0.5, 0.5]) for a position that is only
    // reachable with dof2 ~= 1.2 rad -- well past its upper limit. The target
    // is placed off the arm's resting axis so the request isn't degenerate
    // for a first-order (Gauss-Newton) solver starting from a straight pose.
    let observations = vec![
        KeypointObservation::Missing,
        KeypointObservation::Position3D {
            obs_pos: Vector3::new(1.0, 0.0, 0.0),
            weight: 1.0,
        },
        KeypointObservation::Position3D {
            obs_pos: Vector3::new(2.0, 0.0, 0.0),
            weight: 1.0,
        },
        KeypointObservation::Position3D {
            obs_pos: Vector3::new(2.3624, 0.9320, 0.0),
            weight: 1.0,
        },
    ];

    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, false);

    assert!(
        result.state.dof_angles[1] >= -0.5 - 1e-6 && result.state.dof_angles[1] <= 0.5 + 1e-6,
        "joint2 angle {} exceeded its [-0.5, 0.5] limit",
        result.state.dof_angles[1]
    );
    // The target is unreachable within the limit, so the solver should be
    // pushing hard against the boundary rather than resting comfortably
    // inside it.
    assert!(
        result.state.dof_angles[1] > 0.45,
        "joint2 angle {} did not converge against its upper limit",
        result.state.dof_angles[1]
    );
}

#[test]
fn convergence_tolerance_stops_iterating_early() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

    // Tolerances far larger than any single Gauss-Newton step's magnitude in
    // this toy problem, so every solve below should stop after exactly one
    // iteration regardless of its n_iterations cap.
    let generous_tolerance = 10.0;

    let mut state_few = State::neutral_pose(tree.clone());
    let mut solver_few: Solver = Solver::new(
        &tree,
        NoMapper,
        1,
        0.0,
        generous_tolerance,
        generous_tolerance,
        DAMPING,
    );
    let result_few = solver_few.solve(&mut state_few, &observations, false, false);

    let mut state_many = State::neutral_pose(tree.clone());
    let mut solver_many: Solver = Solver::new(
        &tree,
        NoMapper,
        50,
        0.0,
        generous_tolerance,
        generous_tolerance,
        DAMPING,
    );
    let result_many = solver_many.solve(&mut state_many, &observations, false, false);

    // If early termination weren't stopping solver_many after its first
    // iteration too, it would have kept converging further than solver_few
    // over its remaining 49 iterations, and the two states would differ.
    assert_eq!(result_few.state.dof_angles, result_many.state.dof_angles);
    assert_eq!(result_few.state.root_pos, result_many.state.root_pos);
}

#[test]
fn joint_weight_scaler_zero_matches_missing_observation() {
    let tree = common::two_joint_chain();
    // joint2's own keypoint (index 2) gets its weight_scaler zeroed out --
    // whatever it's observed at should then have no effect on the solve.
    let mut zero_weight_joints = tree.joints.clone();
    zero_weight_joints[2].weight_scaler = 0.0;
    let zero_weight_tree = std::sync::Arc::new(quickik::body_plan::KinematicTree {
        joints: zero_weight_joints,
        root_idx: tree.root_idx,
        fixed_base: tree.fixed_base,
    });

    let tip_target = keypoints_at(&tree, &[0.4, 0.3])[3];
    let mut observations = vec![KeypointObservation::Missing; 4];
    // Deliberately conflicts with tip_target, so it would pull the solve
    // toward a different pose if joint2's weight_scaler didn't zero it out.
    observations[2] = KeypointObservation::Position3D {
        obs_pos: Vector3::new(5.0, 5.0, 5.0),
        weight: 1.0,
    };
    observations[3] = KeypointObservation::Position3D {
        obs_pos: tip_target,
        weight: 1.0,
    };

    let mut state_zero_weight = State::neutral_pose(zero_weight_tree.clone());
    let mut solver_zero_weight: Solver = Solver::new(
        &zero_weight_tree,
        NoMapper,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result_zero_weight =
        solver_zero_weight.solve(&mut state_zero_weight, &observations, false, false);

    let mut observations_missing = observations;
    observations_missing[2] = KeypointObservation::Missing;
    let mut state_missing = State::neutral_pose(tree.clone());
    let mut solver_missing: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result_missing =
        solver_missing.solve(&mut state_missing, &observations_missing, false, false);

    assert_eq!(
        result_zero_weight.state.dof_angles,
        result_missing.state.dof_angles
    );
}

#[test]
fn dof_weight_scaler_zero_recovers_exact_target_despite_nonzero_global_neutral_weight() {
    let tree = common::two_independent_single_dof_branches();
    // branch_b_joint's DOF (flattened index 1) gets its neutral-pose
    // contribution zeroed out, while branch_a_joint's DOF (index 0) keeps the
    // default nonzero global neutral-pose weight active. The two branches
    // share no keypoints, so each DOF's fit is otherwise independent.
    let mut joints = tree.joints.clone();
    joints[3].dofs[0].weight_scaler = 0.0;
    let zero_weight_tree = std::sync::Arc::new(quickik::body_plan::KinematicTree {
        joints,
        root_idx: tree.root_idx,
        fixed_base: tree.fixed_base,
    });

    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

    // Unlike `recovers_pose_from_3d_observations`, the global neutral-pose
    // weight is deliberately left at its default nonzero value here.
    let mut state = State::neutral_pose(zero_weight_tree.clone());
    let mut solver: Solver = Solver::new(
        &zero_weight_tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, false);

    // dof1 (branch_b_joint's), with its neutral-pose contribution zeroed
    // out, recovers the exact target...
    assert!((result.state.dof_angles[1] - 0.3).abs() < 1e-3);
    // ...while dof0 (branch_a_joint's), still pulled toward neutral by the
    // nonzero global weight, is measurably biased away from its exact target.
    assert!((result.state.dof_angles[0] - 0.4).abs() > 1e-3);
}

#[test]
#[should_panic(expected = "a Solver<NoMapper> (no mapper set) was given a Position2D observation")]
fn position2d_observation_on_mapperless_solver_panics() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    let mut observations = vec![KeypointObservation::Missing; tree.n_joints()];
    observations[1] = KeypointObservation::Position2D {
        obs_pos: nalgebra::Vector2::new(1.0, 0.0),
        weight: 1.0,
    };

    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    solver.solve(&mut state, &observations, false, false);
}

#[test]
fn solve_with_grad_jacobian_and_cholesky_reconstruct_normal_equations() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

    // Started from a bent (not neutral) pose: at the exact neutral pose every
    // keypoint of this chain is collinear along the x-axis, which leaves the
    // free root's roll DOF with a zero Jacobian column (a genuine physical
    // degeneracy, not a bug) and makes jtj singular rather than
    // positive-definite.
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles[0] = 0.2;
    state.dof_angles[1] = -0.15;
    // A single iteration, with damping and the neutral-pose prior both
    // disabled, makes `jtj` exactly `sum_k weight_k * J_k^T J_k` -- so it's
    // reconstructible from the returned Jacobian alone, without needing
    // access to the solver's private accumulation logic.
    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        1,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        0.0,
    );
    let result = solver.solve(&mut state, &observations, true, false);
    assert!(
        result.cholesky_l.is_some(),
        "a well-posed single GN step should be positive-definite"
    );

    let jacobian = result.jacobian.unwrap();
    let state_dim = jacobian.ncols();
    let n_joints = tree.n_joints();
    assert_eq!(jacobian.nrows(), 3 * n_joints);

    let mut expected_jtj = DMatrix::<f32>::zeros(state_dim, state_dim);
    for k in 0..n_joints {
        let block = jacobian.rows(3 * k, 3);
        expected_jtj += block.transpose() * block;
    }

    let l = result.cholesky_l.unwrap().l();
    let reconstructed_jtj = &l * l.transpose();

    let max_abs_diff = (reconstructed_jtj - expected_jtj)
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    assert!(
        max_abs_diff < 1e-4,
        "L * L^T should reconstruct jtj built from the returned Jacobian; max abs diff = {max_abs_diff}"
    );
}

#[test]
fn solve_with_grad_tracks_only_the_final_iterations_linearization() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

    let start_angles = [0.2, -0.15];
    let n_iterations = 5;
    // Zero tolerances disable early termination by contract, so this is
    // guaranteed to run all `n_iterations` steps -- letting this test pin
    // down exactly which pose the final iteration's linearization should be
    // at, to catch `solve_impl` snapshotting the wrong (e.g. first) iteration
    // instead of deferring correctly.
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles[0] = start_angles[0];
    state.dof_angles[1] = start_angles[1];
    let mut solver: Solver = Solver::new(&tree, NoMapper, n_iterations, 0.0, 0.0, 0.0, 0.0);
    let result = solver.solve(&mut state, &observations, true, false);
    assert!(result.cholesky_l.is_some());

    // The final iteration's Jacobian is linearized at the pose from just
    // before its own update -- i.e. wherever `n_iterations - 1` steps alone
    // would have landed, starting from the same initial pose.
    let mut second_to_last_state = State::neutral_pose(tree.clone());
    second_to_last_state.dof_angles[0] = start_angles[0];
    second_to_last_state.dof_angles[1] = start_angles[1];
    let mut warmup_solver: Solver =
        Solver::new(&tree, NoMapper, n_iterations - 1, 0.0, 0.0, 0.0, 0.0);
    warmup_solver.solve(&mut second_to_last_state, &observations, false, false);

    let mut expected_workspace = ForwardKinematicsWorkspace::new(&tree);
    evaluate_fwdkin(&mut expected_workspace, &second_to_last_state);

    let max_abs_diff = (result.jacobian.unwrap() - &expected_workspace.kpt_jacobian)
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    assert!(
        max_abs_diff < 1e-5,
        "the returned jacobian should be linearized at the pose from n_iterations-1 steps alone, \
         max abs diff = {max_abs_diff}"
    );
}

#[test]
fn solve_with_grad_returns_no_cholesky_when_unconstrained() {
    let tree = common::two_joint_chain();
    let observations = vec![KeypointObservation::Missing; tree.n_joints()];

    let mut state = State::neutral_pose(tree.clone());
    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        0.0,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        0.0,
    );
    let result = solver.solve(&mut state, &observations, true, false);

    assert!(
        result.cholesky_l.is_none(),
        "an all-Missing, unregularized solve has no PD normal equations"
    );
    // The Jacobian is still computable regardless of positive-definiteness.
    assert!(result.jacobian.is_some());
}

/// Checks `result.keypoint_pos` against a fresh `evaluate_fwdkin` at `state`
/// (rather than reusing `keypoints_at`, which assumes a neutral root):
/// `keypoint_pos` must reflect the pose *actually returned*, including
/// root_pos/root_rot, not just dof_angles.
fn assert_keypoint_pos_matches_state(
    result: &quickik::solver::SolverResult,
    tree: &std::sync::Arc<quickik::body_plan::KinematicTree>,
    state: &State,
) {
    let mut expected_workspace = ForwardKinematicsWorkspace::new(tree);
    evaluate_fwdkin(&mut expected_workspace, state);

    let actual = result.keypoint_pos.as_ref().expect("with_fk was requested");
    assert_eq!(actual.len(), expected_workspace.kpt_positions.len());
    for (a, e) in actual.iter().zip(&expected_workspace.kpt_positions) {
        assert!((a - e).norm() < 1e-5, "actual={a:?} expected={e:?}");
    }
}

#[test]
fn keypoint_pos_matches_the_returned_state() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

    let mut state = State::neutral_pose(tree.clone());
    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, true);

    assert_keypoint_pos_matches_state(&result, &tree, &state);
}

#[test]
fn keypoint_pos_matches_the_returned_state_with_grad_also_requested() {
    let tree = common::two_joint_chain();
    let target_positions = keypoints_at(&tree, &[0.4, 0.3]);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|&obs_pos| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect();

    let mut state = State::neutral_pose(tree.clone());
    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, true, true);

    assert_keypoint_pos_matches_state(&result, &tree, &state);
}

#[test]
fn keypoint_pos_is_populated_even_with_zero_iterations() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    state.dof_angles[0] = 0.3;
    state.dof_angles[1] = -0.2;
    let observations = vec![KeypointObservation::Missing; tree.n_joints()];

    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        0,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, true);

    // n_iterations: 0 means state is untouched, so this should match exactly.
    assert_keypoint_pos_matches_state(&result, &tree, &state);
}

#[test]
fn with_grad_and_with_fk_false_leaves_optional_fields_none() {
    let tree = common::two_joint_chain();
    let observations = vec![KeypointObservation::Missing; tree.n_joints()];
    let mut state = State::neutral_pose(tree.clone());
    let mut solver: Solver = Solver::new(
        &tree,
        NoMapper,
        N_ITERATIONS,
        NEUTRAL_WEIGHT,
        POSITION_TOLERANCE,
        ANGLE_TOLERANCE,
        DAMPING,
    );
    let result = solver.solve(&mut state, &observations, false, false);

    assert!(result.keypoint_pos.is_none());
    assert!(result.jacobian.is_none());
    assert!(result.cholesky_l.is_none());
}
