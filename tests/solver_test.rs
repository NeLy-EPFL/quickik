mod common;

use nalgebra::{Matrix3, Vector3};
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::observation::{Camera, KeypointObservation, Mapper3Dto2D, XYView};
use quickik::solver::{Solver, SolverConfig};
use quickik::state::State;

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
        SolverConfig {
            neutral_weight: 0.0,
            ..SolverConfig::default()
        },
    );
    solver.solve(&mut state, &observations);

    assert!((state.dof_angles[0] - 0.4).abs() < 1e-3);
    assert!((state.dof_angles[1] - 0.3).abs() < 1e-3);
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
        SolverConfig {
            neutral_weight: 0.0,
            ..SolverConfig::default()
        },
    );
    solver.solve(&mut state, &observations);

    assert!((state.dof_angles[0] - 0.4).abs() < 1e-3);
    assert!((state.dof_angles[1] - 0.3).abs() < 1e-3);
    assert_eq!(state.root_pos, Vector3::zeros());
    assert_eq!(state.root_rot, nalgebra::UnitQuaternion::identity());
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
        SolverConfig {
            neutral_weight: 0.0,
            ..SolverConfig::default()
        },
    );
    solver.solve(&mut state, &observations);

    assert!((state.dof_angles[0] - 0.4).abs() < 1e-3);
    assert!((state.dof_angles[1] - 0.3).abs() < 1e-3);
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
        SolverConfig {
            neutral_weight: 0.0,
            mapper: Some(XYView),
            ..SolverConfig::default()
        },
    );
    solver.solve(&mut state, &observations);

    assert!((state.dof_angles[0] - 0.35).abs() < 1e-3);
    assert!((state.dof_angles[1] - (-0.25)).abs() < 1e-3);
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
        SolverConfig {
            neutral_weight: 0.0,
            mapper: Some(camera),
            ..SolverConfig::default()
        },
    );
    solver.solve(&mut state, &observations);

    assert!((state.dof_angles[0] - 0.2).abs() < 1e-3);
    assert!((state.dof_angles[1] - 0.15).abs() < 1e-3);
}

#[test]
fn missing_observations_leave_state_at_neutral_prior() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    let observations = vec![KeypointObservation::Missing; tree.n_joints()];

    let mut solver: Solver = Solver::new(&tree, SolverConfig::default());
    solver.solve(&mut state, &observations);

    for &angle in &state.dof_angles {
        assert!(angle.abs() < 1e-6, "expected no drift, got {angle}");
    }
}

#[test]
fn config_can_be_tuned_between_solve_calls() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    let observations = vec![KeypointObservation::Missing; tree.n_joints()];

    let mut solver: Solver = Solver::new(&tree, SolverConfig::default());
    solver.solve(&mut state, &observations);
    // Mutate the config in place -- no need to reconstruct the Solver (and
    // its preallocated buffers) to change per-call numerics between frames.
    solver.config.n_iterations = 3;
    solver.solve(&mut state, &observations);

    assert_eq!(solver.config.n_iterations, 3);
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

    let mut solver: Solver = Solver::new(&tree, SolverConfig::default());
    solver.solve(&mut state, &observations);

    assert!(
        state.dof_angles[1] >= -0.5 - 1e-6 && state.dof_angles[1] <= 0.5 + 1e-6,
        "joint2 angle {} exceeded its [-0.5, 0.5] limit",
        state.dof_angles[1]
    );
    // The target is unreachable within the limit, so the solver should be
    // pushing hard against the boundary rather than resting comfortably
    // inside it.
    assert!(
        state.dof_angles[1] > 0.45,
        "joint2 angle {} did not converge against its upper limit",
        state.dof_angles[1]
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
    let generous_tolerance_config = SolverConfig {
        neutral_weight: 0.0,
        position_tolerance: 10.0,
        angle_tolerance: 10.0,
        ..SolverConfig::default()
    };

    let mut state_few = State::neutral_pose(tree.clone());
    let mut solver_few: Solver = Solver::new(
        &tree,
        SolverConfig {
            n_iterations: 1,
            ..generous_tolerance_config
        },
    );
    solver_few.solve(&mut state_few, &observations);

    let mut state_many = State::neutral_pose(tree.clone());
    let mut solver_many: Solver = Solver::new(
        &tree,
        SolverConfig {
            n_iterations: 50,
            ..generous_tolerance_config
        },
    );
    solver_many.solve(&mut state_many, &observations);

    // If early termination weren't stopping solver_many after its first
    // iteration too, it would have kept converging further than solver_few
    // over its remaining 49 iterations, and the two states would differ.
    assert_eq!(state_few.dof_angles, state_many.dof_angles);
    assert_eq!(state_few.root_pos, state_many.root_pos);
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

    let config = SolverConfig {
        neutral_weight: 0.0,
        ..SolverConfig::default()
    };

    let mut state_zero_weight = State::neutral_pose(zero_weight_tree.clone());
    let mut solver_zero_weight: Solver = Solver::new(&zero_weight_tree, config);
    solver_zero_weight.solve(&mut state_zero_weight, &observations);

    let mut observations_missing = observations;
    observations_missing[2] = KeypointObservation::Missing;
    let mut state_missing = State::neutral_pose(tree.clone());
    let mut solver_missing: Solver = Solver::new(&tree, config);
    solver_missing.solve(&mut state_missing, &observations_missing);

    assert_eq!(state_zero_weight.dof_angles, state_missing.dof_angles);
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
    let mut solver: Solver = Solver::new(&zero_weight_tree, SolverConfig::default());
    solver.solve(&mut state, &observations);

    // dof1 (branch_b_joint's), with its neutral-pose contribution zeroed
    // out, recovers the exact target...
    assert!((state.dof_angles[1] - 0.3).abs() < 1e-3);
    // ...while dof0 (branch_a_joint's), still pulled toward neutral by the
    // nonzero global weight, is measurably biased away from its exact target.
    assert!((state.dof_angles[0] - 0.4).abs() > 1e-3);
}

#[test]
#[should_panic(expected = "Solver constructed with mapper: None")]
fn position2d_observation_on_mapperless_solver_panics() {
    let tree = common::two_joint_chain();
    let mut state = State::neutral_pose(tree.clone());
    let mut observations = vec![KeypointObservation::Missing; tree.n_joints()];
    observations[1] = KeypointObservation::Position2D {
        obs_pos: nalgebra::Vector2::new(1.0, 0.0),
        weight: 1.0,
    };

    let mut solver: Solver = Solver::new(&tree, SolverConfig::default());
    solver.solve(&mut state, &observations);
}
