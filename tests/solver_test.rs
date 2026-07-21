mod common;

use fastik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use fastik::observation::{Camera, KeypointObservation, Mapper3Dto2D, XYView};
use fastik::solver::{Solver, SolverConfig};
use fastik::state::State;
use nalgebra::{Matrix3, Vector3};

fn keypoints_at(
    tree: &std::sync::Arc<fastik::body_plan::KinematicTree>,
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
            neutral_pose_weight: 0.0,
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
            neutral_pose_weight: 0.0,
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
    // The Jacobian argument only affects the returned projected Jacobian, not
    // the projected position, so a placeholder shape is fine here.
    let jac_placeholder = nalgebra::DMatrix::<f32>::zeros(3, 3);
    let observations: Vec<KeypointObservation> = target_positions
        .iter()
        .map(|pos| {
            let (obs_pos, _) = camera.project_3d_to_2d(pos, &jac_placeholder);
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
            neutral_pose_weight: 0.0,
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
