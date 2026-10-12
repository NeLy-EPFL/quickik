mod common;

use std::sync::Arc;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};
use quickik::body_plan::KinematicTree;
use quickik::observation::{KeypointObservation, PinholeCamera, Projection};
use quickik::solver::{SingleFrameSolver, SolverConfig};
use quickik::state::State;

use common::{branching_tree, keypoint_positions, max_abs_diff, observations_3d, some_pose};

/// No early stopping, no priors.
fn exact_config() -> SolverConfig {
    SolverConfig::new(30, 0.0, 0.0, 1e-6)
}

fn assert_same_pose(actual: &State, expected: &State, tol: f32) {
    let dof_err = max_abs_diff(&actual.dof_values, &expected.dof_values);
    let pos_err = (actual.root_pos - expected.root_pos).norm();
    let rot_err = actual.root_quat.angle_to(&expected.root_quat);
    assert!(
        dof_err < tol && pos_err < tol && rot_err < tol,
        "dof error {dof_err}, root position error {pos_err}, root rotation error {rot_err}"
    );
}

/// Two pinhole cameras 5 units away, looking at the origin along world +z and
/// along world -x respectively.
fn two_cameras() -> [Arc<Projection>; 2] {
    let rot_y = Rotation3::from_axis_angle(&Vector3::y_axis(), std::f32::consts::FRAC_PI_2);
    let cam = |rot: Matrix3<f32>| {
        Arc::new(Projection::from(PinholeCamera::new(
            800.0,
            800.0,
            320.0,
            240.0,
            Vector3::new(0.0, 0.0, 5.0),
            rot,
        )))
    };
    [cam(Matrix3::identity()), cam(*rot_y.matrix())]
}

/// Mirrors the pinhole formula in `PinholeCamera`'s docs.
fn project(pos: &Vector3<f32>, rot: &Matrix3<f32>) -> Vector2<f32> {
    let p = rot * pos + Vector3::new(0.0, 0.0, 5.0);
    Vector2::new(800.0 * p.x / p.z + 320.0, 800.0 * p.y / p.z + 240.0)
}

fn camera_rotations() -> [Matrix3<f32>; 2] {
    let rot_y = Rotation3::from_axis_angle(&Vector3::y_axis(), std::f32::consts::FRAC_PI_2);
    [Matrix3::identity(), *rot_y.matrix()]
}

#[test]
fn recovers_pose_from_3d_observations() {
    for fixed_base in [false, true] {
        let tree = branching_tree(fixed_base);
        let target = some_pose(&tree);
        let mut solver = SingleFrameSolver::new(&tree, &exact_config()).unwrap();
        let mut state = State::neutral_pose(Arc::clone(&tree));
        let result = solver
            .solve(&mut state, &observations_3d(&target), false, true)
            .unwrap();
        assert_same_pose(&state, &target, 1e-4);
        let fk_err = result
            .keypoint_pos
            .unwrap()
            .iter()
            .zip(keypoint_positions(&target))
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f32::max);
        assert!(fk_err < 1e-4);
    }
}

#[test]
fn recovers_pose_from_two_cameras() {
    let tree = branching_tree(false);
    let target = some_pose(&tree);
    let cameras = two_cameras();
    let mut observations = Vec::new();
    for (camera, rot) in cameras.iter().zip(camera_rotations()) {
        for (j, pos) in keypoint_positions(&target).iter().enumerate() {
            let uv = project(pos, &rot);
            observations.push(KeypointObservation::position_2d(
                j,
                uv,
                Arc::clone(camera),
                1.0,
            ));
        }
    }
    let mut solver = SingleFrameSolver::new(&tree, &exact_config()).unwrap();
    let mut state = State::neutral_pose(Arc::clone(&tree));
    solver
        .solve(&mut state, &observations, false, false)
        .unwrap();
    assert_same_pose(&state, &target, 1e-3);
}

#[test]
fn residual_jacobian_matches_finite_differences() {
    let tree = branching_tree(false);
    let pose = some_pose(&tree);
    let [cam_a, cam_b] = two_cameras();
    let [rot_a, rot_b] = camera_rotations();
    let observations = vec![
        KeypointObservation::position_2d(3, Vector2::zeros(), cam_a, 1.0),
        KeypointObservation::position_3d(5, Vector3::zeros(), 1.0),
        KeypointObservation::position_2d(3, Vector2::zeros(), cam_b, 1.0),
        KeypointObservation::position_3d(1, Vector3::zeros(), 0.0),
    ];
    // Stacked predictions in the same row layout as `SolverResult::jacobian`.
    let predict = |state: &State| -> Vec<f32> {
        let p = keypoint_positions(state);
        let mut rows = Vec::new();
        rows.extend(project(&p[3], &rot_a).iter());
        rows.extend(p[5].iter());
        rows.extend(project(&p[3], &rot_b).iter());
        rows.extend(p[1].iter());
        rows
    };

    // One iteration linearizes at the input pose.
    let mut solver = SingleFrameSolver::new(&tree, &SolverConfig::new(1, 0.0, 0.0, 1e-6)).unwrap();
    let result = solver
        .solve(&mut pose.clone(), &observations, true, false)
        .unwrap();
    let jacobian = result.jacobian.unwrap();
    assert_eq!(jacobian.shape(), (10, tree.state_dim()));

    let eps = 1e-3;
    for s in [0, 4, 6, 8, 10] {
        let perturb = |eps: f32| {
            let mut state = pose.clone();
            if s < 3 {
                state.root_pos[s] += eps;
            } else if s < 6 {
                let mut axis_angle = Vector3::zeros();
                axis_angle[s - 3] = eps;
                state.root_quat = UnitQuaternion::from_scaled_axis(axis_angle) * state.root_quat;
            } else {
                state.dof_values[s - 6] += eps;
            }
            predict(&state)
        };
        let (plus, minus) = (perturb(eps), perturb(-eps));
        for row in 0..10 {
            let numeric = (plus[row] - minus[row]) / (2.0 * eps);
            let analytic = jacobian[(row, s)];
            // Pixel rows are ~100x larger than world rows.
            let tol = if matches!(row, 0 | 1 | 5 | 6) {
                0.5
            } else {
                1e-3
            };
            assert!(
                (numeric - analytic).abs() < tol,
                "row {row}, col {s}: {numeric} vs {analytic}"
            );
        }
    }
}

#[test]
fn zero_weight_multiplier_ignores_keypoint() {
    // The hand is the only keypoint constraining the arm's DOF. With its
    // multiplier at 0, a bogus hand observation must have no effect, and the
    // neutral prior keeps the arm at neutral.
    let tree = branching_tree(false);
    let mut joints = tree.joints().to_vec();
    joints[5].weight_multiplier = 0.0;
    let tree = Arc::new(KinematicTree::new(joints, false).unwrap());
    let target = some_pose(&tree);
    let mut observations = observations_3d(&target);
    observations[5] = KeypointObservation::position_3d(5, Vector3::new(9.0, 9.0, 9.0), 1.0);

    let mut config = exact_config();
    config.add_dof_neutral_prior(1e-6).unwrap();
    let mut solver = SingleFrameSolver::new(&tree, &config).unwrap();
    let mut state = State::neutral_pose(Arc::clone(&tree));
    solver
        .solve(&mut state, &observations, false, false)
        .unwrap();
    assert!(
        state.dof_values[4].abs() < 1e-4,
        "arm DOF moved to {}",
        state.dof_values[4]
    );
    assert!(max_abs_diff(&state.dof_values[..4], &target.dof_values[..4]) < 1e-3);
}

#[test]
fn root_upright_prior_aligns_body_up_with_world_up() {
    let tree = branching_tree(false);
    let body_up = Vector3::new(0.0, 0.0, 1.0);
    let world_up = Vector3::new(0.3, 0.0, 1.0).normalize();
    let mut config = exact_config();
    config
        .add_root_upright_prior(1.0, body_up, world_up)
        .unwrap();
    let mut solver = SingleFrameSolver::new(&tree, &config).unwrap();
    let mut state = some_pose(&tree);
    // Nothing observed: only the prior (and damping) acts.
    solver.solve(&mut state, &[], false, false).unwrap();
    let angle = (state.root_quat * body_up).angle(&world_up);
    assert!(angle < 1e-3, "angle to world up: {angle}");
}

#[test]
fn rejects_invalid_configs_and_inputs() {
    let tree = branching_tree(false);
    let fixed = branching_tree(true);

    let mut config = SolverConfig::default();
    config.add_dof_neutral_prior(1e-3).unwrap();
    assert!(config.add_dof_neutral_prior(1e-3).is_err());
    assert!(config.add_dof_smoothness_prior(-1.0).is_err());
    assert!(
        config
            .add_root_upright_prior(1.0, Vector3::zeros(), Vector3::z())
            .is_err()
    );

    let mut smoothness = SolverConfig::default();
    smoothness.add_dof_smoothness_prior(1.0).unwrap();
    assert!(SingleFrameSolver::new(&tree, &smoothness).is_err());

    let mut upright = SolverConfig::default();
    upright
        .add_root_upright_prior(1.0, Vector3::z(), Vector3::z())
        .unwrap();
    assert!(SingleFrameSolver::new(&fixed, &upright).is_err());
    assert!(SingleFrameSolver::new(&tree, &upright).is_ok());

    let mut solver = SingleFrameSolver::new(&tree, &SolverConfig::default()).unwrap();
    let mut state = State::neutral_pose(Arc::clone(&tree));
    let bad_joint = [KeypointObservation::position_3d(6, Vector3::zeros(), 1.0)];
    assert!(solver.solve(&mut state, &bad_joint, false, false).is_err());
    let mut other_tree_state = State::neutral_pose(fixed);
    other_tree_state.dof_values.pop();
    assert!(
        solver
            .solve(&mut other_tree_state, &[], false, false)
            .is_err()
    );
}
