//! Cross-checks the new solvers against the pre-redesign `quickik_old` on the
//! NeuroMechFly fixtures, with equivalent settings: all weight multipliers 1
//! (the old version has none), a neutral prior of 1e-3, and default iteration
//! settings. Temporary, like `quickik_old` itself.

mod common;

use std::sync::Arc;

use nalgebra::{Vector2, Vector3};
use quickik::body_plan::KinematicTree;
use quickik::observation::{
    KeypointObservation, ObservationSequence, OrthographicViewXY, Projection,
};
use quickik::sequential_solver::SequentialSolver;
use quickik::solver::{SingleFrameSolver, SolverConfig};
use quickik::state::State;
use serde::Deserialize;

use common::{assets_dir, body_plan_path, max_abs_diff};

#[derive(Deserialize)]
struct Fixtures {
    real_frames: Vec<Frame>,
    native_rate_frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Frame {
    /// One target per joint except the root, in joint order.
    target_ego: Vec<[f32; 3]>,
}

fn load() -> (
    Arc<KinematicTree>,
    Arc<quickik_old::body_plan::KinematicTree>,
    Fixtures,
) {
    // The old version ignores `weight_multiplier`, so reset them all to 1.
    let path = body_plan_path("neuromechfly");
    let mut joints = KinematicTree::from_json_file(&path)
        .unwrap()
        .joints()
        .to_vec();
    joints.iter_mut().for_each(|j| j.weight_multiplier = 1.0);
    let new_tree = Arc::new(KinematicTree::new(joints, false).unwrap());
    let old_tree = Arc::new(quickik_old::body_plan::KinematicTree::from_json_file(&path));
    let json = std::fs::read_to_string(assets_dir().join("fixtures.json")).unwrap();
    (new_tree, old_tree, serde_json::from_str(&json).unwrap())
}

fn config() -> SolverConfig {
    let mut config = SolverConfig::default();
    config.add_dof_neutral_prior(1e-3).unwrap();
    config
}

fn old_solver(
    tree: &quickik_old::body_plan::KinematicTree,
    projection: quickik_old::observation::Projection,
) -> quickik_old::solver::Solver {
    quickik_old::solver::Solver::new(tree, projection, 10, 1e-3, 1e-3, 1e-3, 1e-6)
}

fn new_observations_3d(frame: &Frame) -> Vec<KeypointObservation> {
    let targets = frame.target_ego.iter().enumerate();
    targets
        .map(|(i, p)| KeypointObservation::position_3d(i + 1, Vector3::from(*p), 1.0))
        .collect()
}

fn old_observations(
    frame: &Frame,
    is_2d: bool,
) -> Vec<quickik_old::observation::KeypointObservation> {
    use quickik_old::observation::KeypointObservation as Old;
    let targets = frame.target_ego.iter().map(|p| match is_2d {
        false => Old::Position3D {
            obs_pos: Vector3::from(*p),
            weight: 1.0,
        },
        true => Old::Position2D {
            obs_pos: Vector2::new(p[0], p[1]),
            weight: 1.0,
        },
    });
    std::iter::once(Old::Missing).chain(targets).collect()
}

#[test]
fn single_frame_matches_old_version() {
    let (new_tree, old_tree, fixtures) = load();
    let ortho = Arc::new(Projection::from(OrthographicViewXY::new(
        1.0,
        Vector2::zeros(),
    )));
    for is_2d in [false, true] {
        let old_projection = if is_2d {
            quickik_old::observation::Projection::new_ortho_xy()
        } else {
            quickik_old::observation::Projection::new_3d()
        };
        let mut old = old_solver(&old_tree, old_projection);
        let mut new = SingleFrameSolver::new(&new_tree, &config()).unwrap();
        for frame in &fixtures.real_frames {
            let observations: Vec<_> = if is_2d {
                let targets = frame.target_ego.iter().enumerate();
                targets
                    .map(|(i, p)| {
                        let uv = Vector2::new(p[0], p[1]);
                        KeypointObservation::position_2d(i + 1, uv, Arc::clone(&ortho), 1.0)
                    })
                    .collect()
            } else {
                new_observations_3d(frame)
            };
            let mut new_state = State::neutral_pose(Arc::clone(&new_tree));
            new.solve(&mut new_state, &observations, false, false)
                .unwrap();
            let mut old_state = quickik_old::state::State::neutral_pose(Arc::clone(&old_tree));
            old.solve(
                &mut old_state,
                &old_observations(frame, is_2d),
                false,
                false,
            );

            let err = max_abs_diff(&new_state.dof_values, &old_state.dof_values);
            assert!(err < 1e-3, "is_2d={is_2d}: max DOF difference {err}");
            assert!((new_state.root_pos - old_state.root_pos).norm() < 1e-3);
        }
    }
}

#[test]
fn warm_started_sequence_matches_old_version() {
    let (new_tree, old_tree, fixtures) = load();
    let frames: Vec<_> = fixtures
        .native_rate_frames
        .iter()
        .map(new_observations_3d)
        .collect();
    let mut new = SequentialSolver::new(&new_tree, &config(), None).unwrap();
    let new_results = new
        .solve(
            &ObservationSequence::from_lists(frames.clone()),
            false,
            false,
        )
        .unwrap();

    let mut old = quickik_old::sequential_solver::SequenceSolver::new(
        &old_tree,
        quickik_old::observation::Projection::new_3d(),
        10,
        1e-3,
        1e-3,
        1e-3,
        1e-6,
    );
    let old_frames: Vec<_> = fixtures
        .native_rate_frames
        .iter()
        .map(|f| old_observations(f, false))
        .collect();
    let old_results = old.solve(&old_frames, false, false);

    let err = new_results
        .iter()
        .zip(&old_results)
        .map(|(a, b)| max_abs_diff(&a.state.dof_values, &b.state.dof_values))
        .fold(0.0, f32::max);
    assert!(err < 1e-3, "max DOF difference over the sequence: {err}");
}
