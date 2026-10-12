//! Shared test fixtures.
//!
//! A joint's own DOFs never move its own keypoint (only its descendants'), so
//! every DOF needs a keypoint downstream of it to be observable, hence the
//! trailing DOF-less tip joints below.

#![allow(dead_code)] // not every test binary uses every helper

use std::path::PathBuf;
use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};
use quickik::body_plan::{Dof, DofType, Joint, KinematicTree};
use quickik::observation::KeypointObservation;
use quickik::state::State;

pub fn joint(name: &str, parent: Option<usize>, offset: [f32; 3], dofs: Vec<Dof>) -> Joint {
    Joint {
        name: name.to_string(),
        parent,
        offset_pos: Vector3::from(offset),
        offset_quat: UnitQuaternion::identity(),
        dofs,
        weight_multiplier: 1.0,
    }
}

pub fn hinge(axis: Vector3<f32>) -> Dof {
    Dof::new(DofType::Hinge, axis, 0.0, None, 1.0)
}

pub fn slide(axis: Vector3<f32>) -> Dof {
    Dof::new(DofType::Slide, axis, 0.0, None, 1.0)
}

/// A branching tree exercising hinges about non-aligned axes, slides, a joint
/// with several DOFs, and non-identity offset rotations:
///
/// root
///   -> hip: [hinge z, hinge y]
///     -> knee: [hinge x, slide x]
///       -> foot
///   -> arm: [hinge (1, 1, 0)], offset rotated 30 deg about x
///     -> hand
pub fn branching_tree(fixed_base: bool) -> Arc<KinematicTree> {
    let mut arm = joint(
        "arm",
        Some(0),
        [0.0, 0.5, 0.3],
        vec![hinge(Vector3::new(1.0, 1.0, 0.0))],
    );
    arm.offset_quat = UnitQuaternion::from_euler_angles(0.5, 0.0, 0.0);
    let joints = vec![
        joint("root", None, [0.0, 0.0, 0.0], vec![]),
        joint(
            "hip",
            Some(0),
            [0.2, 0.0, 0.0],
            vec![hinge(Vector3::z()), hinge(Vector3::y())],
        ),
        joint(
            "knee",
            Some(1),
            [1.0, 0.0, 0.0],
            vec![hinge(Vector3::x()), slide(Vector3::x())],
        ),
        joint("foot", Some(2), [0.0, 1.0, 0.0], vec![]),
        arm,
        joint("hand", Some(4), [0.0, 0.0, 1.0], vec![]),
    ];
    Arc::new(KinematicTree::new(joints, fixed_base).unwrap())
}

/// A non-neutral pose for `branching_tree`.
pub fn some_pose(tree: &Arc<KinematicTree>) -> State {
    let mut state = State::neutral_pose(Arc::clone(tree));
    state.dof_values = vec![0.3, -0.4, 0.5, 0.2, -0.6];
    if !tree.fixed_base() {
        state.root_pos = Vector3::new(0.1, -0.2, 0.3);
        state.root_quat = UnitQuaternion::from_euler_angles(0.2, -0.1, 0.4);
    }
    state
}

/// Exact 3D observations of every keypoint at `state`.
pub fn observations_3d(state: &State) -> Vec<KeypointObservation> {
    keypoint_positions(state)
        .into_iter()
        .enumerate()
        .map(|(j, pos)| KeypointObservation::position_3d(j, pos, 1.0))
        .collect()
}

pub fn keypoint_positions(state: &State) -> Vec<Vector3<f32>> {
    let mut workspace = quickik::forward::ForwardKinematicsWorkspace::new(&state.kinematic_tree);
    quickik::forward::forward_kinematics(&mut workspace, state).unwrap();
    workspace.keypoint_positions().to_vec()
}

/// Benchmark fixtures (`fixtures.json`, `fixtures_g1.json`).
pub fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark/assets")
}

/// Path to an example body's `body_plan.json`, e.g. `"neuromechfly"`.
pub fn body_plan_path(body: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("examples/assets/{body}/body_plan.json"))
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0, |acc, (x, y)| acc.max((x - y).abs()))
}
