//! Correctness cross-check against flygym.ik, using quickik's Rust API only.

use std::sync::Arc;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::high_level::SequenceSolver;
use quickik::observation::{KeypointObservation, Mapper3Dto2D, XYView};
use quickik::solver::{Solver, SolverConfig};
use quickik::state::State;

use crate::fixtures::{Fixtures, RealFrame, SyntheticFrame};

/// `target_ego` covers every joint but the free-floating root (which has no
/// mocap keypoint of its own): prepend `Missing` for it.
pub(crate) fn build_observations(target_ego: &[[f32; 3]]) -> Vec<KeypointObservation> {
    let mut obs = Vec::with_capacity(target_ego.len() + 1);
    obs.push(KeypointObservation::Missing);
    obs.extend(
        target_ego
            .iter()
            .map(|&[x, y, z]| KeypointObservation::Position3D {
                obs_pos: Vector3::new(x, y, z),
                weight: 1.0,
            }),
    );
    obs
}

fn residual_stats(a: &[Vector3<f32>], b: &[[f32; 3]]) -> (f32, f32) {
    // a has one entry per joint (root included); b covers only leg joints
    // (root excluded), so compare a[1..] against b.
    let dists: Vec<f32> = a[1..]
        .iter()
        .zip(b)
        .map(|(p, &[x, y, z])| (p - Vector3::new(x, y, z)).norm())
        .collect();
    let rms = (dists.iter().map(|d| d * d).sum::<f32>() / dists.len() as f32).sqrt();
    let max = dists.iter().cloned().fold(0.0f32, f32::max);
    (rms, max)
}

fn angle_error_deg(solved: &[f32], ground_truth: &[f32]) -> f32 {
    solved
        .iter()
        .zip(ground_truth)
        .map(|(&s, &g)| {
            let d = s - g;
            let wrapped = (d + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI)
                - std::f32::consts::PI;
            wrapped.abs().to_degrees()
        })
        .fold(0.0f32, f32::max)
}

/// Bug-hunt test: feed keypoint targets that are *exactly* reachable (they
/// were generated from this same model's own forward kinematics, driven by
/// real recorded ground-truth joint angles -- see
/// `scripts/generate_fixtures.py`) and check that `Solver` both converges to
/// near-zero residual and recovers the known angles.
pub fn run_synthetic_frame_tests(tree: &Arc<KinematicTree>, frames: &[SyntheticFrame]) {
    println!("== Synthetic exact-fit frames (bug hunt) ==");
    println!(
        "{:>6} {:>16} {:>16} {:>18} {:>18}",
        "frame", "kpt rms", "kpt max", "angle err deg", "angle err deg (w=0)"
    );

    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    let mut default_solver: Solver = Solver::new(tree, SolverConfig::default());
    let mut zero_reg_solver: Solver = Solver::new(
        tree,
        SolverConfig {
            weight: 0.0,
            ..SolverConfig::default()
        },
    );

    for (i, frame) in frames.iter().enumerate() {
        let obs = build_observations(&frame.target_ego);
        let ground_truth = frame.ground_truth_dof_angles_flat();

        let mut state = State::neutral_pose(tree.clone());
        default_solver.solve(&mut state, &obs);
        evaluate_fwdkin(&mut workspace, &state);
        let (rms, max) = residual_stats(&workspace.kpt_positions, &frame.target_ego);
        let angle_err = angle_error_deg(&state.dof_angles, &ground_truth);

        let mut state0 = State::neutral_pose(tree.clone());
        zero_reg_solver.solve(&mut state0, &obs);
        let angle_err0 = angle_error_deg(&state0.dof_angles, &ground_truth);

        println!(
            "{:>6} {rms:>16.6} {max:>16.6} {angle_err:>18.4} {angle_err0:>18.6}",
            i
        );
    }
    println!(
        "(kpt rms/max: 3D distance to target, model units. angle err: max abs error over all \
         {} DOFs, degrees, mod 2*pi. \"w=0\" = weight=0, isolating solver/FK \
         correctness from the intentional regularization bias.)\n",
        tree.state_dim()
    );
}

/// Real-data cross-solver test: feed real (noisy) mocap keypoints, warm
/// started frame-to-frame like real usage, and check quickik's fit quality.
/// When `frames` also carries a reference solver's reconstruction (currently
/// only NeuroMechFly/flygym.ik, via `RealFrame::flygym_ik_reconstructed_ego`),
/// also compares quickik's reconstructed keypoints against it.
pub fn run_real_frame_tests(tree: &Arc<KinematicTree>, frames: &[RealFrame]) {
    println!("== Real mocap frames (cross-solver vs. flygym.ik) ==");

    let mut sequence_solver: SequenceSolver =
        SequenceSolver::new(tree.clone(), SolverConfig::default());
    let mut workspace = ForwardKinematicsWorkspace::new(tree);

    let mut quickik_rms_all = Vec::new();
    let mut quickik_max_all = Vec::new();
    let mut cross_rms_all = Vec::new();
    let mut cross_max_all = Vec::new();

    for frame in frames {
        let obs = build_observations(&frame.target_ego);
        let state = sequence_solver.solve_frame(&obs);
        evaluate_fwdkin(&mut workspace, state);

        let (rms, max) = residual_stats(&workspace.kpt_positions, &frame.target_ego);
        quickik_rms_all.push(rms);
        quickik_max_all.push(max);

        if let Some(flygym_ik_reconstructed_ego) = &frame.flygym_ik_reconstructed_ego {
            let (cross_rms, cross_max) =
                residual_stats(&workspace.kpt_positions, flygym_ik_reconstructed_ego);
            cross_rms_all.push(cross_rms);
            cross_max_all.push(cross_max);
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let rms_of = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let max_of = |v: &[f32]| v.iter().cloned().fold(0.0f32, f32::max);

    println!("over {} frames:", frames.len());
    println!(
        "  quickik fit residual to target:      rms={:.5}  mean={:.5}  max={:.5}",
        rms_of(&quickik_rms_all),
        mean(&quickik_rms_all),
        max_of(&quickik_max_all)
    );
    if cross_rms_all.is_empty() {
        println!("  cross-solver agreement: no reference reconstruction in fixtures, skipped\n");
    } else {
        println!(
            "  cross-solver agreement (vs flygym.ik): rms={:.5}  mean={:.5}  max={:.5}\n",
            rms_of(&cross_rms_all),
            mean(&cross_rms_all),
            max_of(&cross_max_all)
        );
    }
}

pub fn run_all(tree: &Arc<KinematicTree>, fixtures: &Fixtures) {
    run_synthetic_frame_tests(tree, &fixtures.synthetic_frames);
    run_real_frame_tests(tree, &fixtures.real_frames);
}

/// Same bug-hunt check as [`run_synthetic_frame_tests`], but the solver only
/// ever sees `to_2d`'s projection of each frame's exactly-reachable target.
/// Fit quality is still measured in 3D -- the distance between the solved
/// pose's FK output and the *original* 3D target -- since that's the
/// physical quantity that matters, even though the solver never saw it.
pub fn run_synthetic_frame_tests_2d<M: Mapper3Dto2D>(
    tree: &Arc<KinematicTree>,
    frames: &[SyntheticFrame],
    mapper: M,
    label: &str,
    to_2d: impl Fn(&[[f32; 3]]) -> Vec<KeypointObservation>,
) {
    println!("== Synthetic exact-fit frames (bug hunt), 2D via {label} ==");
    println!(
        "{:>6} {:>16} {:>16} {:>18} {:>18}",
        "frame", "kpt rms", "kpt max", "angle err deg", "angle err deg (w=0)"
    );

    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    let mut default_solver: Solver<M> = Solver::new(
        tree,
        SolverConfig {
            mapper: Some(mapper),
            ..SolverConfig::default()
        },
    );
    let mut zero_reg_solver: Solver<M> = Solver::new(
        tree,
        SolverConfig {
            weight: 0.0,
            mapper: Some(mapper),
            ..SolverConfig::default()
        },
    );

    for (i, frame) in frames.iter().enumerate() {
        let obs = to_2d(&frame.target_ego);
        let ground_truth = frame.ground_truth_dof_angles_flat();

        let mut state = State::neutral_pose(tree.clone());
        default_solver.solve(&mut state, &obs);
        evaluate_fwdkin(&mut workspace, &state);
        let (rms, max) = residual_stats(&workspace.kpt_positions, &frame.target_ego);
        let angle_err = angle_error_deg(&state.dof_angles, &ground_truth);

        let mut state0 = State::neutral_pose(tree.clone());
        zero_reg_solver.solve(&mut state0, &obs);
        let angle_err0 = angle_error_deg(&state0.dof_angles, &ground_truth);

        println!(
            "{:>6} {rms:>16.6} {max:>16.6} {angle_err:>18.4} {angle_err0:>18.6}",
            i
        );
    }
    println!(
        "(kpt rms/max: 3D distance between solved FK output and the *original 3D* target -- \
         the solver itself never sees 3D, only its {label} projection. angle err: max abs \
         error over all {} DOFs, degrees, mod 2*pi. \"w=0\" = weight=0.)\n",
        tree.state_dim()
    );
}

/// Same real-data check as [`run_real_frame_tests`], but the solver only ever
/// sees `to_2d`'s projection of each frame's target.
pub fn run_real_frame_tests_2d<M: Mapper3Dto2D>(
    tree: &Arc<KinematicTree>,
    frames: &[RealFrame],
    mapper: M,
    label: &str,
    to_2d: impl Fn(&[[f32; 3]]) -> Vec<KeypointObservation>,
) {
    println!("== Real mocap frames (cross-solver vs. flygym.ik), 2D via {label} ==");

    let mut sequence_solver: SequenceSolver<M> = SequenceSolver::new(
        tree.clone(),
        SolverConfig {
            mapper: Some(mapper),
            ..SolverConfig::default()
        },
    );
    let mut workspace = ForwardKinematicsWorkspace::new(tree);

    let mut quickik_rms_all = Vec::new();
    let mut quickik_max_all = Vec::new();
    let mut cross_rms_all = Vec::new();
    let mut cross_max_all = Vec::new();

    for frame in frames {
        let obs = to_2d(&frame.target_ego);
        let state = sequence_solver.solve_frame(&obs);
        evaluate_fwdkin(&mut workspace, state);

        let (rms, max) = residual_stats(&workspace.kpt_positions, &frame.target_ego);
        quickik_rms_all.push(rms);
        quickik_max_all.push(max);

        if let Some(flygym_ik_reconstructed_ego) = &frame.flygym_ik_reconstructed_ego {
            let (cross_rms, cross_max) =
                residual_stats(&workspace.kpt_positions, flygym_ik_reconstructed_ego);
            cross_rms_all.push(cross_rms);
            cross_max_all.push(cross_max);
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let rms_of = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let max_of = |v: &[f32]| v.iter().cloned().fold(0.0f32, f32::max);

    println!("over {} frames:", frames.len());
    println!(
        "  quickik fit residual to target:      rms={:.5}  mean={:.5}  max={:.5}",
        rms_of(&quickik_rms_all),
        mean(&quickik_rms_all),
        max_of(&quickik_max_all)
    );
    if cross_rms_all.is_empty() {
        println!("  cross-solver agreement: no reference reconstruction in fixtures, skipped\n");
    } else {
        println!(
            "  cross-solver agreement (vs flygym.ik): rms={:.5}  mean={:.5}  max={:.5}\n",
            rms_of(&cross_rms_all),
            mean(&cross_rms_all),
            max_of(&cross_max_all)
        );
    }
}

/// Runs the 2D-observation correctness suite via [`XYView`], on the same
/// fixtures used by [`run_all`].
pub fn run_all_2d(tree: &Arc<KinematicTree>, fixtures: &Fixtures) {
    run_synthetic_frame_tests_2d(tree, &fixtures.synthetic_frames, XYView, "XYView", |t| {
        crate::twod::observations_2d_xyview(t)
    });
    run_real_frame_tests_2d(tree, &fixtures.real_frames, XYView, "XYView", |t| {
        crate::twod::observations_2d_xyview(t)
    });
}
