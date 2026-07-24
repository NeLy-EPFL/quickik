//! Collects pooled per-keypoint 3D fit-residual distributions (3D vs.
//! Camera- vs. XYView-observed) for `../plot/plot_2d_comparison.py`'s error-
//! distribution panel. Computed once, from the Rust API only: every binding
//! (Rust/Python/C++) runs the identical compiled solver, so the fit-quality
//! distribution doesn't depend on which one produced it.

use std::sync::Arc;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::high_level::SequenceSolver;
use quickik::observation::{Camera, Mapper3Dto2D, NoMapper, XYView};
use quickik::solver::SolverConfig;

use crate::correctness::build_observations;
use crate::fixtures::{Fixtures, RealFrame};
use crate::twod::{observations_2d_camera, observations_2d_xyview};

/// Per-keypoint 3D distance from the solved pose's FK output to the
/// *original* 3D target, pooled across every keypoint in every one of
/// `frames` (warm-started sequence, adaptive early stop) -- the same
/// quantity `correctness::residual_stats` reduces to rms/max, kept here as a
/// raw distribution instead.
fn pooled_kpt_distances<M: Mapper3Dto2D>(
    tree: &Arc<KinematicTree>,
    frames: &[RealFrame],
    config: SolverConfig<M>,
    to_obs: impl Fn(&[[f32; 3]]) -> Vec<quickik::observation::KeypointObservation>,
) -> Vec<f32> {
    let mut sequence_solver: SequenceSolver<M> = SequenceSolver::new(tree.clone(), config);
    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    let mut dists = Vec::new();
    for frame in frames {
        let obs = to_obs(&frame.target_ego);
        let state = sequence_solver.solve_frame(&obs);
        evaluate_fwdkin(&mut workspace, state);
        for (p, &[x, y, z]) in workspace.kpt_positions[1..].iter().zip(&frame.target_ego) {
            dists.push((p - Vector3::new(x, y, z)).norm());
        }
    }
    dists
}

/// Writes `../plot/results/errors-<body>.json`: pooled per-keypoint 3D
/// distances (model units) for 3D, XYView, and Camera observations of the
/// same real mocap frames.
pub fn write_errors_json(tree: &Arc<KinematicTree>, fixtures: &Fixtures, body: &str, camera: Camera) {
    let dists_3d = pooled_kpt_distances(
        tree,
        &fixtures.real_frames,
        SolverConfig::<NoMapper>::default(),
        build_observations,
    );
    let dists_xyview = pooled_kpt_distances(
        tree,
        &fixtures.real_frames,
        SolverConfig {
            mapper: Some(XYView),
            ..SolverConfig::default()
        },
        observations_2d_xyview,
    );
    let dists_camera = pooled_kpt_distances(
        tree,
        &fixtures.real_frames,
        SolverConfig {
            mapper: Some(camera),
            ..SolverConfig::default()
        },
        |t| observations_2d_camera(t, &camera),
    );

    let results = serde_json::json!({
        "body": body,
        "source": "quickik-rust",
        "note": "Per-keypoint 3D distance (model units) from the solved pose's FK \
                 output to the original 3D target, pooled across every keypoint in \
                 every real mocap frame (warm-started sequence, adaptive early \
                 stop). Computed once from the Rust API -- every binding runs the \
                 identical compiled solver, so this doesn't depend on which one \
                 solved it.",
        "3d": dists_3d,
        "xyview": dists_xyview,
        "camera": dists_camera,
    });
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plot/results");
    std::fs::create_dir_all(&out_dir).expect("failed to create ../plot/results");
    let out_path = out_dir.join(format!("errors-{body}.json"));
    std::fs::write(&out_path, serde_json::to_string_pretty(&results).unwrap())
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
    println!("Wrote error distributions to {}", out_path.display());
}
