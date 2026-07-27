//! Collects per-frame average fit-residual distributions (3D vs.
//! XYView-observed) for `../plot/plot_2d_comparison.py`'s KDE panel.
//! Computed once, from the Rust API only: every binding (Rust/Python/C++)
//! runs the identical compiled solver, so the fit-quality distribution
//! doesn't depend on which one produced it.

use std::sync::Arc;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::high_level::SequenceSolver;
use quickik::observation::{Mapper3Dto2D, NoMapper, XYView};
use quickik::solver::SolverConfig;

use crate::correctness::build_observations;
use crate::fixtures::{Fixtures, RealFrame};
use crate::twod::observations_2d_xyview;

/// Per-frame average (RMS) 3D distance from the solved pose's FK output to
/// the *original* 3D target, one value per one of `frames` (warm-started
/// sequence, adaptive early stop): the same per-frame quantity
/// `correctness::residual_stats` computes, kept here across every frame
/// instead of reduced further to a single aggregate.
fn per_frame_average_distances<M: Mapper3Dto2D>(
    tree: &Arc<KinematicTree>,
    frames: &[RealFrame],
    config: SolverConfig<M>,
    to_obs: impl Fn(&[[f32; 3]]) -> Vec<quickik::observation::KeypointObservation>,
) -> Vec<f32> {
    let mut sequence_solver: SequenceSolver<M> = SequenceSolver::new(tree.clone(), config);
    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    let mut frame_rms = Vec::with_capacity(frames.len());
    for frame in frames {
        let obs = to_obs(&frame.target_ego);
        let state = sequence_solver.solve_frame(&obs);
        evaluate_fwdkin(&mut workspace, state);
        let sum_sq: f32 = workspace.kpt_positions[1..]
            .iter()
            .zip(&frame.target_ego)
            .map(|(p, &[x, y, z])| (p - Vector3::new(x, y, z)).norm_squared())
            .sum();
        frame_rms.push((sum_sq / frame.target_ego.len() as f32).sqrt());
    }
    frame_rms
}

/// Writes `../plot/results/errors-<body>.json`: per-frame average (RMS) 3D
/// distances (model units) for 3D and XYView observations of the same real
/// mocap frames.
pub fn write_errors_json(tree: &Arc<KinematicTree>, fixtures: &Fixtures, body: &str) {
    let avg_3d = per_frame_average_distances(
        tree,
        &fixtures.real_frames,
        SolverConfig::<NoMapper>::default(),
        build_observations,
    );
    let avg_xyview = per_frame_average_distances(
        tree,
        &fixtures.real_frames,
        SolverConfig {
            mapper: Some(XYView),
            ..SolverConfig::default()
        },
        observations_2d_xyview,
    );

    let results = serde_json::json!({
        "body": body,
        "source": "quickik-rust",
        "note": "Per-frame average (RMS) 3D distance (model units) from the solved \
                 pose's FK output to the original 3D target, one value per real \
                 mocap frame (warm-started sequence, adaptive early stop). Computed \
                 once from the Rust API -- every binding runs the identical compiled \
                 solver, so this doesn't depend on which one solved it.",
        "3d": avg_3d,
        "xyview": avg_xyview,
    });
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plot/results");
    std::fs::create_dir_all(&out_dir).expect("failed to create ../plot/results");
    let out_path = out_dir.join(format!("errors-{body}.json"));
    std::fs::write(&out_path, serde_json::to_string_pretty(&results).unwrap())
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
    println!("Wrote error distributions to {}", out_path.display());
}
