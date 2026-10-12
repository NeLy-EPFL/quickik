//! Example fitting DOF angles from observations in 2D orthographic view.
//!
//! Bodies:
//!   neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
//!   g1: Unitree G1 robot (https://www.unitree.com/g1/)
//!
//! Input data:
//!   Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
//!   Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
//!   (only XY positions used)
//!
//! To run this example (from anywhere in the repository):
//!   cargo run --release -p quickik --example ortho2d -- <body>
//! where <body> is one of "neuromechfly" or "g1". Results will be saved to
//! examples/rust/output/ortho2d_<body>.json
//!
//! To plot the results, run
//!   uv run examples/plot_results.py examples/rust/output/ortho2d_<body>.json
//! and a video will be saved under the same directory. Run the `from_3d`
//! example first to also show its fit for reference.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nalgebra::{Vector2, Vector3};
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, forward_kinematics};
use quickik::observation::{
    KeypointObservation, ObservationSequence, OrthographicViewXY, Projection,
};
use quickik::sequential_solver::SequentialSolver;
use quickik::solver::SolverConfig;
use quickik::state::State;

// Data utilities for the specific demos
mod util;
use util::{CameraView, IkOutput, Trajectory, save_output};

const IMAGE_SIZE: [f32; 2] = [640.0, 480.0];

fn run_ik_from_ortho2d_obs(
    body_plan_path: &Path,
    trajectory_path: &Path,
    scale: f32,
    prior_weight: f32,
    on_ground: bool,
) -> quickik::Result<IkOutput> {
    // ===== Step 1: Load the body plan ========================================
    let tree = Arc::new(KinematicTree::from_json_file(body_plan_path)?);

    // ===== Step 2: Load the motion trajectory ================================
    // This is the part that you would replace with your own data. For these examples, utilities
    // to load the trajectory, quantify errors, and save results are provided in the util module.
    let trajectory = Trajectory::load(trajectory_path)?;
    let joint_indices = trajectory.joint_indices(&tree);
    // We emulate 2D observations by dropping the Z coordinate (orthographic projection). In real
    // applications, these would come directly from a 2D pose model.
    let view = emulate_bottom_view(&trajectory, scale);
    // Make a vector of frames, where each frame is a vector of KeypointObservations, each
    // carrying the projection it was observed in. Pixel errors are `scale` times larger than
    // position errors, so the weight is scaled down to match from_3d.
    let weight = 1.0 / (scale * scale);
    let camera = &view.projection;
    let frames: Vec<Vec<KeypointObservation>> = view
        .observed
        .iter()
        .map(|uvs| {
            let pairs = joint_indices.iter().zip(uvs);
            pairs
                .map(|(&joint, &uv)| {
                    KeypointObservation::position_2d(joint, uv, Arc::clone(camera), weight)
                })
                .collect()
        })
        .collect();

    // ===== Step 3: Configure the solver ======================================
    // With height unobserved, the neutral pose is the main thing keeping each keypoint at a
    // plausible height, so the neutral prior is stronger than in from_3d.
    let mut config = SolverConfig::default();
    config.add_dof_neutral_prior(2.0 * prior_weight)?; // favor natural poses
    config.add_dof_smoothness_prior(prior_weight)?; // favor smooth motion
    // The body's absolute height can't be recovered, so it stays at the initial pose's height.
    let initial_state = if on_ground {
        neutral_pose_on_ground(&tree)?
    } else {
        State::neutral_pose(Arc::clone(&tree))
    };

    // ===== Step 4: Solve =====================================================
    let mut solver = SequentialSolver::new(&tree, &config, Some(initial_state))?;
    let start = Instant::now();
    // Set with_grad to false to skip gradient tracking
    // Set with_fk to true to have forward kinematics (fitted keypoint positions) in the results
    let results = solver.solve(&ObservationSequence::from_lists(frames), false, true)?;
    let solve_time = start.elapsed().as_secs_f64();

    Ok(IkOutput {
        tree,
        trajectory,
        joint_indices,
        results,
        solve_time,
        views: vec![view],
    })
}

// ===== Main and helper functions below (unrelated to core QuickIK logic) =====

/// Emulates a camera below the body with its image axes along world X and Y, at `scale` pixels
/// per unit length (`uv = scale * (x, y) + offset`), centered on the trajectory's mean position.
fn emulate_bottom_view(trajectory: &Trajectory, scale: f32) -> CameraView {
    let all_positions = trajectory.positions.iter().flatten();
    let n_positions = all_positions.clone().count() as f32;
    let center = all_positions.sum::<Vector3<f32>>().xy() / n_positions;
    let offset = Vector2::from(IMAGE_SIZE) / 2.0 - scale * center;
    let projection = Arc::new(Projection::from(OrthographicViewXY::new(scale, offset)));
    let frames = trajectory.positions.iter();
    let observed = frames
        .map(|positions| positions.iter().map(|p| projection.project(p).0).collect())
        .collect();
    CameraView {
        name: "bottom view".to_string(),
        image_size: IMAGE_SIZE,
        projection,
        observed,
    }
}

/// The neutral pose, lifted so that its lowest keypoint is on the ground (z = 0).
fn neutral_pose_on_ground(tree: &Arc<KinematicTree>) -> quickik::Result<State> {
    let mut state = State::neutral_pose(Arc::clone(tree));
    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    forward_kinematics(&mut workspace, &state)?;
    let positions = workspace.keypoint_positions().iter();
    state.root_pos.z = -positions.map(|p| p.z).fold(f32::INFINITY, f32::min);
    Ok(state)
}

fn main() -> quickik::Result<()> {
    let body = std::env::args()
        .nth(1)
        .expect("usage: ortho2d <neuromechfly|g1>");
    // Body-specific settings: image scale (pixels per unit length), prior weight (depends on the
    // units), and whether the recording has the body on the ground at z = 0 (G1) rather than
    // centered on its root (the fly's thorax).
    let (scale, prior_weight, on_ground) = match body.as_str() {
        "g1" => (130.0, 1e-2, true),            // 130 px/m
        "neuromechfly" => (100.0, 1e-3, false), // 100 px/mm
        _ => panic!("Unknown body: {body}"),
    };
    let assets = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/assets")
        .join(&body);

    let output = run_ik_from_ortho2d_obs(
        &assets.join("body_plan.json"),
        &assets.join("example_trajectory.json"),
        scale,
        prior_weight,
        on_ground,
    )?;

    // Compare with the recorded 3D positions, which the solver never saw.
    let n_errors = output.errors().count() as f32;
    let error_xy = output.errors().map(|e| e.xy().norm()).sum::<f32>() / n_errors;
    let error_z = output.errors().map(|e| e.z.abs()).sum::<f32>() / n_errors;
    println!(
        "Fitted {} frames in {:.1} ms, mean keypoint error vs. 3D recording: {error_xy:.4} {units} \
         in XY, {error_z:.4} {units} in Z",
        output.results.len(),
        output.solve_time * 1e3,
        units = output.trajectory.units
    );
    save_output("ortho2d", &body, &output)
}
