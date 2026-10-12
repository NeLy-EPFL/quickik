//! Example fitting DOF angles from keypoint observations in 3D.
//!
//! Bodies:
//!   neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
//!   g1: Unitree G1 robot (https://www.unitree.com/g1/)
//!
//! Input data:
//!   Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
//!   Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
//!
//! To run this example (from anywhere in the repository):
//!   cargo run --release -p quickik --example from_3d -- <body>
//! where <body> is one of "neuromechfly" or "g1". Results will be saved to
//! examples/rust/output/from_3d_<body>.json
//!
//! To plot the results, run
//!   uv run examples/plot_results.py examples/rust/output/from_3d_<body>.json
//! and a video will be saved under the same directory.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use quickik::body_plan::KinematicTree;
use quickik::observation::{KeypointObservation, ObservationSequence};
use quickik::sequential_solver::SequentialSolver;
use quickik::solver::SolverConfig;

// Data utilities for the specific demos
mod util;
use util::{IkOutput, Trajectory, save_output};

fn run_ik_from_3d_obs(
    body_plan_path: &Path,
    trajectory_path: &Path,
    prior_weight: f32,
) -> quickik::Result<IkOutput> {
    // ===== Step 1: Load the body plan ========================================
    let tree = Arc::new(KinematicTree::from_json_file(body_plan_path)?);

    // ===== Step 2: Load the motion trajectory ================================
    // This is the part that you would replace with your own data. For these examples, utilities
    // to load the trajectory, quantify errors, and save results are provided in the util module.
    let trajectory = Trajectory::load(trajectory_path)?;
    let joint_indices = trajectory.joint_indices(&tree);
    // Make a vector of frames, where each frame is a vector of KeypointObservations.
    // Keypoint positions are given in 3D, so we use KeypointObservation::position_3d.
    let frames: Vec<Vec<KeypointObservation>> = trajectory
        .positions
        .iter()
        .map(|positions| {
            let observations = joint_indices.iter().zip(positions);
            observations
                .map(|(&joint, &pos)| KeypointObservation::position_3d(joint, pos, 1.0))
                .collect()
        })
        .collect();

    // ===== Step 3: Configure the solver ======================================
    let mut config = SolverConfig::default();
    config.add_dof_neutral_prior(prior_weight)?; // favor natural poses
    config.add_dof_smoothness_prior(prior_weight)?; // favor smooth motion

    // ===== Step 4: Solve =====================================================
    let mut solver = SequentialSolver::new(&tree, &config, None)?;
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
        views: vec![],
    })
}

// ===== Main and helper functions below (unrelated to core QuickIK logic) =====

fn main() -> quickik::Result<()> {
    let body = std::env::args()
        .nth(1)
        .expect("usage: from_3d <neuromechfly|g1>");
    // Due to body size (and unit) differences, we assign different prior
    // weights for the two bodies.
    let prior_weight = match body.as_str() {
        "g1" => 1e-2,
        "neuromechfly" => 1e-3,
        _ => panic!("Unknown body: {body}"),
    };
    let assets = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/assets")
        .join(&body);

    let output = run_ik_from_3d_obs(
        &assets.join("body_plan.json"),
        &assets.join("example_trajectory.json"),
        prior_weight,
    )?;

    println!(
        "Fitted {} frames in {:.1} ms, mean keypoint error {:.4} {}",
        output.results.len(),
        output.solve_time * 1e3,
        output.mean_error(),
        output.trajectory.units
    );
    save_output("from_3d", &body, &output)
}
