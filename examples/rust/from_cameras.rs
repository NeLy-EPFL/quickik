//! Example fitting DOF angles from 2D keypoint observations from two cameras.
//!
//! Bodies:
//!   neuromechfly: a fly biomechanical model (https://neuromechfly.org/)
//!   g1: Unitree G1 robot (https://www.unitree.com/g1/)
//!
//! Input data:
//!   Body plans: examples/assets/{g1,neuromechfly}/body_plan.json
//!   Trajectories: examples/assets/{g1,neuromechfly}/example_trajectory.json
//!   (2D observations are emulated by projecting the 3D positions into two
//!   pinhole cameras)
//!
//! To run this example (from anywhere in the repository):
//!   cargo run --release -p quickik --example from_cameras -- <body>
//! where <body> is one of "neuromechfly" or "g1". Results will be saved to
//! examples/rust/output/from_cameras_<body>.json
//!
//! To plot the results, run
//!   uv run examples/plot_results.py examples/rust/output/from_cameras_<body>.json
//! and a video will be saved under the same directory. Run the `from_3d`
//! example first to also show its fit for reference.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nalgebra::{Matrix3, Vector3};
use quickik::body_plan::KinematicTree;
use quickik::observation::{KeypointObservation, ObservationSequence, PinholeCamera, Projection};
use quickik::sequential_solver::SequentialSolver;
use quickik::solver::SolverConfig;

// Data utilities for the specific demos
mod util;
use util::{CameraView, IkOutput, Trajectory, save_output};

fn run_ik_from_camera_obs(
    body_plan_path: &Path,
    trajectory_path: &Path,
    distance: f32,
    focal_length: f32,
    image_size: [f32; 2],
    prior_weight: f32,
) -> quickik::Result<IkOutput> {
    // ===== Step 1: Load the body plan ========================================
    let tree = Arc::new(KinematicTree::from_json_file(body_plan_path)?);

    // ===== Step 2: Load the motion trajectory ================================
    // This is the part that you would replace with your own data. For these examples, utilities
    // to load the trajectory, quantify errors, and save results are provided in the util module.
    let trajectory = Trajectory::load(trajectory_path)?;
    let joint_indices = trajectory.joint_indices(&tree);
    // We emulate 2D observations by projecting 3D positions into two cameras. In real
    // applications, these would come directly from a 2D pose model.
    let views = emulate_two_camera_views(&trajectory, distance, focal_length, image_size);
    // Make a vector of frames, where each frame is a vector of KeypointObservations. Each frame's
    // observations are one flat list mixing both cameras, each observation carrying the
    // projection (camera) it was observed in. Pixel errors are about focal_length / distance
    // times larger than position errors, so the weight is scaled down to match from_3d.
    let weight = (distance / focal_length).powi(2);
    let frames: Vec<Vec<KeypointObservation>> = (0..trajectory.positions.len())
        .map(|t| {
            let observations = views.iter().flat_map(|view| {
                let camera = &view.projection;
                let pairs = joint_indices.iter().zip(&view.observed[t]);
                pairs.map(move |(&joint, &uv)| {
                    KeypointObservation::position_2d(joint, uv, Arc::clone(camera), weight)
                })
            });
            observations.collect()
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
        views,
    })
}

// ===== Main and helper functions below (unrelated to core QuickIK logic) =====

/// Emulates two cameras, 45 degrees to the front left and front right of the body (which faces
/// +x), at `distance` from the trajectory's average position and looking horizontally at it.
fn emulate_two_camera_views(
    trajectory: &Trajectory,
    distance: f32,
    focal_length: f32,
    image_size: [f32; 2],
) -> Vec<CameraView> {
    let all_positions = trajectory.positions.iter().flatten();
    let n_positions = all_positions.clone().count() as f32;
    let target = all_positions.sum::<Vector3<f32>>() / n_positions;
    [("left camera", 45.0), ("right camera", -45.0)]
        .into_iter()
        .map(|(name, azimuth_deg)| {
            let camera =
                make_pinhole_camera(target, distance, azimuth_deg, focal_length, image_size);
            let projection = Arc::new(Projection::from(camera));
            let frames = trajectory.positions.iter();
            let observed = frames
                .map(|positions| positions.iter().map(|p| projection.project(p).0).collect())
                .collect();
            CameraView {
                name: name.to_string(),
                image_size,
                projection,
                observed,
            }
        })
        .collect()
}

/// A pinhole camera `distance` away from `target` at the same height, at the given azimuth
/// (counterclockwise from +x, seen from above), facing `target` with it at the image center.
fn make_pinhole_camera(
    target: Vector3<f32>,
    distance: f32,
    azimuth_deg: f32,
    focal_length: f32,
    image_size: [f32; 2],
) -> PinholeCamera {
    let azimuth = azimuth_deg.to_radians();
    let direction = Vector3::new(azimuth.cos(), azimuth.sin(), 0.0);
    let position = target + distance * direction;

    // Camera axes in world coordinates: z forward (toward the target), x to the right in the
    // image, y down in the image.
    let z = -direction;
    let x = z.cross(&Vector3::z()).normalize();
    let y = z.cross(&x);
    let world2cam_rot_mat = Matrix3::from_rows(&[x.transpose(), y.transpose(), z.transpose()]);
    let world2cam_pos = -(world2cam_rot_mat * position);
    let [width, height] = image_size;
    let (cx, cy) = (width / 2.0, height / 2.0);
    PinholeCamera::new(
        focal_length,
        focal_length,
        cx,
        cy,
        world2cam_pos,
        world2cam_rot_mat,
    )
}

fn main() -> quickik::Result<()> {
    let body = std::env::args()
        .nth(1)
        .expect("usage: from_cameras <neuromechfly|g1>");
    // Body-specific settings: camera distance, focal length and image size in pixels (chosen so
    // the whole recording just fits in the image), and prior weight (depends on the units).
    let (distance, focal_length, image_size, prior_weight) = match body.as_str() {
        "g1" => (8.0, 1600.0, [960.0, 320.0], 1e-2), // 8 m away
        "neuromechfly" => (100.0, 13000.0, [640.0, 256.0], 1e-3), // 100 mm away
        _ => panic!("Unknown body: {body}"),
    };
    let assets = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/assets")
        .join(&body);

    let output = run_ik_from_camera_obs(
        &assets.join("body_plan.json"),
        &assets.join("example_trajectory.json"),
        distance,
        focal_length,
        image_size,
        prior_weight,
    )?;

    // Compare with the recorded 3D positions, which the solver never saw.
    println!(
        "Fitted {} frames in {:.1} ms, mean keypoint error vs. 3D recording: {:.4} {}",
        output.results.len(),
        output.solve_time * 1e3,
        output.mean_error(),
        output.trajectory.units
    );
    save_output("from_cameras", &body, &output)
}
