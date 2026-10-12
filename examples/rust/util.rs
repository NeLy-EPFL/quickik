//! Helpers shared by the Rust examples: loading the recorded trajectory, and
//! summarizing and saving results in the format `examples/plot_results.py`
//! reads. Nothing here is specific to QuickIK beyond reading its results.

#![allow(dead_code)] // not every example uses every helper

use std::path::Path;
use std::sync::Arc;

use nalgebra::{Vector2, Vector3};
use quickik::body_plan::KinematicTree;
use quickik::observation::Projection;
use quickik::solver::SolverResult;
use serde_json::json;

/// A recorded keypoint trajectory, as stored in `example_trajectory.json`.
#[derive(serde::Deserialize)]
pub struct Trajectory {
    pub units: String,
    pub fps: f32,
    pub joint_names: Vec<String>,
    /// `positions[frame][i]` is the position of keypoint `joint_names[i]`.
    pub positions: Vec<Vec<Vector3<f32>>>,
}

impl Trajectory {
    pub fn load(path: &Path) -> quickik::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// The body plan's joint index for each recorded keypoint.
    pub fn joint_indices(&self, tree: &KinematicTree) -> Vec<usize> {
        let names = self.joint_names.iter();
        names.map(|name| tree.joint_index(name).unwrap()).collect()
    }
}

/// A camera's projection and what it observed.
pub struct CameraView {
    pub name: String,
    pub image_size: [f32; 2],
    pub projection: Arc<Projection>,
    /// `observed[frame][i]` is the observed pixel position of the trajectory's
    /// i-th keypoint.
    pub observed: Vec<Vec<Vector2<f32>>>,
}

/// Everything an example's IK run produces, for reporting and saving.
pub struct IkOutput {
    pub tree: Arc<KinematicTree>,
    pub trajectory: Trajectory,
    /// The body plan's joint index for each recorded keypoint.
    pub joint_indices: Vec<usize>,
    /// One result per frame.
    pub results: Vec<SolverResult>,
    /// Time spent in the solver, in seconds.
    pub solve_time: f64,
    /// The cameras the observations came from (empty for 3D observations).
    pub views: Vec<CameraView>,
}

impl IkOutput {
    /// Fitted minus recorded position of every recorded keypoint in every frame.
    pub fn errors(&self) -> impl Iterator<Item = Vector3<f32>> + '_ {
        let frames = self.results.iter().zip(&self.trajectory.positions);
        frames.flat_map(|(result, positions)| {
            let fitted = result.keypoint_pos.as_ref().unwrap();
            let pairs = self.joint_indices.iter().zip(positions);
            pairs.map(|(&joint, pos)| fitted[joint] - pos)
        })
    }

    /// Mean distance between fitted and recorded keypoints, over all frames.
    pub fn mean_error(&self) -> f32 {
        let distances: Vec<f32> = self.errors().map(|e| e.norm()).collect();
        distances.iter().sum::<f32>() / distances.len() as f32
    }
}

/// Writes the inputs and results to `examples/rust/output/<example>_<body>.json`
/// in the format `examples/plot_results.py` reads (shared by all languages),
/// with every number rounded to 3 decimal places.
pub fn save_output(example: &str, body: &str, output: &IkOutput) -> quickik::Result<()> {
    let IkOutput {
        tree,
        trajectory,
        joint_indices,
        results,
        ..
    } = output;
    let mut json = json!({
        "example": example,
        "body": body,
        "language": "rust",
        "units": trajectory.units,
        "fps": trajectory.fps,
        "joint_names": tree.joint_names().collect::<Vec<_>>(),
        "parents": tree.joints().iter().map(|j| j.parent).collect::<Vec<_>>(),
        "observations": {"joint_indices": joint_indices, "positions": trajectory.positions},
        "fit": {
            "root_pos": results.iter().map(|r| r.state.root_pos).collect::<Vec<_>>(),
            "root_quat_wxyz": results.iter().map(|r| {
                let q = r.state.root_quat;
                [q.w, q.i, q.j, q.k]
            }).collect::<Vec<_>>(),
            "dof_values": results.iter().map(|r| &r.state.dof_values).collect::<Vec<_>>(),
            "keypoint_positions": results.iter().map(|r| r.keypoint_pos.as_ref().unwrap()).collect::<Vec<_>>(),
        },
        "views": output.views.iter().map(view_json).collect::<Vec<_>>(),
        "solve_time_s": output.solve_time,
    });
    round_numbers(&mut json);
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("examples/rust/output/{example}_{body}.json"));
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, json.to_string())?;
    println!("Saved {}", path.display());
    Ok(())
}

fn view_json(view: &CameraView) -> serde_json::Value {
    let projection = match view.projection.as_ref() {
        Projection::PinholeCamera(camera) => json!({
            "type": "pinhole_camera",
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "world2cam_pos": camera.world2cam_pos,
            "world2cam_rot_mat": camera.world2cam_rot_mat.row_iter().map(|r| [r[0], r[1], r[2]]).collect::<Vec<_>>(),
        }),
        Projection::OrthographicViewXY(ortho) => json!({
            "type": "orthographic_view_xy",
            "scale": ortho.scale,
            "offset": ortho.offset,
        }),
    };
    json!({
        "name": view.name,
        "image_size": view.image_size,
        "projection": projection,
        "observed": view.observed,
    })
}

/// Rounds every non-integer number in `value` to 3 decimal places.
fn round_numbers(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Number(number) if number.is_f64() => {
            let x = number.as_f64().unwrap();
            *value = json!((x * 1e3).round() / 1e3);
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(round_numbers),
        serde_json::Value::Object(fields) => fields.values_mut().for_each(round_numbers),
        _ => {}
    }
}
