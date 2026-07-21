# FastIK

Fast and minimalist inverse kinematics library aimed for both high throughput and low latency.

FastIK provides high-level APIs for processing consecutive frames (i.e. with warm starts) and multi-threaded batch processing, as well as low-level APIs for more specific use cases (e.g. real-time application).

FastIK also comes with Python and C++ bindings.


## Installation

### Rust

```toml
[dependencies]
fastik = { git = "https://github.com/sibocw/fastik" }
```

Or, for local development against a clone:

```toml
[dependencies]
fastik = { path = "../fastik" }
```

### Python

The Python bindings (in `python/`) are a [PyO3](https://pyo3.rs)/[maturin](https://github.com/PyO3/maturin) extension module, so they build from source and require a Rust toolchain plus Python >= 3.8.

```bash
pip install "git+https://github.com/sibocw/fastik.git#subdirectory=python"
# or, with uv:
uv pip install "git+https://github.com/sibocw/fastik.git#subdirectory=python"
```

For local development (editable install that rebuilds the Rust extension in place):

```bash
git clone https://github.com/sibocw/fastik
cd fastik/python
pip install maturin
maturin develop --release
```


## Examples

### Quick start

A body plan is a kinematic tree of joints loaded from JSON, where every joint also doubles as a tracking keypoint:

```json
{
    "joints": [
        {"name": "root", "parent": null, "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []},
        {"name": "elbow", "parent": "root", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "neutral_angle": 0.0, "limits": [-3.0, 3.0]}]},
        {"name": "wrist", "parent": "elbow", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []}
    ]
}
```

`parent` is a joint name (`null` for the root). `offset_pos`/`offset_quat` place a joint relative to its parent, and `dofs` are its rotational degrees of freedom (axis, neutral angle, optional `[min, max]` limits).

Solve one frame with `Solver`, giving one `KeypointObservation` per keypoint (`Missing`, `Position3D`, or `Position2D`) in `kinematic_tree.joints` order:

```rust
use std::sync::Arc;
use fastik::body_plan::KinematicTree;
use fastik::observation::KeypointObservation;
use fastik::solver::{Solver, SolverConfig};
use fastik::state::State;
use nalgebra::Vector3;

let kinematic_tree = Arc::new(KinematicTree::from_json_file("body_plan.json"));
let mut state = State::neutral_pose(kinematic_tree.clone());
let mut solver: Solver = Solver::new(&kinematic_tree, SolverConfig::default());

let observations = vec![
    KeypointObservation::Position3D { obs_pos: Vector3::new(0.0, 0.0, 0.0), weight: 1.0 },
    KeypointObservation::Position3D { obs_pos: Vector3::new(1.0, 0.0, 0.0), weight: 1.0 },
    KeypointObservation::Position3D { obs_pos: Vector3::new(1.0, 1.0, 0.0), weight: 1.0 },
];
solver.solve(&mut state, &observations);
println!("{:?}", state.dof_angles);
```

```python
import fastik

kinematic_tree = fastik.KinematicTree.from_json_file("body_plan.json")
state = fastik.State.neutral_pose(kinematic_tree)
solver = fastik.Solver(kinematic_tree, fastik.SolverConfig())

observations = [
    fastik.KeypointObservation.position_3d((0.0, 0.0, 0.0), 1.0),
    fastik.KeypointObservation.position_3d((1.0, 0.0, 0.0), 1.0),
    fastik.KeypointObservation.position_3d((1.0, 1.0, 0.0), 1.0),
]
solver.solve(state, observations)
print(state.dof_angles)
```

`SolverConfig` bundles the iteration count, damping, regularization weight, and convergence tolerance -- set via `Solver::new`/`Solver(...)`, though `solver.config` stays mutable for retuning between calls (Python: `solver.config` is the same object every time, so `solver.config.n_iterations = 5` takes effect on the next `solve`, just like Rust).

### Sequences

`fastik::high_level` (Python: same names, flat in the `fastik` module) builds on `Solver` for continuous recordings: `SequenceSolver` keeps a `Solver` and `State` together and warm-starts each frame from the last (`solve_frame`/`solve_sequence`); `solve_sequence_segmented_parallel` solves a single long sequence in parallel by splitting it into overlapping, warm-started segments. Independent sequences (e.g. one per subject) can just be solved with their own `SequenceSolver`s, parallelized however you like. See that module's docs for details.

```python
seq_solver = fastik.SequenceSolver(kinematic_tree, fastik.SolverConfig())
for frame_observations in recording:
    pose = seq_solver.solve_frame(frame_observations)

segmented_config = fastik.SegmentedSolveConfig(segment_len=200, overlap_len=20, overlap_tolerance=0.05)
poses = fastik.solve_sequence_segmented_parallel(kinematic_tree, fastik.SolverConfig(), long_recording, segmented_config)
```

### 2D observations

For keypoints given as 2D pixel coordinates, set `SolverConfig::mapper` to a `Camera` (or `XYView` for keypoints already reprojected to physical X-Y coordinates):

```rust
use fastik::observation::Camera;
use nalgebra::Matrix3;

let camera = Camera {
    fx: 800.0,
    fy: 800.0,
    cx: 320.0,
    cy: 240.0,
    world2cam_pos: Vector3::new(0.0, 0.0, 5.0),
    world2cam_rot_mat: Matrix3::identity(),
};
let config = SolverConfig { mapper: Some(camera), ..SolverConfig::default() };
let mut solver: Solver<Camera> = Solver::new(&kinematic_tree, config);
```

```python
camera = fastik.Camera(
    fx=800.0, fy=800.0, cx=320.0, cy=240.0,
    world2cam_pos=(0.0, 0.0, 5.0),
    world2cam_rot_mat=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),  # row-major 3x3
)
solver = fastik.Solver(kinematic_tree, fastik.SolverConfig(), mapper=camera)
```

Rust's `Solver<M>` is generic over the mapper type at compile time; Python has no equivalent, so every `Solver`/`SequenceSolver` is backed by a single mapper enum chosen at runtime -- pass `mapper=None` (the default), a `Camera`, or an `XYView()` as a keyword argument to `Solver`/`SequenceSolver`/`solve_sequence_segmented_parallel`. It's deliberately not part of `SolverConfig`: like Rust's `M`, it's fixed for the solver's lifetime (read-only `solver.mapper` property, no setter), whereas `SolverConfig`'s other fields stay freely mutable. Errors from malformed input (bad JSON, wrong-sized vectors, a `Position2D` observation with no mapper set) raise Python exceptions rather than crashing.


## Benchmarks
TODO