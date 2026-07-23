# Usage

## Body plan

A body plan describes the kinematic tree QuickIK solves against – a robot's joints, or an animal's skeleton – loaded once from JSON and reused across every solve. Unlike a typical robotics kinematic tree, which separates joints (that move) from frames or links (that get tracked), here every joint doubles as a keypoint: its world position is always available as a potential target, whether or not it carries a rotational DOF of its own. This is what lets one uniform `KeypointObservation` list – one entry per joint, in the body plan's own tree order – describe a whole tracked pose, DOF-bearing joints and fixed leaf keypoints alike.

One modeling consequence worth knowing: a joint's own DOF reorients its *children*, not itself – rotating a joint never moves its own keypoint, only the keypoints downstream of it. So every DOF needs at least one keypoint further down its own chain to be observable at all; a chain that ends exactly at its last DOF-bearing joint, with nothing past it, leaves that DOF's angle undetermined by any observation. This is why even a fixed, 0-DOF "tip" joint (a fingertip, a fly's claw, a robot's end effector) is usually worth keeping in the body plan even though it never actuates anything itself.

??? note "Body plan JSON schema"
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

    `parent` is a joint name (`null` for the root). `offset_pos`/`offset_quat` place a joint relative to its parent, and `dofs` are its rotational degrees of freedom (axis, neutral angle, optional `[min, max]` limits). See the [API reference](api/rust/quickik/body_plan/index.html) for the full schema.

## Inverse kinematics on a single frame

QuickIK solves *whole-tree* IK: one `Solver::solve` call takes one `KeypointObservation` per keypoint – `Missing`, `Position3D`, or `Position2D` (see [below](#keypoint-positions-observed-in-2d)) – in `kinematic_tree.joints` order, and jointly fits the free-floating root pose and every joint angle at once against all of them. This is what makes it different from solving each limb as its own small IK problem: a keypoint on one limb can still help constrain the root pose (and therefore every other limb) even if that other limb's own keypoints are all `Missing` this frame.

=== "Rust"

    ```rust
    use std::sync::Arc;
    use quickik::body_plan::KinematicTree;
    use quickik::observation::KeypointObservation;
    use quickik::solver::{Solver, SolverConfig};
    use quickik::state::State;
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

=== "Python"

    ```python
    import quickik

    kinematic_tree = quickik.KinematicTree.from_json_file("body_plan.json")
    state = quickik.State.neutral_pose(kinematic_tree)
    solver = quickik.Solver(kinematic_tree, quickik.SolverConfig())

    observations = [
        quickik.KeypointObservation.position_3d((0.0, 0.0, 0.0), 1.0),
        quickik.KeypointObservation.position_3d((1.0, 0.0, 0.0), 1.0),
        quickik.KeypointObservation.position_3d((1.0, 1.0, 0.0), 1.0),
    ]
    solver.solve(state, observations)
    print(state.dof_angles)
    ```

=== "C++"

    ```cpp
    #include "quickik.h"

    auto tree = quickik::kinematic_tree_from_json_file("body_plan.json");
    auto state = quickik::state_neutral_pose(*tree);
    auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());

    std::vector<quickik::KeypointObservation> observations = {
        quickik::keypoint_position_3d({0.0, 0.0, 0.0}, 1.0),
        quickik::keypoint_position_3d({1.0, 0.0, 0.0}, 1.0),
        quickik::keypoint_position_3d({1.0, 1.0, 0.0}, 1.0),
    };
    solver->solve(*state, rust::Slice<const quickik::KeypointObservation>(observations.data(), observations.size()));
    for (float angle : state->dof_angles()) { /* ... */ }
    ```

`Missing` keypoints (occluded this frame, or simply not tracked by this body plan's data source) don't just get dropped from the residual – with nothing pulling them away, the solve falls back on `SolverConfig`'s neutral-pose prior for whatever DOFs only those keypoints could otherwise constrain, so a body with everything missing settles at its neutral pose rather than an arbitrary one.

`SolverConfig` bundles the iteration count, damping, regularization weight, and convergence tolerance:

- `n_iterations`: how many Gauss-Newton steps to run per `solve` call, and the cap early stopping can cut short.
- `damping`: Levenberg-Marquardt damping added to the normal equations' diagonal, for numerical stability only – keep it very small (the default is `1e-6`).
- `neutral_pose_weight`: how strongly every joint angle is pulled toward its neutral pose. This is what keeps `Missing` keypoints (and, more generally, under-constrained DOFs) from drifting to an arbitrary angle instead of a sensible default, at the cost of some bias on frames where that DOF *is* observed.
- `position_tolerance`/`angle_tolerance`: stop iterating early once an update step's largest position and angle components both drop below these; `0` disables early stopping.

It's set via `Solver::new`/`Solver(...)`, though `solver.config` stays mutable for retuning between calls (Python: `solver.config` is the same object every time, so `solver.config.n_iterations = 5` takes effect on the next `solve`, just like Rust). C++'s `SolverConfig` is a plain value struct instead (no shared live handle): mutate a copy and call `solver->set_config(config)` to apply it.

## Solving IK on continuous sequences of frames

A single `solve()` call always starts from whatever `State` you pass it – usually the neutral pose, the first time. But real tracking data is a sequence of frames, and the previous frame's solved pose is almost always an excellent starting guess for the next one: real motion is continuous, so warm-starting from it both converges faster (Gauss-Newton starts much closer to the answer) and tracks more smoothly (it doesn't have to re-discover the same pose from scratch every frame, so it's less prone to landing in a different local optimum frame to frame). `SequenceSolver` automates exactly this: it keeps a `Solver` and `State` together and warm-starts each new frame from the last one it solved.

=== "Rust"

    ```rust
    use quickik::high_level::{SegmentedSolveConfig, SequenceSolver, solve_sequence_segmented_parallel};

    let mut seq_solver = SequenceSolver::new(kinematic_tree.clone(), SolverConfig::default());
    for frame_observations in &recording {
        let pose = seq_solver.solve_frame(frame_observations);
    }

    let segmented_config = SegmentedSolveConfig { segment_len: 200, overlap_len: 20, overlap_tolerance: 0.05 };
    let poses = solve_sequence_segmented_parallel(&kinematic_tree, SolverConfig::default(), &long_recording, segmented_config);
    ```

=== "Python"

    ```python
    seq_solver = quickik.SequenceSolver(kinematic_tree, quickik.SolverConfig())
    for frame_observations in recording:
        pose = seq_solver.solve_frame(frame_observations)

    segmented_config = quickik.SegmentedSolveConfig(
        segment_len=200, overlap_len=20, overlap_tolerance=0.05
    )
    poses = quickik.solve_sequence_segmented_parallel(
        kinematic_tree, quickik.SolverConfig(), long_recording, segmented_config
    )
    ```

=== "C++"

    ```cpp
    auto seq_solver = quickik::new_sequence_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());
    for (auto &frame_observations : recording) {
        auto pose = seq_solver->solve_frame(
            rust::Slice<const quickik::KeypointObservation>(frame_observations.data(), frame_observations.size()));
    }

    // flattened_long_recording is n_joints * n_frames long – see below.
    quickik::SegmentedSolveConfig segmented_config{200, 20, 0.05f};
    auto poses = quickik::solve_sequence_segmented_parallel(
        *tree, quickik::default_solver_config(),
        rust::Slice<const quickik::KeypointObservation>(flattened_long_recording.data(), flattened_long_recording.size()),
        tree->n_joints(), segmented_config, quickik::no_mapper());
    ```

    C++ has no nested-container binding across the FFI, so a "sequence" is one flat `observations` slice of length `n_joints * n_frames` (frame `i` spanning `[i * n_joints, (i + 1) * n_joints)`) rather than a list of lists, and `solve_sequence`/`solve_sequence_segmented_parallel` return a `StateList` handle (`len()`/`at(i)`) instead of a `Vec<State>`/`list[State]`. See `cpp/src/lib.rs`'s module docs.

A plain, sequential `SequenceSolver` only ever uses one thread, and one frame has to finish before the next can start (each one warm-starts from the last, after all). `solve_sequence_segmented_parallel` gets around that for a single long recording: it splits the sequence into segments with a small overlap, solves every segment on its own thread (cold-started at each segment's own first frame, then warm-started within it), and stitches the results back into one continuous sequence. `overlap_tolerance` is a consistency check, not a correctness requirement – neighboring segments' overlapping frames were solved independently (one cold-started, one warm-started from a different point), so they can disagree slightly even on a real recording; exceeding this per-DOF angle tolerance (radians) just logs a warning; the resulting sequence itself is unaffected. Independent sequences (e.g. one per subject or one per camera) don't need this machinery at all – just solve each with its own `SequenceSolver` and parallelize however you like (a thread pool, `rayon`, Python's `multiprocessing`, ...).

## Keypoint positions observed in 2D

Not every tracking source gives 3D positions directly. A single camera view only ever gives 2D pixel coordinates, and recovering the underlying 3D pose is itself part of what the solve needs to do. `Position2D` observations cover this: set `SolverConfig`'s mapper to a `Camera` (a pinhole projection model – focal lengths, principal point, and the camera's own pose relative to the body plan's world frame) and QuickIK projects each candidate 3D keypoint position through it before comparing to the observed pixel coordinates, rather than comparing 3D positions directly. `XYView` covers the simpler case of keypoints already reprojected onto a physical X-Y plane (e.g. a top-down tracking setup) – no camera intrinsics/extrinsics involved, just the identity projection dropping Z.

=== "Rust"

    ```rust
    use quickik::observation::Camera;
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

=== "Python"

    ```python
    camera = quickik.Camera(
        fx=800.0, fy=800.0, cx=320.0, cy=240.0,
        world2cam_pos=(0.0, 0.0, 5.0),
        world2cam_rot_mat=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),  # row-major 3x3
    )
    solver = quickik.Solver(kinematic_tree, quickik.SolverConfig(), mapper=camera)
    ```

=== "C++"

    ```cpp
    quickik::Camera camera{};
    camera.fx = 800.0f;
    camera.fy = 800.0f;
    camera.cx = 320.0f;
    camera.cy = 240.0f;
    camera.world2cam_pos = {0.0f, 0.0f, 5.0f};
    camera.world2cam_rot_mat = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};  // row-major 3x3
    auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::camera_mapper(camera));
    ```

Rust's `Solver<M>` is generic over the mapper type at compile time; neither Python nor C++ has an equivalent, so every `Solver`/`SequenceSolver` is backed by a single mapper value chosen at runtime instead – Python takes `mapper=None` (the default), a `Camera`, or an `XYView()` as a keyword argument; C++ takes a `quickik::Mapper` built via `no_mapper()`/`camera_mapper(camera)`/`xyview_mapper()` as an ordinary constructor argument in the same spots. It's deliberately not part of `SolverConfig` in either binding: like Rust's `M`, it's fixed for the solver's lifetime (read-only `mapper`/`solver.mapper` accessor, no setter), whereas `SolverConfig`'s other fields stay freely mutable.

Errors from malformed input (bad JSON, wrong-sized vectors, a `Position2D` observation with no mapper set) don't crash either binding – both raise a catchable error instead. In C++, that's a `rust::Error` (a normal `std::exception`). In Python, it's `pyo3_runtime.PanicException`, which – deliberately, on PyO3's part – subclasses `BaseException`, not `Exception`; a bare `except Exception:` won't catch it, so code that needs to handle these needs `except BaseException:` (or the specific `PanicException` type) instead.
