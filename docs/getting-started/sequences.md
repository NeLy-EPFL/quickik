# Solve sequences of frames

A single call to the solve method always starts from whatever state you pass it, usually the neutral pose. Real tracking data is typically a sequence of frames depicting continuous motion, so the previous frame's solved pose is almost always an excellent starting guess for the next one. To take advantage of this, QuickIK provides a `SequenceSolver` that automates the warm-starting process. As a result, the solver almost always converges faster.

## Solving a whole sequence in one call

Assuming you already have the whole recording upfront, the example below solves a whole sequence of frames in one call:

=== "Rust"

    ```rust
    use std::sync::Arc;
    use quickik::body_plan::KinematicTree;
    use quickik::observation::{KeypointObservation, Projection};
    use quickik::sequential_solver::SequenceSolver;

    let kinematic_tree = Arc::new(KinematicTree::from_json_file("body_plan.json"));
    let mut seq_solver = SequenceSolver::new(
        &kinematic_tree, Projection::new_3d(), 10, 1e-3, 1e-3, 1e-3, 1e-6,
    );

    // recording: Vec<Vec<KeypointObservation>>. One inner Vec per frame, each n_joints long.
    let recording = ...;

    let results = seq_solver.solve(&recording, false, false);
    ```

=== "Python"

    ```python
    from quickik import KinematicTree, SequenceSolver

    kinematic_tree = KinematicTree.from_json_file("body_plan.json")
    seq_solver = SequenceSolver(kinematic_tree)

    # positions: NDArray of shape (n_frames, n_joints, 3), in kinematic_tree.joints order
    positions = ...
    # weights: NDArray of shape (n_frames, n_joints); 0, below, or NaN indicates keypoint is missing
    weights = ...

    results = seq_solver.solve(positions, weights)
    ```

    !!! note "Bigger practical performance win in Python"
        Python's `solve` takes `positions`/`weights` NumPy arrays instead of a list of per-frame `KeypointObservation` lists, so it never constructs one Python object per keypoint per frame. That construction is what actually dominates call overhead for a long recording.

    Any dtype is accepted for `positions`/`weights` (e.g. the common case of a `float64` array) and cast to `float32`, following NumPy's own casting rules.

    !!! note "2D keypoints"
        `positions`'s last dimension follows `SequenceSolver`'s `projection` (see ["From 2D keypoint positions"](2d-keypoints.md)): shape `(n_frames, n_joints, 3)` for the default 3D projection, or `(n_frames, n_joints, 2)` for a `Camera`/orthographic X-Y projection. A mismatch between the two raises `ValueError`.

=== "C++"

    ```cpp
    #include "quickik.h"

    auto tree = quickik::kinematic_tree_from_json_file("body_plan.json");
    auto seq_solver = quickik::new_sequence_solver(
        *tree, quickik::default_solver_config(), quickik::projection_3d()
    );

    // flattened_recording is every frame's observations concatenated back to
    // back: n_joints * n_frames long, frame i at [i * n_joints, (i + 1) * n_joints).
    auto flattened_recording = ...;  // std::vector<quickik::KeypointObservation>
    auto observations = rust::Slice<const quickik::KeypointObservation>(
        flattened_recording.data(), flattened_recording.size()
    );

    auto poses = seq_solver->solve_sequence(observations, tree->n_joints());
    ```

    !!! note "Explicit `n_joints` in C++"
        `solve_sequence` needs `n_joints` explicitly in C++, since `flattened_recording` is one long slice rather than a list of per-frame slices – C++ has no nested-container binding across the FFI – so `n_joints` is the stride used to cut that one slice back into individual frames.

### Checking fit quality

Like a plain `Solver` (see ["Checking fit quality"](single-frame.md#checking-fit-quality)), pass `with_fk` to get each frame's converged world-space keypoint positions alongside its pose, as that frame's own `SolverResult.keypoint_pos`.

=== "Rust"

    ```rust
    let results = seq_solver.solve(&recording, false, true);
    for result in &results {
        println!("{:?}", result.keypoint_pos.as_ref().unwrap());
    }
    ```

=== "Python"

    ```python
    results = seq_solver.solve(positions, weights, with_fk=True)
    print(results[0].keypoint_pos.shape)  # (n_joints, 3)
    ```

=== "C++"

    C++ doesn't have a `solve_sequence_with_fk` (its `StateList` return type would need a matching flattened-positions companion type, which isn't worth the added surface for this getting-started guide) -- read `last_fk_positions()` after `solve_frame` instead, one frame at a time.

## Solving long sequences in parallel

A plain `SequenceSolver` only ever uses one thread, and each frame has to finish before the next can start, since every frame warm-starts from the last. For a single long recording, `solve_segments_parallel` gets around that: it splits the recording into exactly `n_workers` contiguous, non-overlapping segments (one per worker, as evenly sized as possible), and solves each on its own thread, cold-started at the neutral pose and then warm-started within itself, same as `solve` above. Segments aren't cross-checked against each other: better load distribution is worth more than that consistency check, since segments are independent either way. This never reads or writes the `SequenceSolver`'s own continuous `solve` state from the section above, so it's safe to call on an object you're also feeding frames into one at a time.

- **`n_workers`:** a positive value is used directly, clipped down (with a warning) if it exceeds the available core count. A negative value counts backward from all available cores: `-1` uses all, `-2` uses all but one, etc. `0` is invalid.

The example below solves a long recording, split across 4 worker threads:

=== "Rust"

    ```rust
    // long_recording: Vec<Vec<KeypointObservation>>. One inner Vec per frame, each n_joints long.
    let long_recording = ...;

    let results = seq_solver.solve_segments_parallel(&long_recording, 4, false, false);
    ```

=== "Python"

    ```python
    import numpy as np

    # long_positions: NDArray of shape (n_frames, n_joints, 3), in kinematic_tree.joints order
    long_positions = ...
    # long_weights: NDArray of shape (n_frames, n_joints); 0, below, or NaN indicates keypoint is missing
    long_weights = ...

    results = seq_solver.solve_segments_parallel(long_positions, long_weights, n_workers=4)
    ```

    Like `solve` above, `long_positions`/`long_weights` accept any dtype and are cast to `float32`; `long_positions`'s last dimension is 2 instead of 3 for a `Camera`/orthographic X-Y projection (see the note above).

=== "C++"

    ```cpp
    // flattened_long_recording is n_joints * n_frames long, same flattened layout
    // as solve_sequence
    auto flattened_long_recording = ...;

    quickik::ParallelSolveConfig parallel_config{200, 10, 0.05f, -1};

    // Wrap all observations in a Rust Slice view
    auto observations = rust::Slice<const quickik::KeypointObservation>(
        flattened_long_recording.data(), flattened_long_recording.size()
    );
    auto poses = quickik::solve_sequence_segmented_parallel(
        *tree,
        quickik::default_solver_config(),
        observations,
        tree->n_joints(),
        parallel_config,
        quickik::projection_3d()
    );

    // poses is a StateList, not a std::vector<State>. Read it out with .len()/.at(i).
    for (size_t i = 0; i < poses->len(); i++) {
        auto pose = poses->at(i);
    }
    ```

In C++, this takes the same flattened-slice-plus-`n_joints` layout as `solve_sequence` above (there's no way to pass a list of per-frame observation lists directly across the FFI). The results come back as a `StateList` rather than a plain vector, as shown above.

Independent sequences (e.g. one per subject or one per camera) don't need this machinery. Just solve each with its own `SequenceSolver` and parallelize however you like (a thread pool, Rust's [Rayon](https://docs.rs/rayon/latest/rayon/), Python's [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) or [Joblib](https://joblib.readthedocs.io/), etc.).

## Solving independent batches (no warm start)

Some workloads solve many *independent* frames rather than a continuous stream, e.g. shuffled minibatches when training a pose model with an autodiff framework. `BatchedSolver` covers this: every `solve` call starts each item fresh at the neutral pose (no warm-starting) and solves the whole batch in parallel on its own thread pool.

Unlike `Solver`/`SequenceSolver`, its keypoint axis is given by an explicit `keypoints_order` (a list of joint names) rather than the kinematic tree's own internal order, since callers (e.g. a pretrained detector) rarely already produce keypoints in that order; and it requires a free-floating-base tree, since its result always reports the root's `base_pos`/`base_quat`.

=== "Rust"

    ```rust
    use quickik::batched_solver::BatchedSolver;

    let keypoints_order = vec!["root".into(), "joint1".into(), "joint2".into(), "tip".into()];
    let batched_solver = BatchedSolver::new(
        &kinematic_tree, Projection::new_3d(), 10, 1e-3, 1e-3, 1e-3, 1e-6, keypoints_order, -1,
    );

    // observations_array: Vec<Vec<KeypointObservation>>, one inner Vec per batch item,
    // each n_joints long, in keypoints_order.
    let observations_array = ...;
    let result = batched_solver.solve(&observations_array, true, false); // with_grad=true
    ```

=== "Python"

    ```python
    from quickik import KinematicTree, BatchedSolver

    keypoints_order = ["root", "joint1", "joint2", "tip"]
    batched_solver = BatchedSolver(kinematic_tree, keypoints_order)

    # positions: NDArray (batch_size, n_joints, 3); weights: NDArray (batch_size, n_joints),
    # both in keypoints_order
    result = batched_solver.solve(positions, weights, with_grad=True)
    print(result.joint_angles.shape)  # (batch_size, n_dofs)
    ```

    `with_grad=True` additionally returns each item's Jacobian and Cholesky factor, which `quickik.torch.SolveIK` uses to differentiate through the solve (via implicit differentiation) when training a model with PyTorch.

See the [Python API reference](../api/python.md) for `BatchedSolver`'s full constructor options (tolerances, `n_workers`, etc.), same as `Solver`/`SequenceSolver`.
