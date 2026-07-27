# Solve sequences of frames

A single call to the solve method always starts from whatever state you pass it, usually the neutral pose. Real tracking data is typically a sequence of frames depicting continuous motion, so the previous frame's solved pose is almost always an excellent starting guess for the next one. To take advantage of this, QuickIK provides a `SequenceSolver` that automates the warm-starting process. As a result, the solver almost always converges faster.

## Solving a whole sequence in one call

Assuming you already have the whole recording upfront, the example below solves a whole sequence of frames in one call:

=== "Rust"

    ```rust
    use std::sync::Arc;
    use quickik::body_plan::KinematicTree;
    use quickik::solver::SolverConfig;
    use quickik::observation::KeypointObservation;
    use quickik::high_level::SequenceSolver;

    let kinematic_tree = Arc::new(KinematicTree::from_json_file("body_plan.json"));
    let mut seq_solver = SequenceSolver::new(kinematic_tree.clone(), SolverConfig::default());

    // recording: Vec<Vec<KeypointObservation>>. One inner Vec per frame, each n_joints long.
    let recording = ...;

    let poses = seq_solver.solve_sequence(&recording);
    ```

=== "Python"

    ```python
    from quickik import KinematicTree, SequenceSolver, SolverConfig

    kinematic_tree = KinematicTree.from_json_file("body_plan.json")
    seq_solver = SequenceSolver(kinematic_tree, SolverConfig())

    # positions: NDArray of shape (n_frames, n_joints, 3), in kinematic_tree.joints order
    positions = ...
    # weights: NDArray of shape (n_frames, n_joints); 0, below, or NaN indicates keypoint is missing
    weights = ...

    poses = seq_solver.solve_sequence(positions, weights)
    ```

    !!! note "Bigger practical performance win in Python"
        Python's `solve_sequence` takes `positions`/`weights` NumPy arrays instead of a list of per-frame `KeypointObservation` lists, so it never constructs one Python object per keypoint per frame. That construction is what actually dominates call overhead for a long recording.

    Any dtype is accepted for `positions`/`weights` (e.g. the common case of a `float64` array) and cast to `float32`, following NumPy's own casting rules.

    !!! note "2D keypoints"
        `positions`'s last dimension follows `mapper` (see ["From 2D keypoint positions"](2d-keypoints.md)): shape `(n_frames, n_joints, 3)` if `mapper` is `None` (the default), or `(n_frames, n_joints, 2)` if a `Camera`/`XYView` mapper was passed to `SequenceSolver`. A mismatch between the two raises `ValueError`.

=== "C++"

    ```cpp
    #include "quickik.h"

    auto tree = quickik::kinematic_tree_from_json_file("body_plan.json");
    auto seq_solver = quickik::new_sequence_solver(
        *tree, quickik::default_solver_config(), quickik::no_mapper()
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

## Solving long sequences in parallel

A plain `SequenceSolver` only ever uses one thread, and each frame has to finish before the next can start, since every frame warm-starts from the last. For a single long recording, `solve_sequence_segmented_parallel` gets around that: it splits the recording into segments with small overlaps, solves each on its own worker thread (cold-started at the segment's first frame, then warm-started within it), and stitches the results back into one continuous sequence. The overlap does double duty: it gives every segment after the first a running start, since its own copy of the shared frames gets to warm up before it reaches genuinely new ones, and it doubles as a consistency check, since two independent solves of the same frames should agree closely. When they don't, by more than `overlap_tolerance`, a warning is logged and the earlier segment's version is kept.

The parallel configuration bundles the segment length and overlap, a consistency-check tolerance, and the worker count:

- **`segment_len`/`overlap_len`:** length of each segment and of the overlap between consecutive segments, in frames.
- **`overlap_tolerance`:** the per-DOF angle disagreement (radians) allowed between overlapping frames before logging the warning described above.
- **`n_workers`:** a positive value is used directly, clipped down (with a warning) if it exceeds the available core count. A negative value counts backward from all available cores: `-1` uses all, `-2` uses all but one, etc. `0` is invalid.

The example below splits a long recording into segments explicitly:

=== "Rust"

    ```rust
    use quickik::high_level::{ParallelSolveConfig, solve_sequence_segmented_parallel};

    let parallel_config = ParallelSolveConfig {
        segment_len: 200,
        overlap_len: 10,
        overlap_tolerance: 0.05,
        n_workers: -1, // -1 = all available threads, -2 = all but one, etc.
    };

    // long_recording: Vec<Vec<KeypointObservation>>. One inner Vec per frame, each n_joints long.
    let long_recording = ...;

    let poses = solve_sequence_segmented_parallel(
        &kinematic_tree,
        SolverConfig::default(),
        &long_recording,
        parallel_config,
    );
    ```

=== "Python"

    ```python
    import numpy as np
    from quickik import ParallelSolveConfig, solve_sequence_segmented_parallel

    parallel_config = ParallelSolveConfig(
        segment_len=200,
        overlap_len=10,
        overlap_tolerance=0.05,
        n_workers=-1,  # -1 = all available threads, -2 = all but one, etc.
    )

    # long_positions: NDArray of shape (n_frames, n_joints, 3), in kinematic_tree.joints order
    long_positions = ...
    # long_weights: NDArray of shape (n_frames, n_joints); 0, below, or NaN indicates keypoint is missing
    long_weights = ...

    poses = solve_sequence_segmented_parallel(
        kinematic_tree, SolverConfig(), long_positions, long_weights, parallel_config
    )
    ```

    Like `solve_sequence` above, `long_positions`/`long_weights` accept any dtype and are cast to `float32`; `long_positions`'s last dimension is 2 instead of 3 if `mapper` is a `Camera`/`XYView` (see the note above).

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
        quickik::no_mapper()
    );

    // poses is a StateList, not a std::vector<State>. Read it out with .len()/.at(i).
    for (size_t i = 0; i < poses->len(); i++) {
        auto pose = poses->at(i);
    }
    ```

If you'd rather not tune `segment_len`/`overlap_len` yourself, a `for_recording` constructor (C++: the free function `parallel_solve_config_for_recording`) builds a `ParallelSolveConfig` that spreads `total_len` frames evenly across every available core – one segment per core, sized by simple division plus a fixed default overlap. Build a `ParallelSolveConfig` directly, as above, for finer control over cold-start frequency.

In C++, this takes the same flattened-slice-plus-`n_joints` layout as `solve_sequence` above (there's no way to pass a list of per-frame observation lists directly across the FFI). The results come back as a `StateList` rather than a plain vector, as shown above.

Independent sequences (e.g. one per subject or one per camera) don't need this machinery. Just solve each with its own `SequenceSolver` and parallelize however you like (a thread pool, Rust's [Rayon](https://docs.rs/rayon/latest/rayon/), Python's [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) or [Joblib](https://joblib.readthedocs.io/), etc.).
