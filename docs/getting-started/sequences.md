# Solve sequences of frames

A single `solve()` call always starts from whatever `State` you pass it – usually the neutral pose, the first time. But real tracking data is a sequence of frames, and the previous frame's solved pose is almost always an excellent starting guess for the next one: real motion is continuous, so warm-starting from it both converges faster (Gauss-Newton starts much closer to the answer) and tracks more smoothly (it doesn't have to re-discover the same pose from scratch every frame, so it's less prone to landing in a different local optimum frame to frame). `SequenceSolver` automates exactly this: it keeps a `Solver` and `State` together and warm-starts each new frame from the last one it solved.

=== "Rust"

    ```rust
    use quickik::high_level::SequenceSolver;

    let mut seq_solver = SequenceSolver::new(kinematic_tree.clone(), SolverConfig::default());
    for frame_observations in &recording {
        let pose = seq_solver.solve_frame(frame_observations);
    }
    ```

=== "Python"

    ```python
    seq_solver = quickik.SequenceSolver(kinematic_tree, quickik.SolverConfig())
    for frame_observations in recording:
        pose = seq_solver.solve_frame(frame_observations)
    ```

=== "C++"

    ```cpp
    auto seq_solver = quickik::new_sequence_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());
    for (auto &frame_observations : recording) {
        auto pose = seq_solver->solve_frame(
            rust::Slice<const quickik::KeypointObservation>(frame_observations.data(), frame_observations.size()));
    }
    ```

A plain, sequential `SequenceSolver` like this only ever uses one thread, and one frame has to finish before the next can start (each one warm-starts from the last, after all). For a single long recording, `solve_sequence_segmented_parallel` gets around that: it splits the sequence into segments with a small overlap, solves every segment on its own worker thread (cold-started at each segment's own first frame, then warm-started within it), and stitches the results back into one continuous sequence. `overlap_tolerance` is a consistency check, not a correctness requirement – neighboring segments' overlapping frames were solved independently (one cold-started, one warm-started from a different point), so they can disagree slightly even on a real recording; exceeding this per-DOF angle tolerance (radians) just logs a warning; the resulting sequence itself is unaffected. A positive `n_workers` is used directly, unless it exceeds the number of available cores – in that case it's clipped to that count and a warning is logged. A negative value counts backward from all available cores: `-1` uses all, `-2` uses all but one, etc.; `0` is invalid.

=== "Rust"

    ```rust
    use quickik::high_level::{ParallelSolveConfig, solve_sequence_segmented_parallel};

    let parallel_config = ParallelSolveConfig { segment_len: 200, overlap_len: 20, overlap_tolerance: 0.05, n_workers: -1 };
    let poses = solve_sequence_segmented_parallel(&kinematic_tree, SolverConfig::default(), &long_recording, parallel_config);
    ```

=== "Python"

    ```python
    import numpy as np

    # positions: (n_frames, n_joints, 3) float32, in kinematic_tree.joints
    # order. weights: (n_frames, n_joints) float32; a keypoint with
    # weight <= 0 counts as Missing.
    positions = np.zeros((len(long_recording), kinematic_tree.n_joints, 3), dtype=np.float32)
    weights = np.ones((len(long_recording), kinematic_tree.n_joints), dtype=np.float32)
    # ... fill positions/weights from long_recording ...

    parallel_config = quickik.ParallelSolveConfig(
        segment_len=200, overlap_len=20, overlap_tolerance=0.05, n_workers=-1
    )
    poses = quickik.solve_sequence_segmented_parallel(
        kinematic_tree, quickik.SolverConfig(), positions, weights, parallel_config
    )
    ```

    Python takes the whole sequence as `positions`/`weights` numpy arrays instead of a list of per-frame `KeypointObservation` lists, unlike `SequenceSolver.solve_frame` above: constructing one Python object per keypoint per frame is fine for a single frame at a time, but its overhead dominates once you're pushing a whole recording through in one call – see the [benchmarks page](../benchmarks.md)'s "QuickIK (Python/C++/Rust)" implementation note.

=== "C++"

    ```cpp
    // flattened_long_recording is n_joints * n_frames long – see below.
    quickik::ParallelSolveConfig parallel_config{200, 20, 0.05f, -1};
    auto poses = quickik::solve_sequence_segmented_parallel(
        *tree, quickik::default_solver_config(),
        rust::Slice<const quickik::KeypointObservation>(flattened_long_recording.data(), flattened_long_recording.size()),
        tree->n_joints(), parallel_config, quickik::no_mapper());
    ```

    C++ has no nested-container binding across the FFI, so a "sequence" is one flat `observations` slice of length `n_joints * n_frames` (frame `i` spanning `[i * n_joints, (i + 1) * n_joints)`) rather than a list of lists, and `solve_sequence`/`solve_sequence_segmented_parallel` return a `StateList` handle (`len()`/`at(i)`) instead of a `Vec<State>`/`list[State]`. See `cpp/src/lib.rs`'s module docs.

Independent sequences (e.g. one per subject or one per camera) don't need this machinery at all – just solve each with its own `SequenceSolver` and parallelize however you like (a thread pool, `rayon`, Python's `multiprocessing`, ...).
