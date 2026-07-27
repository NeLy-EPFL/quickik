# QuickIK C++ bindings

QuickIK's C++ API mirrors the Rust crate: load a `KinematicTree` once, then
call `Solver::solve` per frame with a `State` and per-joint
`KeypointObservation`s -- or, for a continuous sequence of frames,
`SequenceSolver::solve_frame` (one warm-started frame at a time) or
`SequenceSolver::solve_sequence` (a whole sequence in one call).
