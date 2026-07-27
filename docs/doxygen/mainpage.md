# QuickIK C++ bindings

QuickIK's C++ API mirrors the Rust crate: load a `KinematicTree` once, then
call `Solver::solve` (or `SequenceSolver::solve` for whole sequences) per
frame with a `State` and per-joint `KeypointObservation`s.
