//! PyO3 bindings for the `quickik` crate. Mirrors the Rust API where
//! reasonable; the main departure is the mapper: Rust's `Solver<M>` is
//! generic over the mapper type at compile time, but Python has no
//! equivalent, so every Python-facing solver is backed by a single
//! `Mapper` enum (`Camera`, `XYView`, or none) chosen at runtime instead.
//! To preserve Rust's "fixed once upon construction" invariant, `mapper` is
//! a constructor-only, read-only argument on `Solver`/`SequenceSolver`/
//! `solve_sequence_segmented_parallel` -- not a field of the otherwise
//! freely mutable `SolverConfig` -- so it can never be swapped mid-lifetime.
//!
//! Submodules mirror the core crate's own layout (`body_plan`,
//! `observation`, `state`, `solver`, `high_level`).

mod body_plan;
mod high_level;
mod observation;
mod solver;
mod state;

use pyo3::prelude::*;

use body_plan::KinematicTree;
use high_level::{
    ParallelSolveConfig, SequenceSolver, solve_sequence_segmented_parallel,
    solve_sequence_segmented_parallel_from_observations,
};
use observation::{Camera, KeypointObservation, XYView};
use solver::{Solver, SolverConfig};
use state::State;

#[pymodule]
fn quickik(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KinematicTree>()?;
    m.add_class::<State>()?;
    m.add_class::<KeypointObservation>()?;
    m.add_class::<Camera>()?;
    m.add_class::<XYView>()?;
    m.add_class::<SolverConfig>()?;
    m.add_class::<Solver>()?;
    m.add_class::<SequenceSolver>()?;
    m.add_class::<ParallelSolveConfig>()?;
    m.add_function(wrap_pyfunction!(solve_sequence_segmented_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(
        solve_sequence_segmented_parallel_from_observations,
        m
    )?)?;
    Ok(())
}
