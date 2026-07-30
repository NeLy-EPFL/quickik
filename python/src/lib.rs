//! PyO3 bindings for the `quickik` crate. Mirrors the Rust API where
//! reasonable; the main departure is the mapper: Rust's `Solver<M>` /
//! `SequenceSolver<M>` / `BatchedSolver<M>` are generic over the mapper type
//! at compile time, but Python has no equivalent, so every Python-facing
//! solver is backed by a single `Mapper` enum (`Camera`, `XYView`, or none)
//! chosen at runtime instead. `mapper` is a constructor-only, read-only
//! property on all three, so it can never be swapped mid-lifetime.
//!
//! Submodules mirror the core crate's own layout (`body_plan`, `observation`,
//! `state`, `solver`, `sequential_solver`, `batched_solver`).

mod batched_solver;
mod body_plan;
mod observation;
mod sequential_solver;
mod solver;
mod state;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Runs `f`, converting a panic (e.g. from malformed body-plan JSON, an
/// invalid `ParallelSolveConfig`, or a mismatched observation count) into a
/// `PyValueError` instead of an uncaught `pyo3_runtime.PanicException`.
/// Mirrors the C++ bindings' own `catch_panic` (`cpp/src/lib.rs`). Every
/// mutation `f` might have made before panicking is just plain data with no
/// unsafe invariants to uphold, so asserting unwind-safety here is fine.
pub(crate) fn catch_panic<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic");
        PyValueError::new_err(msg.to_string())
    })
}

use batched_solver::{BatchedSolver, BatchedSolverResult};
use body_plan::KinematicTree;
use observation::{Camera, KeypointObservation, XYView};
use sequential_solver::SequenceSolver;
use solver::{Solver, SolverResult};
use state::State;

#[pymodule]
fn quickik(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KinematicTree>()?;
    m.add_class::<State>()?;
    m.add_class::<KeypointObservation>()?;
    m.add_class::<Camera>()?;
    m.add_class::<XYView>()?;
    m.add_class::<SolverResult>()?;
    m.add_class::<Solver>()?;
    m.add_class::<SequenceSolver>()?;
    m.add_class::<BatchedSolverResult>()?;
    m.add_class::<BatchedSolver>()?;
    Ok(())
}
