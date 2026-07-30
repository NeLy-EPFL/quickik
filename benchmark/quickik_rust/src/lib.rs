//! Shared correctness/fixtures/perf helpers for the Rust benchmark binary
//! (`src/main.rs`) and `../quickik_scaling` (which reuses `perf`'s tiling and
//! multi-thread-throughput helpers for its own weak-scaling sweep).

pub mod correctness;
pub mod errors;
pub mod fixtures;
pub mod perf;
pub mod regression;
pub mod twod;
