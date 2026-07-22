//! Shared correctness/fixtures/perf helpers for the Rust benchmark binary
//! (`src/main.rs`) and `../fastik_scaling` (which reuses `perf`'s tiling and
//! multi-thread-throughput helpers for its own weak-scaling sweep).

pub mod correctness;
pub mod fixtures;
pub mod perf;
