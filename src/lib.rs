//! QuickIK is a fast inverse kinematics library aimed for both high throughput
//! and low latency. It provides high-level APIs for processing consecutive
//! frames (i.e. with warm starts) and multi-threaded batch processing, as well
//! as low-level APIs for more specific use cases (e.g. real-time application).

pub mod batched_solver;
pub mod body_plan;
pub mod forward;
pub mod observation;
pub mod sequential_solver;
pub mod solver;
pub mod state;
mod utils;
