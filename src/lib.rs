//! QuickIK is a fast inverse kinematics library aimed at both high throughput
//! and low latency. It solves single frames, continuous sequences (warm-started,
//! optionally in parallel chunks), and batches of independent items (e.g. for
//! training with autodiff), from any mix of 3D and projected 2D keypoint
//! observations, with optional priors.

pub mod batched_solver;
pub mod body_plan;
pub mod error;
pub mod forward;
pub mod observation;
mod priors;
pub mod sequential_solver;
pub mod solver;
pub mod state;
mod utils;

pub use error::{Error, Result};
