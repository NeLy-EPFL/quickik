//! Consolidated throughput/latency benchmark for `fastik`'s Rust API: single-
//! frame latency, single-thread sequence throughput, and multi-thread
//! sequence throughput -- all at the default config (early stopping via
//! `position_tolerance`/`angle_tolerance` enabled, i.e. `n_iterations` acts as
//! a ceiling rather than a fixed cost). Also a weak-scaling sweep (Rust
//! only): see `run_weak_scaling_point`'s docs and README.md for how to sweep
//! thread counts via `taskset`.

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastik::body_plan::KinematicTree;
use fastik::high_level::{SegmentedSolveConfig, SequenceSolver, solve_sequence_segmented_parallel};
use fastik::observation::KeypointObservation;
use fastik::solver::{Solver, SolverConfig};
use fastik::state::State;

use crate::correctness::build_observations;
use crate::fixtures::{Fixtures, NativeRateFrame};

/// Frames per segment/thread for both the multi-thread throughput benchmark
/// and the weak-scaling sweep.
const SEGMENT_LEN: usize = 200;
const OVERLAP_LEN: usize = 20;

/// Total frame count that `solve_sequence_segmented_parallel`'s own
/// `segment_bounds` splits into exactly `n_segments` segments of
/// `SEGMENT_LEN` frames each (stride `SEGMENT_LEN - OVERLAP_LEN` between
/// segment starts) -- so a `n_segments`-thread run gets exactly one segment
/// per thread, rather than some threads getting two while others idle.
fn frames_for_n_segments(n_segments: usize) -> usize {
    SEGMENT_LEN + n_segments.saturating_sub(1) * (SEGMENT_LEN - OVERLAP_LEN)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn summarize(label: &str, mut samples: Vec<Duration>) {
    samples.sort();
    let n = samples.len();
    let mean = samples.iter().sum::<Duration>() / n as u32;
    println!(
        "{label:<32} n={n:<7} mean={:>9.3?}  median={:>9.3?}  p95={:>9.3?}  p99={:>9.3?}  \
         min={:>9.3?}  max={:>9.3?}  throughput={:>10.0} frames/s",
        mean,
        percentile(&samples, 0.50),
        percentile(&samples, 0.95),
        percentile(&samples, 0.99),
        samples[0],
        samples[n - 1],
        1.0 / mean.as_secs_f64(),
    );
}

/// Builds one `Missing` (root) + `Position3D` (per leg joint) observation
/// vector per frame, from anything shaped like a `target_ego: &[[f32; 3]]`.
fn observations_from_target_egos<'a>(
    target_egos: impl Iterator<Item = &'a [[f32; 3]]>,
) -> Vec<Vec<KeypointObservation>> {
    target_egos.map(build_observations).collect()
}

/// Tiles the native-rate fixture (real, contiguous frame-to-frame motion) up
/// to `len` frames, for benchmarks that need more frames than the 300-frame
/// fixture has (e.g. to occupy many threads).
fn tiled_native_rate_sequence(native_rate_frames: &[NativeRateFrame], len: usize) -> Vec<Vec<KeypointObservation>> {
    let base = observations_from_target_egos(native_rate_frames.iter().map(|f| f.target_ego.as_slice()));
    (0..len).map(|i| base[i % base.len()].clone()).collect()
}

/// Single-frame latency: a fresh `State::neutral_pose()` solved against a
/// fixed real target every call (no warm start), default config.
fn bench_single_frame_latency(tree: &Arc<KinematicTree>, target_obs: &[KeypointObservation], n_calls: usize) -> Vec<Duration> {
    let mut solver: Solver = Solver::new(tree, SolverConfig::default());
    for _ in 0..1000 {
        let mut state = State::neutral_pose(tree.clone());
        solver.solve(&mut state, black_box(target_obs));
        black_box(&state);
    }
    let mut samples = Vec::with_capacity(n_calls);
    for _ in 0..n_calls {
        let mut state = State::neutral_pose(tree.clone());
        let t0 = Instant::now();
        solver.solve(&mut state, black_box(target_obs));
        samples.push(t0.elapsed());
        black_box(&state);
    }
    samples
}

/// Single-thread sequence throughput: `SequenceSolver::solve_frame` warm
/// started across the native-rate fixture (a contiguous run of consecutive
/// recorded frames -- the frame-to-frame motion an actual continuous
/// tracking pipeline would see), default config. A second, fresh
/// `SequenceSolver` is used for the timed pass after warming up once, so the
/// sequence's own frame-to-frame warm-starting is what's measured.
fn bench_single_thread_sequence_throughput(tree: &Arc<KinematicTree>, native_rate_frames: &[NativeRateFrame]) -> Vec<Duration> {
    let all_obs = observations_from_target_egos(native_rate_frames.iter().map(|f| f.target_ego.as_slice()));
    let config = SolverConfig::default();

    let mut seq: SequenceSolver = SequenceSolver::new(tree.clone(), config);
    for obs in &all_obs {
        seq.solve_frame(black_box(obs));
    }

    let mut timed_seq: SequenceSolver = SequenceSolver::new(tree.clone(), config);
    let mut samples = Vec::with_capacity(all_obs.len());
    for obs in &all_obs {
        let t0 = Instant::now();
        timed_seq.solve_frame(black_box(obs));
        samples.push(t0.elapsed());
    }
    samples
}

/// Multi-thread sequence throughput: `solve_sequence_segmented_parallel` on
/// a longer tiled sequence, using however many threads
/// [`std::thread::available_parallelism`] reports (the full machine, unless
/// constrained via `taskset`). Warms up once, then times a second run.
fn bench_multithread_sequence_throughput(tree: &Arc<KinematicTree>, sequence: &[Vec<KeypointObservation>]) -> Duration {
    let config: SolverConfig = SolverConfig::default();
    let segmented_config = SegmentedSolveConfig {
        segment_len: SEGMENT_LEN,
        overlap_len: OVERLAP_LEN,
        overlap_tolerance: 0.05,
    };
    let _ = solve_sequence_segmented_parallel(tree, config, sequence, segmented_config);
    let t0 = Instant::now();
    let states = solve_sequence_segmented_parallel(tree, config, sequence, segmented_config);
    let elapsed = t0.elapsed();
    black_box(&states);
    elapsed
}

/// One weak-scaling data point: fixes frames-per-thread at [`SEGMENT_LEN`]
/// (one segment per thread, via [`frames_for_n_segments`]) and scales total
/// frames with however many threads `available_parallelism` currently
/// reports, so the amount of work per thread stays constant as thread count
/// grows. Ideal weak scaling keeps `elapsed` constant across thread counts.
/// Run this binary repeatedly under `taskset -c 0`, `-c 0-1`, `-c 0-3`,
/// `-c 0-7`, `-c 0-15` to sweep 1/2/4/8/16 threads -- see README.md.
fn run_weak_scaling_point(tree: &Arc<KinematicTree>, native_rate_frames: &[NativeRateFrame]) {
    let n_threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let total_frames = frames_for_n_segments(n_threads);
    let sequence = tiled_native_rate_sequence(native_rate_frames, total_frames);

    let elapsed = bench_multithread_sequence_throughput(tree, &sequence);
    println!(
        "n_threads={n_threads:<3} total_frames={total_frames:<6} elapsed={elapsed:>9.3?}  \
         throughput={:>10.1} frames/s",
        total_frames as f64 / elapsed.as_secs_f64(),
    );
}

pub fn run_all(tree: &Arc<KinematicTree>, fixtures: &Fixtures) {
    println!("fastik Rust benchmark (state_dim={})\n", tree.state_dim());

    // Same fixture-derived target used by the Python and C++ benchmarks, so
    // this number is directly comparable across all three.
    let target_obs = build_observations(&fixtures.synthetic_frames[0].target_ego);
    println!("-- single-frame time (latency), default config (adaptive early stop) --");
    summarize("solve()", bench_single_frame_latency(tree, &target_obs, 20_000));

    println!("\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --");
    summarize(
        "SequenceSolver.solve_frame",
        bench_single_thread_sequence_throughput(tree, &fixtures.native_rate_frames),
    );

    println!("\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop) --");
    let n_threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let sequence = tiled_native_rate_sequence(&fixtures.native_rate_frames, frames_for_n_segments(n_threads));
    let elapsed = bench_multithread_sequence_throughput(tree, &sequence);
    println!(
        "solve_sequence_segmented_parallel   n_frames={:<6} elapsed={elapsed:>9.3?}  throughput={:>10.1} frames/s",
        sequence.len(),
        sequence.len() as f64 / elapsed.as_secs_f64(),
    );

    println!(
        "\n-- weak scaling (this run's available_parallelism; sweep via taskset -c, see README) --"
    );
    run_weak_scaling_point(tree, &fixtures.native_rate_frames);
}
