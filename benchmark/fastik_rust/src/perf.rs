//! Consolidated throughput/latency benchmark for `fastik`'s Rust API: single-
//! frame latency, single-thread sequence throughput, and multi-thread
//! sequence throughput -- all at the default config (early stopping via
//! `position_tolerance`/`angle_tolerance` enabled, i.e. `n_iterations` acts as
//! a ceiling rather than a fixed cost). The weak-scaling sweep lives in
//! `../../fastik_scaling`, reusing this module's tiling/multi-thread-
//! throughput helpers (`pub` below for that reason).

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
/// and `fastik_scaling`'s weak-scaling sweep.
pub const SEGMENT_LEN: usize = 200;
pub const OVERLAP_LEN: usize = 20;
/// Thread count for the main "multi-thread sequence throughput" metric --
/// fixed rather than detected, so the number is reproducible regardless of
/// the machine/taskset state (sizing the sequence to exactly this many
/// segments also naturally caps `solve_sequence_segmented_parallel`'s own
/// thread spawning at this count, via its `available_parallelism().min(n_segments)`).
/// See `../../fastik_scaling` for the separate 1/2/4/8/16 sweep.
const MULTITHREAD_N_THREADS: usize = 8;

/// Total frame count that `solve_sequence_segmented_parallel`'s own
/// `segment_bounds` splits into exactly `n_segments` segments of
/// `SEGMENT_LEN` frames each (stride `SEGMENT_LEN - OVERLAP_LEN` between
/// segment starts) -- so a `n_segments`-thread run gets exactly one segment
/// per thread, rather than some threads getting two while others idle.
pub fn frames_for_n_segments(n_segments: usize) -> usize {
    SEGMENT_LEN + n_segments.saturating_sub(1) * (SEGMENT_LEN - OVERLAP_LEN)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

/// Prints the usual latency/throughput summary and returns the mean, for
/// callers that also want the number for `results/fastik-rust.json`.
fn summarize(label: &str, mut samples: Vec<Duration>) -> Duration {
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
    mean
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
pub fn tiled_native_rate_sequence(native_rate_frames: &[NativeRateFrame], len: usize) -> Vec<Vec<KeypointObservation>> {
    let base = observations_from_target_egos(native_rate_frames.iter().map(|f| f.target_ego.as_slice()));
    (0..len).map(|i| base[i % base.len()].clone()).collect()
}

/// Single-frame latency: a fresh `State::neutral_pose()` solved against a
/// fixed real target every call (no warm start).
fn bench_single_frame_latency(
    tree: &Arc<KinematicTree>,
    target_obs: &[KeypointObservation],
    n_calls: usize,
    config: SolverConfig,
) -> Vec<Duration> {
    let mut solver: Solver = Solver::new(tree, config);
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
pub fn bench_multithread_sequence_throughput(tree: &Arc<KinematicTree>, sequence: &[Vec<KeypointObservation>]) -> Duration {
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

pub fn run_all(tree: &Arc<KinematicTree>, fixtures: &Fixtures, body: &str) {
    println!("fastik Rust benchmark (state_dim={})\n", tree.state_dim());

    // Same fixture-derived target used by the Python and C++ benchmarks, so
    // this number is directly comparable across all three.
    let target_obs = build_observations(&fixtures.synthetic_frames[0].target_ego);
    println!("-- single-frame time (latency), default config (adaptive early stop) --");
    let single_frame_latency = summarize(
        "solve()",
        bench_single_frame_latency(tree, &target_obs, 20_000, SolverConfig::default()),
    );

    // Early stop disabled (tolerances = 0), so every call runs the full
    // `n_iterations` -- the worst case if a frame never converges early.
    let max_iterations_config: SolverConfig = SolverConfig {
        position_tolerance: 0.0,
        angle_tolerance: 0.0,
        ..SolverConfig::default()
    };
    println!(
        "\n-- single-frame time (latency), early stop disabled ({} iterations) --",
        max_iterations_config.n_iterations
    );
    let single_frame_latency_max = summarize(
        "solve() (forced max iterations)",
        bench_single_frame_latency(tree, &target_obs, 20_000, max_iterations_config),
    );

    println!("\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --");
    let single_thread_mean = summarize(
        "SequenceSolver.solve_frame",
        bench_single_thread_sequence_throughput(tree, &fixtures.native_rate_frames),
    );

    println!("\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop, {MULTITHREAD_N_THREADS} threads) --");
    let sequence = tiled_native_rate_sequence(&fixtures.native_rate_frames, frames_for_n_segments(MULTITHREAD_N_THREADS));
    let elapsed = bench_multithread_sequence_throughput(tree, &sequence);
    let multithread_fps = sequence.len() as f64 / elapsed.as_secs_f64();
    println!(
        "solve_sequence_segmented_parallel   n_frames={:<6} elapsed={elapsed:>9.3?}  throughput={:>10.1} frames/s",
        sequence.len(),
        multithread_fps,
    );

    write_results_json(
        body,
        single_frame_latency.as_secs_f64() * 1e6,
        single_frame_latency_max.as_secs_f64() * 1e6,
        1.0 / single_thread_mean.as_secs_f64(),
        multithread_fps,
    );
}

/// Writes `../plot/results/fastik-rust-<body>.json` for
/// `../plot/plot_comparison.py` to pick up.
fn write_results_json(
    body: &str,
    single_frame_latency_us: f64,
    single_frame_latency_max_us: f64,
    single_thread_throughput_fps: f64,
    multi_thread_throughput_fps: f64,
) {
    let results = serde_json::json!({
        "name": "fastik-rust",
        "body": body,
        "language": "rust",
        "formulation": "whole-tree",
        "single_frame_latency_us": single_frame_latency_us,
        "single_frame_latency_max_us": single_frame_latency_max_us,
        "single_thread_throughput_fps": single_thread_throughput_fps,
        "multi_thread_throughput_fps": multi_thread_throughput_fps,
        "multi_thread_n_threads": MULTITHREAD_N_THREADS,
        "notes": serde_json::Value::Null,
    });
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plot/results");
    std::fs::create_dir_all(&out_dir).expect("failed to create ../plot/results");
    let out_path = out_dir.join(format!("fastik-rust-{body}.json"));
    std::fs::write(&out_path, serde_json::to_string_pretty(&results).unwrap())
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
}
