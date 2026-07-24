//! Consolidated throughput/latency benchmark for `quickik`'s Rust API: single-
//! frame latency, single-thread sequence throughput, and multi-thread
//! sequence throughput -- all at the default config (early stopping via
//! `position_tolerance`/`angle_tolerance` enabled, i.e. `n_iterations` acts as
//! a ceiling rather than a fixed cost). The weak-scaling sweep lives in
//! `../../quickik_scaling`, reusing this module's tiling/multi-thread-
//! throughput helpers (`pub` below for that reason).

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use quickik::body_plan::KinematicTree;
use quickik::high_level::{ParallelSolveConfig, SequenceSolver, solve_sequence_segmented_parallel};
use quickik::observation::{Camera, KeypointObservation, Mapper3Dto2D, NoMapper, XYView};
use quickik::solver::{Solver, SolverConfig};
use quickik::state::State;

use crate::correctness::build_observations;
use crate::fixtures::{Fixtures, NativeRateFrame};
use crate::twod::{observations_2d_camera, observations_2d_xyview};

/// Frames per segment/thread for both the multi-thread throughput benchmark
/// and `quickik_scaling`'s weak-scaling sweep.
pub const SEGMENT_LEN: usize = 200;
pub const OVERLAP_LEN: usize = 20;
/// Worker count for the main "multi-thread sequence throughput" metric,
/// passed explicitly via `ParallelSolveConfig::n_workers` -- fixed rather
/// than detected, so the number is reproducible regardless of the machine's
/// core count. See `../../quickik_scaling` for the separate 1/2/4/8/16 sweep.
const MULTITHREAD_N_THREADS: usize = 8;
/// Frame count for the single-thread sequence-throughput metric, tiled from
/// the 300-frame native-rate fixture -- larger than the multi-thread
/// metric's per-worker segment since this one has no worker count to divide
/// by.
const SINGLE_THREAD_N_FRAMES: usize = 1000;

/// Total frame count that `solve_sequence_segmented_parallel`'s own
/// `segment_bounds` splits into exactly `n_segments` segments of
/// `SEGMENT_LEN` frames each (stride `SEGMENT_LEN - OVERLAP_LEN` between
/// segment starts) -- so a `n_segments`-worker run gets exactly one segment
/// per worker, rather than some workers getting two while others idle.
pub fn frames_for_n_segments(n_segments: usize) -> usize {
    SEGMENT_LEN + n_segments.saturating_sub(1) * (SEGMENT_LEN - OVERLAP_LEN)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

/// Prints the usual latency/throughput summary and returns the mean, for
/// callers that also want the number for `results/quickik-rust.json`.
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
pub fn tiled_native_rate_sequence(
    native_rate_frames: &[NativeRateFrame],
    len: usize,
) -> Vec<Vec<KeypointObservation>> {
    let base =
        observations_from_target_egos(native_rate_frames.iter().map(|f| f.target_ego.as_slice()));
    (0..len).map(|i| base[i % base.len()].clone()).collect()
}

/// Single-frame latency: a fresh `State::neutral_pose()` solved against a
/// fixed real target every call (no warm start).
fn bench_single_frame_latency<M: Mapper3Dto2D>(
    tree: &Arc<KinematicTree>,
    target_obs: &[KeypointObservation],
    n_calls: usize,
    config: SolverConfig<M>,
) -> Vec<Duration> {
    let mut solver: Solver<M> = Solver::new(tree, config);
    for _ in 0..500 {
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
/// started across a tiled native-rate sequence (the frame-to-frame motion an
/// actual continuous tracking pipeline would see), default config. A second,
/// fresh `SequenceSolver` is used for the timed pass after warming up once,
/// so the sequence's own frame-to-frame warm-starting is what's measured.
fn bench_single_thread_sequence_throughput<M: Mapper3Dto2D>(
    tree: &Arc<KinematicTree>,
    sequence: &[Vec<KeypointObservation>],
    config: SolverConfig<M>,
) -> Vec<Duration> {
    let mut seq: SequenceSolver<M> = SequenceSolver::new(tree.clone(), config);
    for obs in sequence {
        seq.solve_frame(black_box(obs));
    }

    let mut timed_seq: SequenceSolver<M> = SequenceSolver::new(tree.clone(), config);
    let mut samples = Vec::with_capacity(sequence.len());
    for obs in sequence {
        let t0 = Instant::now();
        timed_seq.solve_frame(black_box(obs));
        samples.push(t0.elapsed());
    }
    samples
}

/// Multi-thread sequence throughput: `solve_sequence_segmented_parallel` on
/// a longer tiled sequence, using exactly `n_workers` threads (joblib
/// convention -- see `ParallelSolveConfig::n_workers`). Warms up once, then
/// times a second run.
pub fn bench_multithread_sequence_throughput(
    tree: &Arc<KinematicTree>,
    sequence: &[Vec<KeypointObservation>],
    n_workers: isize,
) -> Duration {
    bench_multithread_sequence_throughput_with_config(
        tree,
        sequence,
        n_workers,
        SolverConfig::<NoMapper>::default(),
    )
}

fn bench_multithread_sequence_throughput_with_config<M: Mapper3Dto2D + Sync>(
    tree: &Arc<KinematicTree>,
    sequence: &[Vec<KeypointObservation>],
    n_workers: isize,
    config: SolverConfig<M>,
) -> Duration {
    let parallel_config = ParallelSolveConfig {
        segment_len: SEGMENT_LEN,
        overlap_len: OVERLAP_LEN,
        overlap_tolerance: 0.05,
        n_workers,
    };
    let _ = solve_sequence_segmented_parallel(tree, config, sequence, parallel_config);
    let t0 = Instant::now();
    let states = solve_sequence_segmented_parallel(tree, config, sequence, parallel_config);
    let elapsed = t0.elapsed();
    black_box(&states);
    elapsed
}

pub fn run_all(tree: &Arc<KinematicTree>, fixtures: &Fixtures, body: &str) {
    println!("quickik Rust benchmark (state_dim={})\n", tree.state_dim());

    // Same fixture-derived target used by the Python and C++ benchmarks, so
    // this number is directly comparable across all three.
    let target_obs = build_observations(&fixtures.synthetic_frames[0].target_ego);
    println!("-- single-frame time (latency), default config (adaptive early stop) --");
    let single_frame_latency = summarize(
        "solve()",
        bench_single_frame_latency(tree, &target_obs, 10_000, SolverConfig::<NoMapper>::default()),
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
        bench_single_frame_latency(tree, &target_obs, 10_000, max_iterations_config),
    );

    println!("\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --");
    let single_thread_sequence =
        tiled_native_rate_sequence(&fixtures.native_rate_frames, SINGLE_THREAD_N_FRAMES);
    let single_thread_mean = summarize(
        "SequenceSolver.solve_frame",
        bench_single_thread_sequence_throughput(
            tree,
            &single_thread_sequence,
            SolverConfig::<NoMapper>::default(),
        ),
    );

    println!(
        "\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop, {MULTITHREAD_N_THREADS} threads) --"
    );
    let sequence = tiled_native_rate_sequence(
        &fixtures.native_rate_frames,
        frames_for_n_segments(MULTITHREAD_N_THREADS),
    );
    let elapsed =
        bench_multithread_sequence_throughput(tree, &sequence, MULTITHREAD_N_THREADS as isize);
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

/// Writes `../plot/results/quickik-rust-<body>.json` for
/// `../plot/plot_comparison.py` to pick up.
fn write_results_json(
    body: &str,
    single_frame_latency_us: f64,
    single_frame_latency_max_us: f64,
    single_thread_throughput_fps: f64,
    multi_thread_throughput_fps: f64,
) {
    let results = serde_json::json!({
        "name": "quickik-rust",
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
    let out_path = out_dir.join(format!("quickik-rust-{body}.json"));
    std::fs::write(&out_path, serde_json::to_string_pretty(&results).unwrap())
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
}

/// Tiles the native-rate fixture up to `len` frames, reprojecting each
/// frame's `target_ego` to 2D via `to_2d` -- the 2D counterpart of
/// [`tiled_native_rate_sequence`].
fn tiled_native_rate_sequence_2d(
    native_rate_frames: &[NativeRateFrame],
    len: usize,
    to_2d: &impl Fn(&[[f32; 3]]) -> Vec<KeypointObservation>,
) -> Vec<Vec<KeypointObservation>> {
    let base: Vec<Vec<KeypointObservation>> = native_rate_frames
        .iter()
        .map(|f| to_2d(&f.target_ego))
        .collect();
    (0..len).map(|i| base[i % base.len()].clone()).collect()
}

/// Runs the same latency/throughput suite as [`run_all`], but every
/// observation is `to_2d`'s projection of the fixture's usual 3D target --
/// for both a synthetic pinhole [`Camera`] and the trivial [`XYView`], on
/// both bodies. Prints results only (no `results/*.json` output yet -- this
/// isn't wired into `plot_comparison.py`).
pub fn run_all_2d(tree: &Arc<KinematicTree>, fixtures: &Fixtures, camera: Camera) {
    run_all_2d_for_mapper(tree, fixtures, camera, "Camera", |t| {
        observations_2d_camera(t, &camera)
    });
    run_all_2d_for_mapper(tree, fixtures, XYView, "XYView", |t| observations_2d_xyview(t));
}

fn run_all_2d_for_mapper<M: Mapper3Dto2D + Sync>(
    tree: &Arc<KinematicTree>,
    fixtures: &Fixtures,
    mapper: M,
    label: &str,
    to_2d: impl Fn(&[[f32; 3]]) -> Vec<KeypointObservation>,
) {
    println!(
        "quickik Rust benchmark, 2D via {label} (state_dim={})\n",
        tree.state_dim()
    );

    let target_obs = to_2d(&fixtures.synthetic_frames[0].target_ego);
    println!("-- single-frame time (latency), default config (adaptive early stop) --");
    summarize(
        "solve()",
        bench_single_frame_latency(
            tree,
            &target_obs,
            10_000,
            SolverConfig {
                mapper: Some(mapper),
                ..SolverConfig::default()
            },
        ),
    );

    // Early stop disabled (tolerances = 0), so every call runs the full
    // `n_iterations` -- the worst case if a frame never converges early.
    let max_iterations_config = SolverConfig {
        position_tolerance: 0.0,
        angle_tolerance: 0.0,
        mapper: Some(mapper),
        ..SolverConfig::default()
    };
    println!(
        "\n-- single-frame time (latency), early stop disabled ({} iterations) --",
        max_iterations_config.n_iterations
    );
    summarize(
        "solve() (forced max iterations)",
        bench_single_frame_latency(tree, &target_obs, 10_000, max_iterations_config),
    );

    println!(
        "\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --"
    );
    let single_thread_sequence =
        tiled_native_rate_sequence_2d(&fixtures.native_rate_frames, SINGLE_THREAD_N_FRAMES, &to_2d);
    summarize(
        "SequenceSolver.solve_frame",
        bench_single_thread_sequence_throughput(
            tree,
            &single_thread_sequence,
            SolverConfig {
                mapper: Some(mapper),
                ..SolverConfig::default()
            },
        ),
    );

    println!(
        "\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop, {MULTITHREAD_N_THREADS} threads) --"
    );
    let sequence = tiled_native_rate_sequence_2d(
        &fixtures.native_rate_frames,
        frames_for_n_segments(MULTITHREAD_N_THREADS),
        &to_2d,
    );
    let elapsed = bench_multithread_sequence_throughput_with_config(
        tree,
        &sequence,
        MULTITHREAD_N_THREADS as isize,
        SolverConfig {
            mapper: Some(mapper),
            ..SolverConfig::default()
        },
    );
    let multithread_fps = sequence.len() as f64 / elapsed.as_secs_f64();
    println!(
        "solve_sequence_segmented_parallel   n_frames={:<6} elapsed={elapsed:>9.3?}  throughput={:>10.1} frames/s\n",
        sequence.len(),
        multithread_fps,
    );
}
