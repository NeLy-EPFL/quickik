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
use quickik::observation::{KeypointObservation, Mapper3Dto2D, NoMapper, XYView};
use quickik::sequential_solver::SequenceSolver;
use quickik::solver::Solver;
use quickik::state::State;

use crate::correctness::build_observations;
use crate::fixtures::{Fixtures, NativeRateFrame};
use crate::twod::observations_2d_xyview;

/// Frames per segment/thread for both the multi-thread throughput benchmark
/// and `quickik_scaling`'s weak-scaling sweep.
pub const SEGMENT_LEN: usize = 200;
/// Worker count for the main "multi-thread sequence throughput" metric,
/// passed explicitly via `solve_segments_parallel`'s `n_workers`: fixed
/// rather than detected, so the number is reproducible regardless of the
/// machine's core count. See `../../quickik_scaling` for the separate
/// 1/2/4/8/16 sweep.
const MULTITHREAD_N_THREADS: usize = 8;
/// Frame count for the single-thread sequence-throughput metric, tiled from
/// the 300-frame native-rate fixture: larger than the multi-thread
/// metric's per-worker segment since this one has no worker count to divide
/// by.
const SINGLE_THREAD_N_FRAMES: usize = 1000;

/// Bundles a `Solver`/`SequenceSolver` config so benchmark call sites don't
/// have to spell out all five numeric constructor args every time. Not part
/// of `quickik`'s own public API -- purely a benchmark-internal convenience.
#[derive(Clone, Copy)]
pub struct BenchConfig<M: Mapper3Dto2D> {
    pub mapper: M,
    pub n_iterations: usize,
    pub neutral_weight: f32,
    pub position_tolerance: f32,
    pub angle_tolerance: f32,
    pub damping: f32,
}

impl BenchConfig<NoMapper> {
    /// Default config (adaptive early stop enabled), 3D observations.
    pub fn default_3d() -> Self {
        Self::default_with_mapper(NoMapper)
    }
}

impl<M: Mapper3Dto2D> BenchConfig<M> {
    /// Default config (adaptive early stop enabled) with the given mapper.
    pub fn default_with_mapper(mapper: M) -> Self {
        Self {
            mapper,
            n_iterations: 10,
            neutral_weight: 1e-3,
            position_tolerance: 1e-3,
            angle_tolerance: 1e-3,
            damping: 1e-6,
        }
    }

    /// Same config, but with early stop disabled (`position_tolerance`/
    /// `angle_tolerance` set to 0), so every call runs the full
    /// `n_iterations` -- the worst case if a frame never converges early.
    pub fn forced_max_iterations(self) -> Self {
        Self {
            position_tolerance: 0.0,
            angle_tolerance: 0.0,
            ..self
        }
    }

    pub(crate) fn new_solver(self, tree: &KinematicTree) -> Solver<M> {
        Solver::new(
            tree,
            self.mapper,
            self.n_iterations,
            self.neutral_weight,
            self.position_tolerance,
            self.angle_tolerance,
            self.damping,
        )
    }

    fn new_sequence_solver(self, tree: &Arc<KinematicTree>) -> SequenceSolver<M>
    where
        M: Sync + Send,
    {
        SequenceSolver::new(
            tree,
            self.mapper,
            self.n_iterations,
            self.neutral_weight,
            self.position_tolerance,
            self.angle_tolerance,
            self.damping,
        )
    }
}

/// Total frame count for a `n_segments`-worker `solve_segments_parallel` run
/// to divide evenly into `n_segments` segments of `SEGMENT_LEN` frames each.
pub fn frames_for_n_segments(n_segments: usize) -> usize {
    SEGMENT_LEN * n_segments
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
    config: BenchConfig<M>,
) -> Vec<Duration> {
    let mut solver = config.new_solver(tree);
    for _ in 0..500 {
        let mut state = State::neutral_pose(tree.clone());
        solver.solve(&mut state, black_box(target_obs), false, false);
        black_box(&state);
    }
    let mut samples = Vec::with_capacity(n_calls);
    for _ in 0..n_calls {
        let mut state = State::neutral_pose(tree.clone());
        let t0 = Instant::now();
        solver.solve(&mut state, black_box(target_obs), false, false);
        samples.push(t0.elapsed());
        black_box(&state);
    }
    samples
}

/// Single-thread sequence throughput: `SequenceSolver::solve` warm started
/// across a tiled native-rate sequence (the frame-to-frame motion an actual
/// continuous tracking pipeline would see), default config. A second, fresh
/// `SequenceSolver` is used for the timed pass after warming up once, so the
/// sequence's own frame-to-frame warm-starting is what's measured.
fn bench_single_thread_sequence_throughput<M: Mapper3Dto2D + Sync + Send>(
    tree: &Arc<KinematicTree>,
    sequence: &[Vec<KeypointObservation>],
    config: BenchConfig<M>,
) -> Vec<Duration> {
    let mut seq = config.new_sequence_solver(tree);
    for obs in sequence {
        seq.solve(std::slice::from_ref(black_box(obs)), false, false);
    }

    let mut timed_seq = config.new_sequence_solver(tree);
    let mut samples = Vec::with_capacity(sequence.len());
    for obs in sequence {
        let t0 = Instant::now();
        timed_seq.solve(std::slice::from_ref(black_box(obs)), false, false);
        samples.push(t0.elapsed());
    }
    samples
}

/// Multi-thread sequence throughput: `solve_segments_parallel` on a longer
/// tiled sequence, using exactly `n_workers` threads (joblib convention, see
/// `SequenceSolver::solve_segments_parallel`). Warms up once, then times a
/// second run.
pub fn bench_multithread_sequence_throughput(
    tree: &Arc<KinematicTree>,
    sequence: &[Vec<KeypointObservation>],
    n_workers: isize,
) -> Duration {
    bench_multithread_sequence_throughput_with_config(
        tree,
        sequence,
        n_workers,
        BenchConfig::default_3d(),
    )
}

fn bench_multithread_sequence_throughput_with_config<M: Mapper3Dto2D + Sync + Send>(
    tree: &Arc<KinematicTree>,
    sequence: &[Vec<KeypointObservation>],
    n_workers: isize,
    config: BenchConfig<M>,
) -> Duration {
    let seq = config.new_sequence_solver(tree);
    let _ = seq.solve_segments_parallel(sequence, n_workers, false, false);
    let t0 = Instant::now();
    let results = seq.solve_segments_parallel(sequence, n_workers, false, false);
    let elapsed = t0.elapsed();
    black_box(&results);
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
        bench_single_frame_latency(tree, &target_obs, 10_000, BenchConfig::default_3d()),
    );

    // Early stop disabled (tolerances = 0), so every call runs the full
    // `n_iterations` -- the worst case if a frame never converges early.
    let max_iterations_config = BenchConfig::default_3d().forced_max_iterations();
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
        "SequenceSolver.solve",
        bench_single_thread_sequence_throughput(
            tree,
            &single_thread_sequence,
            BenchConfig::default_3d(),
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
        "solve_segments_parallel             n_frames={:<6} elapsed={elapsed:>9.3?}  throughput={:>10.1} frames/s",
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
/// frame's `target_ego` to 2D via `to_2d`: the 2D counterpart of
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
/// observation is its fixture target reprojected through [`XYView`]. Writes
/// `../plot/results/quickik-rust-2d-xyview-<body>.json` for
/// `../plot/plot_2d_comparison.py` to pick up.
pub fn run_all_2d(tree: &Arc<KinematicTree>, fixtures: &Fixtures, body: &str) {
    println!(
        "quickik Rust benchmark, 2D via XYView (state_dim={})\n",
        tree.state_dim()
    );

    let target_obs = observations_2d_xyview(&fixtures.synthetic_frames[0].target_ego);
    println!("-- single-frame time (latency), default config (adaptive early stop) --");
    let single_frame_latency = summarize(
        "solve()",
        bench_single_frame_latency(
            tree,
            &target_obs,
            10_000,
            BenchConfig::default_with_mapper(XYView),
        ),
    );

    // Early stop disabled (tolerances = 0), so every call runs the full
    // `n_iterations` -- the worst case if a frame never converges early.
    let max_iterations_config = BenchConfig::default_with_mapper(XYView).forced_max_iterations();
    println!(
        "\n-- single-frame time (latency), early stop disabled ({} iterations) --",
        max_iterations_config.n_iterations
    );
    let single_frame_latency_max = summarize(
        "solve() (forced max iterations)",
        bench_single_frame_latency(tree, &target_obs, 10_000, max_iterations_config),
    );

    println!("\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --");
    let single_thread_sequence = tiled_native_rate_sequence_2d(
        &fixtures.native_rate_frames,
        SINGLE_THREAD_N_FRAMES,
        &observations_2d_xyview,
    );
    let single_thread_mean = summarize(
        "SequenceSolver.solve",
        bench_single_thread_sequence_throughput(
            tree,
            &single_thread_sequence,
            BenchConfig::default_with_mapper(XYView),
        ),
    );

    println!(
        "\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop, {MULTITHREAD_N_THREADS} threads) --"
    );
    let sequence = tiled_native_rate_sequence_2d(
        &fixtures.native_rate_frames,
        frames_for_n_segments(MULTITHREAD_N_THREADS),
        &observations_2d_xyview,
    );
    let elapsed = bench_multithread_sequence_throughput_with_config(
        tree,
        &sequence,
        MULTITHREAD_N_THREADS as isize,
        BenchConfig::default_with_mapper(XYView),
    );
    let multithread_fps = sequence.len() as f64 / elapsed.as_secs_f64();
    println!(
        "solve_segments_parallel             n_frames={:<6} elapsed={elapsed:>9.3?}  throughput={:>10.1} frames/s\n",
        sequence.len(),
        multithread_fps,
    );

    write_results_json_2d(
        body,
        single_frame_latency.as_secs_f64() * 1e6,
        single_frame_latency_max.as_secs_f64() * 1e6,
        1.0 / single_thread_mean.as_secs_f64(),
        multithread_fps,
    );
}

/// Writes `../plot/results/quickik-rust-2d-xyview-<body>.json` for
/// `../plot/plot_2d_comparison.py` to pick up. Same schema as
/// [`write_results_json`] plus an `"observation"` field (always `"xyview"`),
/// so both scripts can read files from the same directory without colliding.
fn write_results_json_2d(
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
        "observation": "xyview",
        "single_frame_latency_us": single_frame_latency_us,
        "single_frame_latency_max_us": single_frame_latency_max_us,
        "single_thread_throughput_fps": single_thread_throughput_fps,
        "multi_thread_throughput_fps": multi_thread_throughput_fps,
        "multi_thread_n_threads": MULTITHREAD_N_THREADS,
        "notes": serde_json::Value::Null,
    });
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plot/results");
    std::fs::create_dir_all(&out_dir).expect("failed to create ../plot/results");
    let out_path = out_dir.join(format!("quickik-rust-2d-xyview-{body}.json"));
    std::fs::write(&out_path, serde_json::to_string_pretty(&results).unwrap())
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
}
