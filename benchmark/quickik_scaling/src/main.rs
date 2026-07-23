//! Weak-scaling sweep for `solve_sequence_segmented_parallel`: fixes
//! frames-per-worker at `quickik_benchmark::perf::SEGMENT_LEN` (one segment
//! per worker) and scales total frames with `n_workers`, so the amount of
//! work per worker stays constant as `n_workers` grows. Ideal weak scaling
//! keeps `elapsed` constant across the sweep.
//!
//! `n_workers` is passed explicitly via `ParallelSolveConfig::n_workers` (see
//! `../README.md`), so -- unlike before that field existed -- this no longer
//! needs `taskset` to vary the thread count; one run of this binary sweeps
//! 1/2/4/8/16 workers itself. Writes `../plot/results/quickik-scaling.json`
//! (a JSON array, not the single-object shape `RESULTS_SCHEMA.md` describes
//! for the other benchmarks, since this is a sweep over several data points
//! rather than one result) for `../plot/plot_scaling.py` to chart.

use std::sync::Arc;

use quickik::body_plan::KinematicTree;
use quickik_benchmark::fixtures;
use quickik_benchmark::perf::{
    bench_multithread_sequence_throughput, frames_for_n_segments, tiled_native_rate_sequence,
};

const N_WORKERS_SWEEP: [usize; 5] = [1, 2, 4, 8, 16];

fn write_results_json(points: &[serde_json::Value]) {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plot/results");
    std::fs::create_dir_all(&out_dir).expect("failed to create ../plot/results");
    let out_path = out_dir.join("quickik-scaling.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(points).unwrap())
        .expect("failed to write ../plot/results/quickik-scaling.json");
}

fn main() {
    let assets_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets");
    let tree = Arc::new(KinematicTree::from_json_file(
        assets_dir.join("neuromechfly_ypr_legs.json"),
    ));
    let fixtures = fixtures::load(assets_dir.join("fixtures.json"));

    let mut points = Vec::with_capacity(N_WORKERS_SWEEP.len());
    for n_workers in N_WORKERS_SWEEP {
        let total_frames = frames_for_n_segments(n_workers);
        let sequence = tiled_native_rate_sequence(&fixtures.native_rate_frames, total_frames);

        let elapsed = bench_multithread_sequence_throughput(&tree, &sequence, n_workers as isize);
        let throughput_fps = total_frames as f64 / elapsed.as_secs_f64();
        println!(
            "n_workers={n_workers:<3} total_frames={total_frames:<6} elapsed={elapsed:>9.3?}  \
             throughput={throughput_fps:>10.1} frames/s",
        );
        points.push(serde_json::json!({
            "n_threads": n_workers,
            "total_frames": total_frames,
            "elapsed_s": elapsed.as_secs_f64(),
            "throughput_fps": throughput_fps,
        }));
    }
    write_results_json(&points);
}
