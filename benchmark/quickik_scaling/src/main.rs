//! One weak-scaling data point for `solve_sequence_segmented_parallel`:
//! fixes frames-per-thread at `quickik_benchmark::perf::SEGMENT_LEN` (one
//! segment per thread) and scales total frames with however many threads
//! [`std::thread::available_parallelism`] currently reports, so the amount
//! of work per thread stays constant as thread count grows. Ideal weak
//! scaling keeps `elapsed` constant across thread counts.
//!
//! Run this binary repeatedly under `taskset -c 0`, `-c 0-1`, `-c 0-3`,
//! `-c 0-7`, `-c 0-15` (which constrains the CPU affinity mask
//! `available_parallelism` reads) to sweep 1/2/4/8/16 threads -- see
//! `../README.md` and `run_sweep.sh`. Each run appends/replaces its own
//! `n_threads` entry in `../plot/results/quickik-scaling.json` (a JSON array,
//! not the single-object shape `RESULTS_SCHEMA.md` describes for the other
//! benchmarks, since this is a sweep over several data points rather than
//! one result) for `../plot/plot_scaling.py` to chart.

use std::sync::Arc;

use quickik::body_plan::KinematicTree;
use quickik_benchmark::fixtures;
use quickik_benchmark::perf::{
    bench_multithread_sequence_throughput, frames_for_n_segments, tiled_native_rate_sequence,
};

fn write_results_json(n_threads: usize, total_frames: usize, elapsed_s: f64, throughput_fps: f64) {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plot/results");
    std::fs::create_dir_all(&out_dir).expect("failed to create ../plot/results");
    let out_path = out_dir.join("quickik-scaling.json");

    let mut points: Vec<serde_json::Value> = std::fs::read_to_string(&out_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    points.retain(|p| p["n_threads"].as_u64() != Some(n_threads as u64));
    points.push(serde_json::json!({
        "n_threads": n_threads,
        "total_frames": total_frames,
        "elapsed_s": elapsed_s,
        "throughput_fps": throughput_fps,
    }));
    points.sort_by_key(|p| p["n_threads"].as_u64().unwrap());

    std::fs::write(&out_path, serde_json::to_string_pretty(&points).unwrap())
        .expect("failed to write ../plot/results/quickik-scaling.json");
}

fn main() {
    let assets_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets");
    let tree = Arc::new(KinematicTree::from_json_file(
        assets_dir.join("neuromechfly_ypr_legs.json"),
    ));
    let fixtures = fixtures::load(assets_dir.join("fixtures.json"));

    let n_threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let total_frames = frames_for_n_segments(n_threads);
    let sequence = tiled_native_rate_sequence(&fixtures.native_rate_frames, total_frames);

    let elapsed = bench_multithread_sequence_throughput(&tree, &sequence);
    let throughput_fps = total_frames as f64 / elapsed.as_secs_f64();
    println!(
        "n_threads={n_threads:<3} total_frames={total_frames:<6} elapsed={elapsed:>9.3?}  \
         throughput={throughput_fps:>10.1} frames/s",
    );

    write_results_json(
        n_threads,
        total_frames,
        elapsed.as_secs_f64(),
        throughput_fps,
    );
}
