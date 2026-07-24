# QuickIK benchmark

Compares QuickIK's IK solve speed against KDL, Pinocchio, and RBDL, across QuickIK's Rust API, Python bindings, and C++ bindings, on two bodies: NeuroMechFly (a fly) and G1 (a Unitree humanoid). See the [Benchmarks docs page](../docs/benchmarks.md) for what's being compared, why, and the current results – this file only covers how to reproduce them.

## Running it

To run everything at once (assuming every benchmark below is already built/set up on your machine) and regenerate both charts, use `scripts/run_all_benchmarks.sh` -- it checks all prerequisites up front and errors out listing whatever's missing before running anything. Otherwise, each language/library's benchmark loops over both bodies on its own and writes one results file per body under `plot/results/`. See:

- `quickik_rust/`, `quickik_python/`, `quickik_cpp/` for QuickIK's own three bindings (each directory's own header comment has the exact run command).
- `extern/{kdl,pinocchio,rbdl}/README.md` for each external library's build and run steps.
- `preprocessing/README.md` for how G1's assets are generated; `scripts/generate_fixtures.py` for the fly's.

Once whichever benchmarks you want are run, aggregate everything into a chart and table per body (with `devtools-pyenv`'s shared venv active, for matplotlib/numpy/scipy):

```sh
python plot/plot_comparison.py
```

`quickik_scaling/` is a separate weak-scaling sweep (1/2/4/8/16 threads, fly body only, Rust only) – see `run_sweep.sh` and `plot/plot_scaling.py`.

To visually sanity-check a fit, `plot/render_video.py` renders a side-by-side comparison video (both bodies) overlaying real mocap keypoints against QuickIK's solved skeleton over a contiguous warm-started sequence – see that script's own header comment for the run command.

To rebuild the docs site (including these charts and the Rust API reference) after running any of the above, run `../docs/build.sh` from the repo root.

## Reducing measurement noise

On a shared, multi-tasking machine, other processes competing for CPU time can add double-digit percentage noise to these numbers – enough to look like a real regression. `taskset -c <cores>` pins a benchmark process to specific CPU cores so the scheduler can't migrate it mid-run; pick currently idle cores first (e.g. with `mpstat -P ALL 1 1`):

```sh
taskset -c 0,5,6,8,10,11,13,15 ./target/release/quickik-benchmark
```

Pin to *at least as many cores as the multi-thread benchmark's worker count* (`MULTITHREAD_N_THREADS` in `quickik_rust/src/perf.rs`, 8 by default) – pinning to fewer cores than that forces its worker threads to share them, which serializes the very parallelism that metric is measuring and craters its throughput (the single-frame-latency metrics are single-threaded and unaffected by this). `quickik_scaling`'s sweep goes up to 16 workers, so pin it to all available cores (`taskset -c 0-15` on a 16-core machine) rather than a subset. This doesn't reserve the cores exclusively – without root/cgroups, another process can still land on them – but it removes migration jitter and noticeably tightens the p95/p99 tail latencies. For a real before/after comparison, always pin both runs to the same cores.
