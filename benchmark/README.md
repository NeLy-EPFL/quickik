# QuickIK benchmark

Compares QuickIK's IK solve speed against KDL, Pinocchio, and RBDL, across QuickIK's Rust API, Python bindings, and C++ bindings, on two bodies: NeuroMechFly (a fly) and G1 (a Unitree humanoid). See the [Benchmarks docs page](../docs/benchmarks.md) for what's being compared, why, and the current results -- this file only covers how to reproduce them.

## Running it

Each language/library's benchmark loops over both bodies on its own and writes one results file per body under `plot/results/`. See:

- `quickik_rust/`, `quickik_python/`, `quickik_cpp/` for QuickIK's own three bindings (each directory's own header comment has the exact run command).
- `extern/{kdl,pinocchio,rbdl}/README.md` for each external library's build and run steps.
- `preprocessing/README.md` for how G1's assets are generated; `scripts/generate_fixtures.py` for the fly's.

Once whichever benchmarks you want are run, aggregate everything into a chart and table per body:

```sh
python plot/plot_comparison.py
```

`quickik_scaling/` is a separate weak-scaling sweep (1/2/4/8/16 threads, fly body only, Rust only) -- see `run_sweep.sh` and `plot/plot_scaling.py`.

To visually sanity-check a fit, `plot/render_video.py` renders a side-by-side comparison video (both bodies) overlaying real mocap keypoints against QuickIK's solved skeleton over a contiguous warm-started sequence -- see that script's own header comment for the run command.

To rebuild the docs site (including these charts and the Rust API reference) after running any of the above, run `../docs/build.sh` from the repo root.
