# fastik benchmark

Compares fastik's IK solve speed against other IK/kinematics libraries -- KDL, Pinocchio, and RBDL -- on the same task, across fastik's Rust API, Python bindings, and C++ bindings. Two robot bodies are benchmarked:

- **NeuroMechFly**: a fly, 42 leg DOFs, driven by real recorded fly motion capture.
- **G1**: a Unitree humanoid, 29 DOFs, driven by real human motion capture retargeted onto the robot.

Each produces its own chart, `plot/results/comparison_<body>.png`.

## The task being compared

fastik solves *whole-tree* inverse kinematics: given a robot with a free-floating base (unconstrained 6-DOF root, e.g. a pelvis or thorax not bolted to anything) and many tracked keypoints spread across multiple limbs, find the one root pose + joint-angle vector that best matches every keypoint's target position, jointly, in a single solve. This is the actual problem fastik is built for -- a legged or limbed body whose every tracked point moves as one connected system, not a set of independent problems.

Only libraries that can solve *that* problem are compared here. Several well-known IK libraries and algorithms (TRAC-IK, FABRIK, QuIK, and similar) are deliberately not included, because they solve a categorically easier problem: a single fixed-base kinematic chain reaching one end-effector target. There's no way to run them "the same task, just slower" -- doing so would mean either bolting the base down and solving one limb at a time (losing the floating base and the joint multi-limb fit entirely), or writing a different, non-standard algorithm that isn't really that library anymore. Benchmarking them here would produce a number, but not an answer to "how fast is this library at fastik's problem" -- it would measure a different, strictly smaller problem, which flatters *every* excluded library rather than fastik. KDL, Pinocchio, and RBDL are included precisely because each one, in some form, can fit an arbitrary floating base against multiple simultaneous keypoints -- see each library's own `extern/<name>/README.md` for how.

## The data

Every body's fixtures (`assets/fixtures*.json`) contain three kinds of frames, answering three different questions:

- **synthetic**: a handful of poses where the target keypoints were generated from *known* joint angles via forward kinematics, so the correct answer is known in advance. This checks that a solver actually recovers the right pose, not just some pose that happens to fit the keypoints reasonably well.
- **real**: a sparse sample of real, physically-recorded poses, covering a diverse range of motion. This checks how well a solver fits genuine data, which never satisfies the body's kinematic model quite exactly -- no ground truth here, just a residual-fit-quality check (and, for the fly only, a cross-check against an independent reference solver).
- **native rate**: a long, contiguous run of real recorded motion at its original frame rate. This is what the throughput benchmarks replay, because it's the frame-to-frame motion a real continuous tracking pipeline would actually see -- consecutive frames are close to each other, so a solver that warm-starts from the previous frame's answer gets to show that off, the way it would in production.

## The metrics

- **Latency**: how long one `solve()` call takes, starting cold (no warm start) against a fixed target. Reported two ways: the *mean*, under each library's own normal early-stopping behavior, and a *worst case*, with early stopping disabled so every solve runs the same fixed number of optimizer iterations regardless of how easy the target is. The worst case exists because "mean, with early stopping" rewards a library for how often its specific tolerance happens to trigger early on this data, which isn't really the same thing as raw per-iteration speed.
- **Throughput (single thread)**: frames per second solving the native-rate sequence one frame at a time, each warm-started from the previous frame's solution -- the realistic continuous-tracking case.
- **Throughput (multi-thread, 8 threads)**: the same sequence split across 8 threads and solved in parallel, for libraries/bindings that support it.

## Running it

Each language/library's benchmark loops over both bodies on its own and writes one results file per body under `plot/results/`. See:

- `fastik_rust/`, `fastik_python/`, `fastik_cpp/` for fastik's own three bindings (each directory's own header comment has the exact run command).
- `extern/{kdl,pinocchio,rbdl}/README.md` for each external library's build and run steps.
- `preprocessing/README.md` for how G1's assets are generated; `scripts/generate_fixtures.py` for the fly's.

Once whichever benchmarks you want are run, aggregate everything into a chart and table per body:

```sh
python plot/plot_comparison.py
```

`fastik_scaling/` is a separate weak-scaling sweep (1/2/4/8/16 threads, fly body only, Rust only) -- see `run_sweep.sh` and `plot/plot_scaling.py`.

## Results

See `plot/results/comparison_neuromechfly.png` and `comparison_g1.png` for current numbers. In short: fastik's own Gauss-Newton solver is purpose-built for exactly this per-frame IK problem, and is faster than every general tree-IK solver compared here on both bodies, across every metric -- from roughly 2x (RBDL) to two orders of magnitude (KDL) on single-frame latency. Each `extern/<name>/README.md` documents that library's own modeling choices and any caveats worth knowing before trusting its number.

