# KDL benchmark

Benchmarks Orocos KDL's tree-based inverse kinematics against fastik, on the
same NeuroMechFly body plan and fixtures as the other benchmarks in this
repo. See `bench_kdl.cpp`'s header comment for the full modeling-compromise
writeup; summarized below.

## Build

KDL and Eigen are built from source into a local prefix (no sudo needed) at
`install/` (see `eigen-src/`, `okd-src/` for the sources; only `install/` is
needed to build the benchmark itself):

```sh
cd benchmark/extern/kdl
g++ -O3 -std=c++17 \
    -I install/include -I install/include/eigen3 -I install/include/kdl \
    -DFASTIK_ASSETS_DIR='"../../assets"' \
    -o bench_kdl bench_kdl.cpp \
    -L install/lib -lorocos-kdl -Wl,-rpath,install/lib -pthread
./bench_kdl
```

`json.hpp` and `forward_kinematics.hpp` are verbatim copies of
`../../fastik_cpp/`'s (dependency-free JSON reader + FK replica), kept local
so this directory builds standalone.

## Modeling compromises

1. **Floating base**: KDL has no native 6-DOF floating joint. fastik's
   free-floating "thorax" root (a translation + a singularity-free unit
   quaternion) is represented here as 6 scalar joints in series (TransX/Y/Z,
   RotZ/Y/X) -- a sequential Euler-angle-like parametrization. Functionally
   workable (same reachable pose space, modulo gimbal lock) but not
   mathematically identical to fastik's quaternion root.
2. **Position-only IK via a full-SE(3) solver**: `TreeIkSolverVel_wdls`
   natively solves for full 6D pose per endpoint; fastik only fits 3D
   positions. The 3 rotational rows of every endpoint's task-space weight
   matrix are zeroed (`setWeightTS`) so orientation error never drives the
   solve.
3. **Multi-thread**: KDL has no built-in parallel solve. `multi_thread_throughput_fps`
   uses simple contiguous chunking (8 independent chunks, each internally
   warm-started but cold at its own start), not fastik's overlap-stitched
   segmented solve.

## A fairness bug found and fixed: the solve loop, not just the tolerance

An earlier version of this benchmark used KDL's own convenience wrapper,
`TreeIkSolverPos_NR_JL`, with `eps=1e-3` (matching fastik's tolerance) and
`maxiter=100`. That version measured **~102ms/frame** -- roughly 700x slower
than fastik. Before accepting that as "KDL is just slow," we read
`treeiksolverpos_nr_jl.cpp` directly: its *only* early-stop check is

```cpp
double res = iksolver.CartToJnt(q_out, delta_twists, delta_q);
if (res < eps) return res;
```

`res` is `TreeIkSolverVel_wdls::CartToJnt`'s return value -- the L2 norm of
the *residual* (target minus current position) across all 30 endpoints at
once (`Wy_t.norm()` in `treeiksolvervel_wdls.cpp`). On real, imperfectly-
fittable mocap data this residual has a floor around 0.08-0.13 (see
Accuracy below) -- far above any `eps` tight enough to be a meaningful
convergence criterion. So `res < eps` **never triggers on real frames**,
regardless of `eps`, and every single solve silently burned the entire
100-iteration cap. This is a solve loop with a non-functional early-stop for
this kind of data, not a tuning choice -- the same class of bug the RBDL
benchmark's `README.md` describes finding and fixing, though the specific
mechanism differs (RBDL's own solver *did* have a working step-size check,
just tuned too tight; KDL's convenience wrapper has no step-size check at
all).

**Fix**: `solve()` in `bench_kdl.cpp` reimplements `TreeIkSolverPos_NR_JL`'s
own per-iteration math directly against `TreeIkSolverVel_wdls` (FK each
endpoint, diff to target, one velocity-IK step, apply the update), adding
the missing check: stop once the max absolute component of the per-iteration
joint delta drops below `kStepTol = 1e-3` -- literally fastik's own
`position_tolerance`/`angle_tolerance` value, applied to a step-size
criterion instead of `TreeIkSolverPos_NR_JL`'s residual-only one. A sweep
over `step_tol` (using this hand-rolled loop) against the 300-frame
native-rate warm-started sequence:

| step_tol | mean iterations | max iterations | mean rms residual |
|---|---|---|---|
| 1e-6 | 17.4 | 80 | (same floor every time) |
| 1e-4 | 10.8 | 51 | (same floor every time) |
| 1e-3 | 7.6 | 37 | (same floor every time) |
| 1e-2 | 4.6 | 22 | (same floor every time) |

Residual quality is unaffected by `step_tol` across this entire range (the
0.08-0.13 floor is a real property of fitting noisy mocap data to a rigid
rotation-axis model, not something more iterations can fix -- same
phenomenon RBDL's README documents). `step_tol=1e-3` was kept as the final
choice specifically because it's fastik's own number, not because it was
fastest.

## Even after the fix, KDL is still much slower than fastik/RBDL -- and that part *is* genuine

With the fixed solve loop, mean iterations per frame drop from "always 100"
to ~7-8 (comparable to fastik's and RBDL's own typical iteration counts on
this data). But **per-iteration cost itself is still far higher** than
RBDL's or fastik's: `TreeIkSolverVel_wdls::CartToJnt` computes a dense SVD
(`svd_eigen_HH`) of the full weighted task-space Jacobian -- **180 rows** (30
endpoints x 6, even though only 90 of those rows carry nonzero weight after
zeroing rotation) **by 48 columns** -- every single iteration. RBDL's
`InverseKinematicsConstraintSet`, by contrast, forms the 48x48 normal-
equations matrix `J^T J` once per iteration and solves it with a QR
decomposition -- a fundamentally cheaper operation at this problem size.
This is a genuine algorithmic/implementation difference between the two
libraries' provided building blocks, not something a stopping-criterion fix
can close.

## Results

```
-- single-frame time (latency) --
solve() (cold)   n=20000  mean=6301.1us   throughput=158.7 calls/s

-- single-thread sequence throughput (native-rate frames, warm-started) --
solve() (warm)   n=300    mean=7980.9us   throughput=125.3 calls/s

-- multi-thread sequence throughput (8 contiguous chunks, 8 threads) --
n_frames=1600    elapsed=1.841s   throughput=869.3 frames/s
```

(see `../../plot/results/kdl.json` for the exact numbers from the last run)

The fix above is a real, substantial win: ~102ms -> ~6.3ms single-frame
latency (~16x), ~9.85 -> ~125.3 fps single-thread throughput (~12.7x),
~70.5 -> ~869.3 fps multi-thread throughput (~12.3x) -- all for identical
residual accuracy. But KDL remains roughly **20-40x slower than fastik and
RBDL** on this same whole-tree problem (fastik-cpp: ~161us / ~4,690 fps /
~30,400 fps; RBDL: ~310us / ~2,364 fps / ~15,700 fps), consistent with the
~19x per-iteration-cost gap predicted above from the SVD-vs-normal-equations
difference.

**KDL is included in `plot/plot_results.py`'s main comparison table and
chart** alongside the other whole-tree solvers (RBDL, Pinocchio, fastik) --
now that the early-stop bug above is fixed, its numbers reflect a fair
implementation, even though it remains the slowest whole-tree solver here
for the genuine algorithmic reason described above. On the single-frame
latency panel specifically, KDL's bar (~6.3ms) is roughly 40x every other
bar, so it dominates that linear-scale chart the way a large outlier always
will; the value labels on every bar keep the smaller ones readable anyway.
