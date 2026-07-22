# KDL benchmark

Benchmarks Orocos KDL's tree-based inverse kinematics against fastik, on both bodies (see `../../README.md`). See `bench_kdl.cpp`'s header comment for the full modeling-compromise writeup; summarized below.

## Build

KDL and Eigen are built from source into a local prefix (no sudo needed) at `install/` (see `eigen-src/`, `okd-src/` for the sources; only `install/` is needed to build the benchmark itself):

```sh
cd benchmark/extern/kdl
g++ -O3 -std=c++17 \
    -I install/include -I install/include/eigen3 -I install/include/kdl \
    -DFASTIK_ASSETS_DIR='"../../assets"' \
    -o bench_kdl bench_kdl.cpp \
    -L install/lib -lorocos-kdl -Wl,-rpath,install/lib -pthread
./bench_kdl
```

`json.hpp` and `forward_kinematics.hpp` are verbatim copies of `../../fastik_cpp/`'s (dependency-free JSON reader + FK replica), kept local so this directory builds standalone.

## Modeling compromises

1. **Floating base**: KDL has no native 6-DOF floating joint. fastik's free-floating "thorax" root (a translation + a singularity-free unit quaternion) is represented here as 6 scalar joints in series (TransX/Y/Z, RotZ/Y/X) -- a sequential Euler-angle-like parametrization. Functionally workable (same reachable pose space, modulo gimbal lock) but not mathematically identical to fastik's quaternion root.
2. **Position-only IK via a full-SE(3) solver**: `TreeIkSolverVel_wdls` natively solves for full 6D pose per endpoint; fastik only fits 3D positions. The 3 rotational rows of every endpoint's task-space weight matrix are zeroed (`setWeightTS`) so orientation error never drives the solve.
3. **Multi-thread**: KDL has no built-in parallel solve. `multi_thread_throughput_fps` uses simple contiguous chunking (8 independent chunks, each internally warm-started but cold at its own start), not fastik's overlap-stitched segmented solve.

## Solve loop

KDL's own convenience wrapper, `TreeIkSolverPos_NR_JL`, has only one early-stop check:

```cpp
double res = iksolver.CartToJnt(q_out, delta_twists, delta_q);
if (res < eps) return res;
```

`res` is `TreeIkSolverVel_wdls::CartToJnt`'s return value -- the L2 norm of the *residual* (target minus current position) across all 30 endpoints at once (`Wy_t.norm()` in `treeiksolvervel_wdls.cpp`). On real, imperfectly-fittable mocap data this residual has a floor around 0.08-0.13, far above any `eps` tight enough to be a meaningful convergence criterion, so `res < eps` never triggers on real frames and every solve would silently burn the entire iteration cap. `solve()` in `bench_kdl.cpp` instead reimplements `TreeIkSolverPos_NR_JL`'s own per-iteration math directly against `TreeIkSolverVel_wdls` (FK each endpoint, diff to target, one velocity-IK step, apply the update), adding a step-size check in its place: stop once the max absolute component of the per-iteration joint delta drops below `kStepTol = 1e-3` -- fastik's own `position_tolerance`/`angle_tolerance` value. This cuts mean iterations per frame from a fixed 100 to ~7-8 on the fly body, with no change in residual accuracy -- the 0.08-0.13 floor is a real property of fitting noisy mocap data to a rigid rotation-axis model, not something more iterations can fix.

## Why KDL remains slower than fastik/RBDL

Even with mean iterations per frame down to ~7-8 (comparable to fastik's and RBDL's own typical counts on this data), **per-iteration cost itself is far higher** than RBDL's or fastik's: `TreeIkSolverVel_wdls::CartToJnt` computes a dense SVD (`svd_eigen_HH`) of the full weighted task-space Jacobian -- **180 rows** (30 endpoints x 6, even though only 90 of those rows carry nonzero weight after zeroing rotation) **by 48 columns** -- every single iteration. RBDL's `InverseKinematicsConstraintSet`, by contrast, forms the 48x48 normal-equations matrix `J^T J` once per iteration and solves it with a QR decomposition -- a fundamentally cheaper operation at this problem size. This is a genuine algorithmic/implementation difference between the two libraries' provided building blocks, not something a stopping-criterion fix can close.

## Results

Current numbers: `../../plot/results/kdl-<body>.json`, or the comparison chart/table in `../../README.md` (which also explains why KDL's bar needs a capped/truncated treatment on G1 specifically). Not reproduced here to avoid a second, driftable copy of the same data -- KDL is consistently the slowest whole-tree solver in the comparison, for the per-iteration-cost reason described above (SVD vs. normal-equations), not a stopping-criterion artifact.

