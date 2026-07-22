# Pinocchio benchmark

Benchmarks [Pinocchio](https://github.com/stack-of-tasks/pinocchio) against fastik, on both bodies (see `../../README.md`). Methodology mirrors `../../fastik_python/bench.py` and `../../fastik_rust/src/perf.rs` exactly (same fixtures, same metrics, same config values) so the numbers are directly comparable.

## Running

Pinocchio's pip wheels don't support Python 3.13+, so a dedicated 3.12 venv is used:

```
cd /path/to/fastik/benchmark/extern/pinocchio
.venv312/bin/python bench_pinocchio.py
```

Prints a correctness cross-check (synthetic exact-fit frames) followed by the 3 performance numbers, and writes one `../../plot/results/pinocchio-<body>.json` per body.

## Modeling compromises

- **No built-in IK solver.** Pinocchio only provides forward kinematics and Jacobians. `bench_pinocchio.py` implements its own Gauss-Newton/ Levenberg-Marquardt loop (`solve_ik`) on top of `pin.computeJointJacobians`/`pin.getFrameJacobian`, matching fastik's own solver (`src/solver.rs`) as closely as possible: stacked 3-row position-only residuals per keypoint, an LM damping term on the normal equations' diagonal, a neutral-pose Tikhonov prior on the leg DOFs, and the same early-stopping rule (stop once the position and angle components of the update both drop below `1e-3`, capped at 10 iterations) -- so `n_iterations` is a ceiling, not a fixed cost, exactly as in fastik. Updates are applied via `pin.integrate(model, q, delta)` rather than naive addition, since the free-flyer root's quaternion components can't be updated by simple addition.
- **Mirrored (right-side) leg axes.** fastik's body plan encodes mirrored legs via signed rotation axes (e.g. `[-1, 0, 0]` for `rf`/`rm`/`rh`'s coxa yaw/roll). Pinocchio's `JointModelRX/RY/RZ` only rotate about the *positive* axis. Rather than bake an extra 180-degree rotation into the joint placement (which would also flip the sign of every subsequent child-joint axis in that subtree), each mirrored DOF is driven with a sign-flipped angle: rotating by `-theta` about `+X` is identical to rotating by `theta` about `-X` (`R(-n, -t) = R(n, t)`), so a `JointModelRX` reproduces the JSON's mirrored DOF exactly when its own angle is negated. This sign is applied once when building the neutral configuration (`dof_signs * neutral_angle`) and once more when converting a solved angle back to the JSON's convention for the correctness check; the IK solve itself works entirely in Pinocchio's own (already-consistent) coordinates, so no Jacobian-column flipping is needed.
- **Multi-thread throughput = multiprocessing.** Python's GIL means CPU-bound numpy/Pinocchio code can't run in real parallel threads. The `multi_thread_throughput_fps` metric instead uses `multiprocessing.Pool(8)`: a 2400-frame tiled sequence (8 x 300 native-rate frames) is split into 8 contiguous 300-frame chunks, each solved in its own process (warm-started within the chunk, cold -- from the neutral pose -- at the chunk's start). Wall-clock time is measured from `pool.map` dispatch to all chunks returning (model construction happens once per worker in the pool initializer, outside the timed region, analogous to fastik-rust building its `Arc<KinematicTree>` once before timing). This is not directly comparable to fastik's in-process segmented thread-pool solve -- noted in the results JSON's `notes` field.

## Results

Current numbers: `../../plot/results/pinocchio-<body>.json`, or the comparison chart/table in `../../README.md`. Not reproduced here to avoid a second, driftable copy of the same data.

One pattern worth knowing when reading those numbers: the single-thread warm-started sequence can be *slower* per call than the cold-start single-frame latency number, which looks backwards at first glance -- it's a property of the fixture data, not a bug. The latency benchmark's fixed synthetic target is an easy, near-exact fit reachable from neutral in very few Gauss-Newton iterations, while the native-rate (real-motion) sequence's targets are noisier and take more iterations per frame even when warm-started. All three numbers here are Python-loop-and-numpy-allocation-bound (Pinocchio's own C++ core is fast; the per-iteration Python `for` loop over every tracked keypoint, plus per-call numpy allocations, dominates) -- that overhead, not Pinocchio's FK/Jacobian computation itself, is the main cost being measured.

## Native C++ benchmark

`bench_pinocchio_cpp.cpp` is a native C++ port of `bench_pinocchio.py`: same model construction and Gauss-Newton/LM math, ported line for line to Pinocchio's C++ API, with the outer-loop linear algebra done in Eigen (`colPivHouseholderQr`, preallocated `jtj`/`jtr`/`J` buffers, no per-iteration heap allocation) instead of numpy. It exists to measure Pinocchio's own C++ speed on this workload without Python/numpy overhead.

### Build

Pinocchio's C++ headers/libs and Boost are already available inside the Python venv's `cmeel.prefix` (no separate C++ install needed); Eigen (header-only) is reused from `../rbdl/eigen-src`. Pinocchio's joint-model `boost::variant` has more alternatives (25) than Boost's default `BOOST_MPL_LIMIT_LIST_SIZE` (20), so the same three defines Pinocchio's own `pinocchioTargets.cmake` uses for downstream consumers are required:

```sh
cd benchmark/extern/pinocchio
CMEEL=.venv312/lib/python3.12/site-packages/cmeel.prefix
g++ -O3 -std=c++17 -pthread \
    -DBOOST_MPL_LIMIT_LIST_SIZE=30 -DBOOST_MPL_LIMIT_VECTOR_SIZE=30 \
    -DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS -DBOOST_FUSION_INVOKE_MAX_ARITY=12 \
    -I "$CMEEL/include" -I ../rbdl/eigen-src \
    -L "$CMEEL/lib" -Wl,-rpath,'$ORIGIN'/"$CMEEL/lib" \
    -o bench_pinocchio_cpp bench_pinocchio_cpp.cpp -lpinocchio_default
LD_LIBRARY_PATH="$CMEEL/lib" ./bench_pinocchio_cpp
```

`json.hpp` is a verbatim copy of `../rbdl/json.hpp` (dependency-free JSON reader), kept local so this directory builds standalone. Unlike the RBDL/KDL benchmarks, the JSON body plan is parsed directly in double precision here (not via `../rbdl/forward_kinematics.hpp`'s float-based `BodyPlan`), to match Python's float64 arrays exactly. Modeling is otherwise identical to `bench_pinocchio.py`: thorax `JointModelFreeFlyer` root, one `JointModelRX/ RY/RZ` per scalar leg DOF (mirrored-leg signed axes handled by negating the driven angle), all 30 leg keypoints tracked via `OP_FRAME` operational frames and fit jointly, same fastik `SolverConfig::default()` tuning.

### Results

Current numbers: `../../plot/results/pinocchio-cpp-<body>.json`, alongside the Python benchmark's own results and the comparison chart/table in `../../README.md`. Not reproduced here to avoid a second, driftable copy of the same data.

Going native cuts latency and roughly doubles throughput versus the Python benchmark, confirming the Python/numpy overhead described above is real -- but Pinocchio C++ is still slower than RBDL's native C++ numbers on this workload. Most of that gap is Pinocchio's own `getFrameJacobian` API: with many tracked keypoints, the per-frame Jacobian extractions (one call per keypoint) cost more combined than the rest of the loop (Eigen `jtj`/`jtr` accumulation + the QR solve). That's a property of Pinocchio's per-frame Jacobian API for a many-keypoints-per-solve problem, not a Python-vs-C++ artifact -- RBDL's `InverseKinematicsConstraintSet` avoids it by computing every point constraint's Jacobian contribution in one internal pass rather than exposing a per-frame extraction call the caller has to invoke repeatedly.

`multi_thread_throughput_fps` uses the same simple contiguous-chunking scheme as `bench_rbdl.cpp` (8 `std::thread` workers, each with its own `pinocchio::Data` and warm-started within its own chunk but cold at the chunk's start, all sharing one read-only `pinocchio::Model`) -- comparable in spirit to the RBDL C++ benchmark's multi-thread metric, unlike this directory's own Python benchmark, which uses multiprocessing instead.

