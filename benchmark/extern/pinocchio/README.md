# Pinocchio benchmark

Benchmarks [Pinocchio](https://github.com/stack-of-tasks/pinocchio) against
fastik on the neuromechfly body plan (`../../assets/neuromechfly_ypr_legs.json`):
thorax free-flyer root + 6 legs x 7 DOFs each (42 DOFs total, 30 tracked leg
keypoints). Methodology mirrors `../../fastik_python/bench.py` and
`../../fastik_rust/src/perf.rs` exactly (same fixtures, same 3 metrics, same
config values) so the numbers are directly comparable.

## Running

Pinocchio's pip wheels don't support Python 3.13+, so a dedicated 3.12 venv
is used:

```
cd /path/to/fastik/benchmark/extern/pinocchio
.venv312/bin/python bench_pinocchio.py
```

Prints a correctness cross-check (synthetic exact-fit frames) followed by the
3 performance numbers, and writes `../../plot/results/pinocchio.json`.

## Modeling compromises

- **No built-in IK solver.** Pinocchio only provides forward kinematics and
  Jacobians. `bench_pinocchio.py` implements its own Gauss-Newton/
  Levenberg-Marquardt loop (`solve_ik`) on top of
  `pin.computeJointJacobians`/`pin.getFrameJacobian`, matching fastik's own
  solver (`src/solver.rs`) as closely as possible: stacked 3-row
  position-only residuals per keypoint, an LM damping term on the normal
  equations' diagonal, a neutral-pose Tikhonov prior on the leg DOFs, and the
  same early-stopping rule (stop once the position and angle components of
  the update both drop below `1e-3`, capped at 10 iterations) -- so
  `n_iterations` is a ceiling, not a fixed cost, exactly as in fastik.
  Updates are applied via `pin.integrate(model, q, delta)` rather than naive
  addition, since the free-flyer root's quaternion components can't be
  updated by simple addition.
- **Mirrored (right-side) leg axes.** fastik's body plan encodes mirrored
  legs via signed rotation axes (e.g. `[-1, 0, 0]` for `rf`/`rm`/`rh`'s coxa
  yaw/roll). Pinocchio's `JointModelRX/RY/RZ` only rotate about the
  *positive* axis. Rather than bake an extra 180-degree rotation into the
  joint placement (which would also flip the sign of every subsequent
  child-joint axis in that subtree), each mirrored DOF is driven with a
  sign-flipped angle: rotating by `-theta` about `+X` is identical to
  rotating by `theta` about `-X` (`R(-n, -t) = R(n, t)`), so a `JointModelRX`
  reproduces the JSON's mirrored DOF exactly when its own angle is negated.
  This sign is applied once when building the neutral configuration
  (`dof_signs * neutral_angle`) and once more when converting a solved angle
  back to the JSON's convention for the correctness check; the IK solve
  itself works entirely in Pinocchio's own (already-consistent) coordinates,
  so no Jacobian-column flipping is needed. Verified with a finite-difference
  check of a mirrored leg's frame Jacobian (max abs error ~6e-7, i.e. exactly
  the finite-difference step size) before benchmarking.
- **Multi-thread throughput = multiprocessing.** Python's GIL means CPU-bound
  numpy/Pinocchio code can't run in real parallel threads. The
  `multi_thread_throughput_fps` metric instead uses `multiprocessing.Pool(8)`:
  a 2400-frame tiled sequence (8 x 300 native-rate frames) is split into 8
  contiguous 300-frame chunks, each solved in its own process (warm-started
  within the chunk, cold -- from the neutral pose -- at the chunk's start).
  Wall-clock time is measured from `pool.map` dispatch to all chunks
  returning (model construction happens once per worker in the pool
  initializer, outside the timed region, analogous to fastik-rust building
  its `Arc<KinematicTree>` once before timing). This is not directly
  comparable to fastik's in-process segmented thread-pool solve -- noted in
  the results JSON's `notes` field.

## Results (last run)

```
== Synthetic exact-fit frames (bug hunt) ==
 frame          kpt rms          kpt max      angle err deg
     0         0.002567         0.004643             3.4883
     ...
     7         0.001599         0.002947             2.0736

Pinocchio benchmark (nq=49, nv=48)

-- single-frame time (latency), no warm start --
solve_ik()   n=20000   mean=1179.3us   throughput=847.9 calls/s

-- single-thread sequence throughput (native-rate frames, warm start) --
solve_ik() (warm-started)   n=300   mean=1598.1us   throughput=625.8 calls/s

-- multi-thread sequence throughput (multiprocessing, 8 processes x 300 frames each) --
solve (multiprocess)   n_frames=2400   elapsed=732.3ms   throughput=3277.3 frames/s
```

- `single_frame_latency_us` = 1179.3
- `single_thread_throughput_fps` = 625.8
- `multi_thread_throughput_fps` = 3277.3

The single-thread warm-started sequence is *slower* per call than the
cold-start single-frame benchmark, which looks backwards at first glance --
but it's a property of the fixture data, not a bug: the synthetic
`target_ego` used for the latency benchmark is an exact fit reachable from
neutral in ~5 Gauss-Newton iterations, while the native-rate (real-motion)
sequence's targets are noisier/less exactly reachable and take ~6-10
iterations per frame even when warm-started (checked directly by
instrumenting `solve_ik`'s iteration count). All three numbers are
Python-loop-and-numpy-allocation-bound (Pinocchio's own C++ core is fast;
the per-iteration Python `for` loop over 30 keypoints plus per-call numpy
allocations dominate), so they are expected to trail fastik's Rust and C++
bindings substantially -- that overhead, not Pinocchio's FK/Jacobian
computation itself, is the main cost being measured here.

## Native C++ benchmark

`bench_pinocchio_cpp.cpp` is a from-scratch native C++ port of this
benchmark, built to answer a specific question: is `pinocchio.json`'s
~1179us/626fps/3277fps really Pinocchio's speed, or mostly Python overhead?
Our RBDL benchmark (`../rbdl/bench_rbdl.cpp`) is pure C++ start-to-finish and
came in much faster (~309us/2364fps/15710fps), which contradicts Carpentier
et al. (IROS 2019, https://laas.hal.science/hal-01866228), which reports
Pinocchio's core algorithms are competitive with or faster than RBDL's when
both are benchmarked natively in C++. A quick diagnostic instrumenting
`bench_pinocchio.py`'s `solve_ik` found Pinocchio's actual C++ calls
(`computeJointJacobians`/`updateFramePlacements`/30x `getFrameJacobian`) take
only ~17.8us/iteration, while the surrounding pure-Python/numpy bookkeeping
(building the 42x42 normal-equations matrix in a Python `for` loop,
`np.linalg.solve`, LM damping, the neutral-pose prior, `pin.integrate`) takes
~145us -- i.e. ~92% of the Python benchmark's measured time is Python/numpy
overhead, not Pinocchio's own speed. This file tests that hypothesis by
porting `build_full_model`/`solve_ik` line for line to Pinocchio's C++ API,
with the outer-loop linear algebra done in Eigen (`colPivHouseholderQr`,
preallocated `jtj`/`jtr`/`J` buffers reused across iterations and calls, no
per-iteration heap allocation) instead of numpy.

### Build

Pinocchio's C++ headers/libs and Boost are already available inside the
Python venv's `cmeel.prefix` (no separate C++ install needed); Eigen
(header-only) is reused from `../rbdl/eigen-src`. Pinocchio's joint-model
`boost::variant` has more alternatives (25) than Boost's default
`BOOST_MPL_LIMIT_LIST_SIZE` (20), so the same three defines Pinocchio's own
`pinocchioTargets.cmake` uses for downstream consumers are required:

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

`json.hpp` is a verbatim copy of `../rbdl/json.hpp` (dependency-free JSON
reader), kept local so this directory builds standalone. Unlike the RBDL/KDL
benchmarks, the JSON body plan is parsed directly in double precision here
(not via `../rbdl/forward_kinematics.hpp`'s float-based `BodyPlan`), to match
Python's float64 arrays exactly. Modeling is otherwise identical to
`bench_pinocchio.py`: thorax `JointModelFreeFlyer` root, one `JointModelRX/
RY/RZ` per scalar leg DOF (mirrored-leg signed axes handled by negating the
driven angle), all 30 leg keypoints tracked via `OP_FRAME` operational
frames and fit jointly, same fastik `SolverConfig::default()` tuning.

### Results (this machine)

```
-- single-frame time (latency), no warm start --
solve_ik() (cold)   n=20000   mean=768.3us   throughput=1301.6 calls/s

-- single-thread sequence throughput (native-rate frames, warm-started) --
solve_ik() (warm)   n=300     mean=1030.8us  throughput=970.1 calls/s

-- multi-thread sequence throughput (8 contiguous chunks, 8 threads) --
n_frames=1600   elapsed=0.236s   throughput=6779.4 frames/s
```

(`single_thread_throughput_fps`/`multi_thread_throughput_fps` varied
~940-980fps / ~6550-6900fps across repeated runs; see
`../../plot/results/pinocchio-cpp.json` for the exact numbers from the last
run.)

- `single_frame_latency_us` = 768.3 (Python: 1179.3; RBDL: 308.7)
- `single_thread_throughput_fps` = 970.1 (Python: 625.8; RBDL: 2363.9)
- `multi_thread_throughput_fps` = 6779.4 (Python: 3277.3; RBDL: 15709.7)

**Going native helped a lot but didn't close the gap with RBDL.** Latency
dropped ~35% and both throughput numbers roughly doubled versus the Python
benchmark -- confirming Python/numpy overhead was real and significant. But
Pinocchio C++ is still ~2.3x slower than RBDL on latency and single-thread
throughput, and ~2.3x slower on multi-thread throughput, on this exact same
problem. That's the opposite of what Carpentier et al. would predict, so it
was worth digging further.

Instrumenting `solve_ik` with `std::chrono` around each stage (summed over
the ~105,000 Gauss-Newton iterations executed during the
`single_frame_latency` run) gives a per-iteration breakdown:

| stage | us/iteration | share |
|---|---|---|
| `computeJointJacobians` + `updateFramePlacements` | ~5.3 | 3.4% |
| 30x `getFrameJacobian` (one per tracked keypoint) | ~83.4 | 54.0% |
| Eigen `jtj`/`jtr` accumulation over 30 keypoints | ~38.6 | 25.0% |
| `colPivHouseholderQr().solve()` (48x48) | ~26.7 | 17.3% |
| `pinocchio::integrate` | ~0.4 | 0.3% |
| **total** | **~154.4** | |

So in the native C++ version, the outer Eigen bookkeeping (`jtj`/`jtr`
accumulation + the QR solve, ~65us combined) is *not* the dominant cost the
way the Python/numpy bookkeeping was. Instead, the 30 separate
`getFrameJacobian` calls -- one per tracked keypoint, each extracting a 6x48
Jacobian block from `data.J` -- alone cost more than the rest of the
iteration combined. This is a real property of Pinocchio's API surface for
a many-keypoints-per-solve problem like this one (30 separate frame-Jacobian
extractions per iteration), not a Python-vs-C++ artifact: RBDL's
`InverseKinematicsConstraintSet` sidesteps it entirely by computing every
point constraint's Jacobian contribution in one internal pass rather than
exposing a per-frame extraction call that the caller has to invoke
repeatedly. In other words, the RBDL-vs-Pinocchio gap on *this specific
benchmark* looks like a real difference in how well each library's public
API fits a "many keypoints, one solve" workload, not an artifact of the
original Python benchmark being unfairly Python-bound (though that was also
true, and worth the ~2x improvement seen here).

Two follow-up checks, to make sure the ~83.4us/iteration wasn't just an
easy-to-fix implementation slip on our part before accepting it as genuine:
- **Not a missed non-allocating overload.** `solve_ik` already calls the
  output-parameter form, `getFrameJacobian(model, data, fid, ref_frame, J)`
  writing into a preallocated `Scratch::J`, not the allocating
  convenience overload that returns a fresh `MatrixXd` -- so heap allocation
  inside the hot loop isn't the explanation.
- **Not the `LOCAL_WORLD_ALIGNED` reference-frame conversion.** A standalone
  microbenchmark (30 calls/iteration, 20,000 iterations, isolated from the
  rest of `solve_ik`) timed `getFrameJacobian` under `LOCAL_WORLD_ALIGNED`
  vs. plain `LOCAL`: 85.6us vs. 87.0us for the 30 calls combined --
  statistically the same, actually marginally *slower* for `LOCAL`. If the
  cost were dominated by the extra rotation `LOCAL_WORLD_ALIGNED` applies to
  reorient each Jacobian column, `LOCAL` should have been meaningfully
  cheaper. It wasn't, so the ~2.85us/call cost is inherent to the frame's
  support-set lookup and column extraction from `data.J`, not the reference
  frame choice.

Both checks point the same way: the cost is a real property of calling
`getFrameJacobian` once per tracked keypoint on this branching tree, not a
benchmark bug.

`multi_thread_throughput_fps` uses the same simple contiguous-chunking
scheme as `bench_rbdl.cpp` (8 `std::thread` workers, each with its own
`pinocchio::Data` and warm-started within its own chunk but cold at the
chunk's start, all sharing one read-only `pinocchio::Model`) -- comparable
in spirit to `rbdl.json`'s multi-thread metric, unlike `pinocchio.json`'s
multiprocessing-based one.
