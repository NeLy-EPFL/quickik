# TRAC-IK benchmark

A ROS-free, vanilla-g++ build of TRAC-IK's core solving code (`include/`,
`src/`), plus a benchmark (`bench_trac_ik.cpp`) that produces
`../../plot/results/trac-ik.json` per `../../plot/RESULTS_SCHEMA.md`.

`test_lf_leg.cpp` is the original proof-of-concept (single leg, correctness
check only) that established the working API pattern; `bench_trac_ik.cpp`
generalizes it to all 6 legs and adds the 3 performance metrics.

## Modeling compromise

TRAC-IK's `CartToJnt()` solves exactly one `KDL::Chain` against exactly one
Cartesian end-effector target -- there is no floating base and no
intermediate-waypoint fitting (confirmed against the API in
`include/trac_ik/trac_ik.hpp`). This is fundamentally less expressive than
fastik's solve, which jointly fits a floating thorax root plus all 6 legs'
worth of keypoints (coxa/femur/tibia/tarsus/claw, 30 in total) in one shot.

To benchmark it anyway, honestly:

- **Thorax is a fixed base.** No floating-base DOFs are solved at all -- this
  is a real asymmetry vs. fastik, not something to work around, and is
  reported rather than hidden (see the `notes` field of `trac-ik.json`).
- **6 independent chains, 6 independent solves per frame.** Each leg
  (`lf`/`lm`/`lh`/`rf`/`rm`/`rh`) gets its own `KDL::Chain` (thorax -> claw,
  7 actuated DOFs: 3 for the thorax-coxa joint, 2 for coxa-trochanterfemur, 1
  each for trochanterfemur-tibia and tibia-tarsus) and its own `TRAC_IK`
  solver instance, reused across frames. "One frame" = 6 sequential
  `CartToJnt` calls, one per leg.
- **Claw (tip) position only.** Each call fits only that leg's claw 3D
  position (`KDL::Twist` bounds: zero positional tolerance, huge (1e6)
  rotational tolerance -- effectively unconstrained orientation, since the
  body plan has no orientation keypoints and TRAC-IK can't fit the
  intermediate coxa/femur/tibia positions at all).
- **Multi-thread throughput has no real counterpart in TRAC-IK.** There's no
  built-in parallel/segmented solve path, so `bench_trac_ik.cpp` implements
  its own simplified scheme: a longer tiled sequence (native-rate frames,
  repeated) split into 8 equal, **contiguous, non-overlapping** chunks of 200
  frames each, solved independently -- own chain/solver copies, cold/neutral
  start at the chunk's beginning, warm-started within it -- on its own
  `std::thread`. This is simpler than fastik's overlap-stitched
  `solve_sequence_segmented_parallel` (no continuity blending across chunk
  boundaries); noted in the results JSON.

Body plan joints are loaded generically from
`../../assets/neuromechfly_ypr_legs.json` (not hardcoded per leg): each
leg's joint chain is found by walking parent links from its
`*_thorax_coxa` root (parent `"thorax"`) down to its claw, and each joint
becomes one fixed-translation KDL segment (its `offset_pos`) followed by one
`RotAxis` segment per DOF, with `neutral_angle` encoded as the KDL joint
offset -- exactly `test_lf_leg.cpp`'s verified pattern, generalized.

## Build

Requires the same dependencies as `test_lf_leg.cpp`: KDL + Eigen3 at
`../kdl/install`, NLopt at `~/nlopt-install`. From this directory:

```bash
KDL=/home/sibwang/Projects/fastik/benchmark/extern/kdl/install
NLOPT=/home/sibwang/nlopt-install
INCS="-Iinclude -I. -I$KDL/include -I$KDL/include/eigen3 -I$NLOPT/include"

g++ -std=c++20 -O3 $INCS -c src/trac_ik.cpp   -o build/trac_ik.o
g++ -std=c++20 -O3 $INCS -c src/nlopt_ik.cpp  -o build/nlopt_ik.o
g++ -std=c++20 -O3 $INCS -c src/kdl_tl.cpp    -o build/kdl_tl.o
g++ -std=c++20 -O3 $INCS -c bench_trac_ik.cpp -o build/bench_trac_ik.o

g++ -O3 build/trac_ik.o build/nlopt_ik.o build/kdl_tl.o build/bench_trac_ik.o \
  -o build/bench_trac_ik \
  -L$KDL/lib -L$NLOPT/lib -Wl,-rpath,$KDL/lib -Wl,-rpath,$NLOPT/lib \
  -lorocos-kdl -lnlopt -lpthread

./build/bench_trac_ik
```

(Note: as of writing, shell-variable expansion of `-I.../-L...` flags was
unreliable in this sandbox's Bash tool -- if `g++` reports missing headers
despite the paths looking right, spell out each `-I`/`-L` flag literally
instead of via a shell variable.)

This writes `../../plot/results/trac-ik.json` and prints a summary to
stdout. Total runtime is a few seconds.

## Results (this machine)

```
Loaded 6 leg chains (fixed thorax base), total actuated DOFs = 42

-- single-frame latency (fresh/neutral start, fixed synthetic_frames[0] target) --
CartToJnt x6 (fresh, fixed target)         n=3000    mean=  836.757us  median=  825.276us  p95=  916.387us  p99=  991.482us  min=  746.563us  max= 2127.705us  throughput=    1195.1 frames/s

-- single-thread sequence throughput (native-rate frames, warm-started) --
solve_frame (warm-started, native-rate)    n=300     mean=  482.896us  median=  477.062us  p95=  545.240us  p99=  613.885us  min=  414.855us  max=  785.940us  throughput=    2070.8 frames/s

-- multi-thread sequence throughput (8 threads, 200 contiguous frames each) --
total_frames=1600  throughput=    5676.3 frames/s
```

Summary (see `../../plot/results/trac-ik.json`):

| metric | value |
|---|---|
| single_frame_latency_us | 836.8 |
| single_thread_throughput_fps | 2070.8 |
| multi_thread_throughput_fps (8 threads) | 5676.3 |

Caveats: every `CartToJnt` call internally races 2 of its own
`std::thread`s (KDL NR-JL vs. NLopt; see `src/trac_ik.cpp`), so even the
"single-thread" metrics involve real concurrency under the hood, and the
8-thread benchmark oversubscribes cores (8 outer threads x 2 inner threads x
6 legs in flight at points). Numbers above are single-run, not averaged
across repeated process invocations.
