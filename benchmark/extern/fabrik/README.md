# FABRIK (reference implementation)

A from-scratch, dependency-free implementation of classic FABRIK (Forward
And Backward Reaching Inverse Kinematics; Aristidou & Lazarus, 2011), used
as a baseline in this benchmark. No dominant standard FABRIK C++ library
exists (unlike KDL/Pinocchio/RBDL), so this is written from the paper's
algorithm rather than wrapping an existing library. It is a reference/
baseline, not a polished library.

## Formulation

Matches the TRAC-IK benchmark's approach, for a consistent baseline:
classic FABRIK (like TRAC-IK) can only solve a single open chain with a
fixed base -- no floating base, no branching tree. So:

- **Thorax is a fixed base**: always at the origin with identity
  orientation, no floating-base solving at all.
- **Six independent chain solves per frame**: each leg (`lf`, `lm`, `lh`,
  `rf`, `rm`, `rh`) is its own FABRIK problem, fitting only that leg's claw
  (tip) 3D position target. Intermediate joint (coxa/femur/tibia) keypoints
  are not fit -- classic FABRIK only targets a chain's end effector, so this
  is inherent to the algorithm, not a shortcut.

Each leg's chain has 6 points (thorax + 5 body-plan joints:
`thorax_coxa`, `coxa_trochanterfemur`, `trochanterfemur_tibia`,
`tibia_tarsus`, `claw`) connected by 5 segments. Segment lengths are the
norm of each joint's `offset_pos` from
`assets/neuromechfly_ypr_legs.json`. The initial ("neutral") configuration
is the cumulative sum of `offset_pos` assuming zero rotation -- a valid
FABRIK starting position (FABRIK has no angle concept, only point
positions), not a claim about the mechanism's real neutral pose.

## Algorithm choice: unconstrained positional FABRIK

Classic FABRIK was designed for simple ball-joint/point chains: each joint
is free to bend in any direction, with only the segment *lengths* between
points fixed. It does **not** natively support constrained rotation axes
(e.g. "this joint only rotates about local Z") or realistic joint limits
the way an articulated-mechanism solver (fastik, KDL, TRAC-IK, Pinocchio)
does.

Two options were considered:

- **(a) Positional FABRIK** (implemented here): the classic, simplest form
  -- every joint is a free ball joint, no rotation-axis constraints at all.
  Easiest to implement correctly and the most standard form of FABRIK, but
  it does not respect the body plan's per-joint rotation-axis constraints,
  so it can reach configurations physically impossible for the real leg
  mechanism (it has strictly more degrees of freedom than fastik's
  axis-constrained joints).
- (b) A constrained/hybrid variant that respects rotation axes --
  significantly more implementation work, and arguably no longer "FABRIK"
  in the classical sense. **Not attempted**, for scope reasons.

**Consequence**: this solver's keypoint-fit "accuracy" is not directly
comparable to axis-constrained libraries (it has an easier problem to
solve, geometrically). Only its raw solve *speed* is a fair comparison
point -- see `notes` in `../../plot/results/fabrik.json`.

## Files

- `fabrik.hpp`: the algorithm (`fabrik::FabrikChain::solve`), templated on
  nothing, no external dependencies.
- `json.hpp`: verbatim copy of `../../fastik_cpp/json.hpp` (dependency-free
  JSON reader), duplicated here so this directory is self-contained.
- `bench_fabrik.cpp`: the benchmark driver, mirroring
  `../../fastik_cpp/bench_cpp.cpp`'s methodology and
  `../../plot/RESULTS_SCHEMA.md`'s output format.

## Build & run

No CMake, no external dependencies beyond the standard library and
`json.hpp`:

```sh
cd extern/fabrik
g++ -O3 -std=c++17 -DFASTIK_ASSETS_DIR='"../../assets"' -pthread -o bench_fabrik bench_fabrik.cpp
./bench_fabrik
```

Writes `../../plot/results/fabrik.json`.

## Results (this machine: Intel i9-11900K, 8 physical cores / 16 logical threads via SMT)

| metric | value |
|---|---|
| single-frame latency (cold, `synthetic_frames[0]` target) | 2.3 us |
| single-thread sequence throughput (300 native-rate frames, warm-started) | ~1,060,000 fps |
| multi-thread sequence throughput (8 threads, 2,400 tiled frames) | ~5,950,000 fps |

FABRIK converges in a handful of iterations (typically 2-4 of the 15-
iteration cap, tolerance `1e-4` model units) for these leg lengths and
targets -- verified by hand that the solved tip lands within tolerance and
every segment length is preserved exactly.

These numbers are 1-2 orders of magnitude faster than fastik/KDL/TRAC-IK,
which is expected and not a meaningful "FABRIK is better" result: each
per-leg chain here is a tiny 5-segment point chain with trivial vector
arithmetic (no quaternion composition, no Jacobians, no full 42-DOF
tree/regularization), and the problem itself is geometrically easier
(unconstrained ball joints always admit a solution once the target is
within reach, vs. constrained solvers that may need more iterations, or an
optimizer restart, to satisfy axis limits). Treat this column as a speed
floor for "chain-only, unconstrained, single-target" IK, not as a
like-for-like accuracy or performance comparison with the axis-constrained
libraries.

## Multi-thread benchmark note

Unlike fastik's `solve_sequence_segmented_parallel` (which needs
overlapping segments to blend continuity at chunk boundaries), FABRIK's
per-leg chains have no cross-frame smoothing to blend, so the multi-thread
benchmark here simply splits a 2,400-frame tiled sequence into 8 equal
300-frame contiguous chunks, each solved independently (warm-started
within its chunk, cold/neutral at the chunk's start) on its own
`std::thread`.
