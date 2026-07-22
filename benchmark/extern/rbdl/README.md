# RBDL benchmark

Benchmarks RBDL's core-library `InverseKinematicsConstraintSet` (not an addon -- see `rbdl-src/include/rbdl/Kinematics.h`) against fastik, on both bodies (see `../../README.md`). See `bench_rbdl.cpp`'s header comment for the full write-up; summarized below. `leg_poc.cpp` is the earlier one-leg proof of concept this was built up from.

## Build

RBDL and Eigen (header-only) are already built from source at `rbdl-src/build/librbdl.a` / `eigen-src/` (see that build's `CMakeCache.txt` for the exact CMake invocation: `-DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON -DEIGEN3_INCLUDE_DIR=<eigen-src>`, all addons off -- RBDL's own `find_package(Eigen3)` doesn't work in this environment). Only that static lib and the headers are needed to build the benchmark itself:

```sh
cd benchmark/extern/rbdl
g++ -O3 -std=c++17 -DFASTIK_ASSETS_DIR='"../../assets"' -pthread \
    -I rbdl-src/include -I rbdl-src/build/include -I eigen-src \
    -o bench_rbdl bench_rbdl.cpp rbdl-src/build/librbdl.a
./bench_rbdl
```

`json.hpp` and `forward_kinematics.hpp` are verbatim copies of `../../fastik_cpp/`'s (dependency-free JSON reader + FK replica), kept local so this directory builds standalone.

## Formulation

Unlike KDL (which needs an explicit per-endpoint task-space hack to do position-only multi-endpoint fitting, see `../kdl/bench_kdl.cpp`), RBDL's `InverseKinematicsConstraintSet` natively takes an arbitrary list of 3D point constraints solved jointly in one linear system (`AddPointConstraint` per keypoint), so -- like fastik -- this benchmark fits all 30 leg keypoints (every coxa/femur/tibia/claw, not just the 6 claws) simultaneously, in one `Model` covering the floating thorax root plus all 6 legs.

Each named JSON joint with N dofs is expanded into N chained 1-dof `JointTypeRevolute` bodies (first dof carries the joint's offset translation, later ones use a zero-offset frame), exactly as in `leg_poc.cpp`. A joint's own keypoint is its first chain body's local origin `(0,0,0)`, which is invariant to that joint's own rotation -- reproducing fastik's convention that a joint's own dofs re-orient only its children. Leaf (0-dof) joints, i.e. the claws, are not separate RBDL bodies at all, just a body-point offset off their parent's last chain body.

## A real RBDL limitation, not a modeling compromise

RBDL's native `JointTypeFloatingBase` crashes `InverseKinematicsConstraintSet` for the free-floating thorax root.

RBDL stores a spherical/floating-base joint's quaternion with its *w* component appended at the very end of the `Q` vector (`Model::multdof3_w_index`), so any model using it has `q_size != qdot_size` (49 vs. 48 here). But `InverseKinematicsConstraintSet`'s Newton step (`src/Kinematics.cc`, the `Wn`/`delta_theta` block) sizes its damping matrix and does `Qres = Qres + delta_theta` using `Qres.size()` (== `q_size`) while `delta_theta` (derived from the Jacobian) is sized `qdot_size` -- a dimension mismatch that segfaults on the very first `InverseKinematics()` call for any model with a quaternion-based joint. This looks like a genuine upstream bug in RBDL's IK code, not a mistake in how it's used here.

**Workaround** (functionally the same trade-off the KDL benchmark already makes, for the same underlying reason -- no native non-quaternion 6-DOF joint): the thorax root is `JointTypeTranslationXYZ` + `JointTypeEulerZYX` in series. Same reachable pose space as a true floating base (modulo gimbal lock), q_size == qdot_size everywhere, sidesteps the bug entirely. Not fastik's singularity-free quaternion root.

## Solver algorithm and tuning

`InverseKinematicsConstraintSet` is **not** the transpose/damped-least-squares method RBDL's simpler free `InverseKinematics()` function uses (whose own docstring warns accuracy may only reach ~1e-2). Reading `src/Kinematics.cc` shows it instead solves the *joint-space* damped Levenberg-Marquardt normal equations `(J^T J + Wn) dq = J^T e` -- much closer in spirit to fastik's own Gauss-Newton solve than to a transpose method.

Tuning: `lambda=1e-6`, `max_steps=10`, `step_tol=1e-3` -- matches fastik's own `SolverConfig::default()`, for a fair comparison. `constraint_tol` is also set to `step_tol` but is dead code in this solve path -- only `step_tol` gates both the residual and the step-size early-stop checks. RBDL's own defaults (`max_steps=300`, `step_tol=1e-10`) are far tighter than this problem needs: real, imperfectly-fittable mocap data never gets close enough to trigger them, so they only burn extra iterations without improving accuracy.

## Accuracy

- **Synthetic exact-fit frames** (`synthetic_frames`, cold solve from neutral): converges in 6-7 steps to residual **rms ~2e-8, max ~8e-8** model units -- far tighter than the free function's ~1e-2 warning, and tighter than fastik needs to be.
- **Real native-rate frames** (`native_rate_frames`, warm-started): converges to **mean rms 0.076, worst-frame rms 0.131, worst single-point error 0.46** model units -- a real, tuning-independent residual floor, not an artifact of a stalled or under-iterated solve. These particular real mocap keypoint trajectories don't exactly satisfy this rigid rotation-axis kinematic model -- expected for real data (fastik's own correctness check reports nonzero, if smaller, residual on its analogous `real_frames` fixture for the same reason), not an RBDL-specific weakness.

## Results

Current numbers: `../../plot/results/rbdl-<body>.json`, or the comparison chart/table in `../../README.md`. Not reproduced here to avoid a second, driftable copy of the same data. Two patterns worth knowing when reading those numbers:

- Single-frame latency is noticeably faster than the warm-started sequence's per-frame mean, despite solving the same problem shape -- the fixed synthetic latency target is a small, well-conditioned perturbation from neutral, while real native-rate frames sit at the residual floor described above and burn far more LM steps hunting for a stationary point they'll never fully reach.
- Multi-thread throughput scales sub-linearly with thread count (well short of 8x at 8 threads) -- likely Eigen's `colPivHouseholderQr` (run every LM step, on a dense 48x48 matrix) contending for memory bandwidth/cache across threads, not a serialization bug (each thread has its own `Model` instance).

## Multi-thread benchmark note

RBDL has no built-in parallel solve path. `multi_thread_throughput_fps` uses simple contiguous chunking: an 1,600-frame tiled `native_rate_frames` sequence split into 8 equal 200-frame chunks, each solved independently (warm-started within its chunk, cold/neutral at the chunk's start, its own `Model` instance) on its own `std::thread` -- not fastik's overlap-stitched segmented solve.

## Python bindings

RBDL ships a Cython wrapper (`rbdl-src/python/`: `rbdl.pxd`, `rbdl-wrapper.pyx`, `wrappergen.py`, `CMakeLists.txt`) gated behind `RBDL_BUILD_PYTHON_WRAPPER` (default `OFF`), not built by RBDL's default CMake configuration. It's real Cython, not SWIG, and it exposes the same `InverseKinematicsConstraintSet`/`InverseKinematicsCS` solver `bench_rbdl.cpp` benchmarks -- so `bench_rbdl.py` calls RBDL's actual C++ solver through Python, not a hand-written Python IK loop (contrast with `../pinocchio/bench_pinocchio.py`, which has to hand-write Gauss-Newton because Pinocchio has no built-in solver).

### Building the wrapper

The system Python (3.12.3, Ubuntu) has no `python3-dev` headers installed and no sudo is available, so a `uv`-managed, header-complete Python 3.12 build (python-build-standalone) is used instead of the system interpreter:

```sh
cd benchmark/extern/rbdl

# A Python whose headers/libpython actually exist locally (no python3-dev
# installed, no sudo to install it):
uv python install 3.12
PYROOT="$(dirname "$(dirname "$(uv python find 3.12)")")"

uv venv --python "$PYROOT/bin/python3.12" .venv312
uv pip install --python .venv312/bin/python Cython numpy

mkdir rbdl-src/build-python && cd rbdl-src/build-python
PATH="$PWD/../../.venv312/bin:$PATH" cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON \
    -DEIGEN3_INCLUDE_DIR="$(pwd)/../../eigen-src" \
    -DRBDL_BUILD_ADDON_BALANCE=OFF -DRBDL_BUILD_ADDON_BENCHMARK=OFF \
    -DRBDL_BUILD_ADDON_GEOMETRY=OFF -DRBDL_BUILD_ADDON_LUAMODEL=OFF \
    -DRBDL_BUILD_ADDON_MUSCLE=OFF -DRBDL_BUILD_ADDON_MUSCLE_FITTING=OFF \
    -DRBDL_BUILD_ADDON_URDFREADER=OFF -DRBDL_BUILD_CASADI=OFF \
    -DRBDL_BUILD_EXECUTABLES=OFF -DRBDL_BUILD_STATIC=OFF -DRBDL_BUILD_TESTS=OFF \
    -DRBDL_BUILD_PYTHON_WRAPPER=ON \
    -DPYTHON_EXECUTABLE="$PWD/../../.venv312/bin/python3" \
    -DPYTHON_LIBRARY="$PYROOT/lib/libpython3.12.so" \
    -DPYTHON_INCLUDE_DIR="$PYROOT/include/python3.12"
PATH="$PWD/../../.venv312/bin:$PATH" make -j"$(nproc)" rbdl-python
```

This builds `rbdl-src/build-python/python/rbdl.so`, the compiled module `bench_rbdl.py` imports (via `sys.path.insert`, no install step needed -- its rpath already resolves `librbdl.so` and `libpython3.12.so`, both built into non-standard prefixes). Three build-system issues had to be worked around, none of them modeling choices:

1. **Wrong Python found.** CMake's legacy `FindPythonLibs` ignores `PYTHON_EXECUTABLE` and picks whatever `python3` resolves to system-wide (3.13 on this machine) regardless of which Python the Cython module is actually being built for (3.12, via the venv) -- silently producing an extension linked against the wrong CPython ABI. Fixed by passing `-DPYTHON_LIBRARY`/`-DPYTHON_INCLUDE_DIR` explicitly.
2. **Missing headers.** The system Python 3.12 has no `python3-dev` package installed (`/usr/include/python3.12` exists but lacks `Python.h`) and there's no sudo to install one. Fixed by using `uv python install 3.12`, which downloads a self-contained python-build-standalone build with its own headers/`libpython3.12.so`.
3. **`rbdl-python` links against a nonexistent `rbdl` target.** `python/CMakeLists.txt` does `TARGET_LINK_LIBRARIES(rbdl-python rbdl)`, but the top-level `CMakeLists.txt` only defines a target named `rbdl` when `RBDL_BUILD_STATIC=OFF`; with `RBDL_BUILD_STATIC=ON` (this repo's default, used by `bench_rbdl.cpp`) the target is named `rbdl-static` instead, so CMake silently treated the unresolved `rbdl` as a plain `-lrbdl` linker flag with no matching library -- and even after finding it, static libraries aren't built with `-fPIC` by default, which a Cython *shared* module needs. Building this tree with `RBDL_BUILD_STATIC=OFF` (a real `rbdl` shared-library target, PIC by default) fixed both problems at once, at the cost of a second RBDL build tree (`build-python/`, separate from `build/`, whose static `librbdl.a` the C++ benchmark still links).

### Running the benchmark

```sh
cd benchmark/extern/rbdl
.venv312/bin/python bench_rbdl.py
```

`bench_rbdl.py`'s `build_model` is a line-for-line port of `bench_rbdl.cpp`'s `build_model`/`neutral_q` to RBDL's Python API (`rbdl.Model.AddBody`, `rbdl.Joint(axes=[...])` for the arbitrary-axis 1-dof revolute chain links, `rbdl.SpatialTransform`) -- same TranslationXYZ + EulerZYX floating-base workaround (RBDL's native `JointTypeFloatingBase` crashes `InverseKinematicsConstraintSet` regardless of language binding -- it's a C++-level bug), same per-dof revolute chain expansion, same `lambda=1e-6, max_steps=10, step_tol=1e-3` tuning. All 30 leg keypoints are fit jointly in one `InverseKinematicsCS()` call per frame, exactly like the C++ benchmark.

`InverseKinematicsConstraintSet.target_positions` is a **getter-only** property in the generated wrapper (`wrappergen.py`'s `AddProperty` template only emits `__get__`, never `__set__`) -- every access reconstructs a fresh Python list of `Vector3d` objects that each alias the correct C++ memory address, so `cs.target_positions[k] = some_vector3d` silently assigns into that throwaway list and is discarded; the underlying solver state never changes. The fix (see `bench_rbdl.py`'s `set_targets`) is to fetch the aliased `Vector3d` list once per frame and mutate each element in place via its own (real) `__setitem__`, e.g. `cs.target_positions[k][:] = target[k]`, rather than reassigning list entries.

### Results

Current numbers: `../../plot/results/rbdl-python-<body>.json`, alongside the C++ benchmark's own `rbdl-<body>.json` and the comparison chart/table in `../../README.md`. Not reproduced here to avoid a second, driftable copy of the same data. Correctness matches the C++ benchmark closely, as expected (same model, same solver, same tuning).

Python is essentially indistinguishable from C++ here on latency and single-thread throughput -- a much smaller Python-vs-C++ gap than Pinocchio's own (see `../pinocchio/README.md`), because this benchmark makes a *single* Cython-wrapped `InverseKinematicsCS()` call per frame that runs RBDL's whole LM solve natively in C++ (Python/Cython marshaling overhead is paid once per frame, not once per solver iteration), whereas Pinocchio's Python benchmark has to run an entire hand-written Gauss-Newton loop in Python/numpy, paying Python-level overhead every iteration.

Multi-thread throughput has a real (if modest) gap from the C++ version, and it's a methodology difference more than a language one: `multi_thread_throughput_fps` here uses `multiprocessing.Pool` (8 separate OS processes, each importing `rbdl.so` and building its own `Model`) rather than `bench_rbdl.cpp`'s `std::thread`-based in-process chunking, since Python's GIL blocks real multi-threaded parallelism for CPU-bound Cython/C++ calls -- process dispatch/IPC adds overhead `std::thread` doesn't pay.

