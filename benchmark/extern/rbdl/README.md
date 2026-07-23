# RBDL benchmark

Benchmarks RBDL's core-library `InverseKinematicsConstraintSet` (not an addon -- see `rbdl-src/include/rbdl/Kinematics.h`) against QuickIK, on both bodies (see `../../README.md`). See the [Benchmarks docs page](../../../docs/benchmarks.md) for RBDL's modeling compromises and results; `bench_rbdl.cpp`'s header comment has the full write-up. `leg_poc.cpp` is the earlier one-leg proof of concept this was built up from.

## Build

RBDL and Eigen (header-only) are already built from source at `rbdl-src/build/librbdl.a` / `eigen-src/` (see that build's `CMakeCache.txt` for the exact CMake invocation: `-DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON -DEIGEN3_INCLUDE_DIR=<eigen-src>`, all addons off -- RBDL's own `find_package(Eigen3)` doesn't work in this environment). Only that static lib and the headers are needed to build the benchmark itself:

```sh
cd benchmark/extern/rbdl
g++ -O3 -std=c++17 -DQUICKIK_ASSETS_DIR='"../../assets"' -pthread \
    -I rbdl-src/include -I rbdl-src/build/include -I eigen-src \
    -o bench_rbdl bench_rbdl.cpp rbdl-src/build/librbdl.a
./bench_rbdl
```

`json.hpp` and `forward_kinematics.hpp` are verbatim copies of `../../quickik_cpp/`'s (dependency-free JSON reader + FK replica), kept local so this directory builds standalone.

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

`bench_rbdl.py`'s `build_model` is a line-for-line port of `bench_rbdl.cpp`'s `build_model`/`neutral_q` to RBDL's Python API (`rbdl.Model.AddBody`, `rbdl.Joint(axes=[...])` for the arbitrary-axis 1-dof revolute chain links, `rbdl.SpatialTransform`) -- same TranslationXYZ + EulerZYX floating-base workaround, same per-dof revolute chain expansion, same `lambda=1e-6, max_steps=10, step_tol=1e-3` tuning as the C++ benchmark.

`InverseKinematicsConstraintSet.target_positions` is a **getter-only** property in the generated wrapper (`wrappergen.py`'s `AddProperty` template only emits `__get__`, never `__set__`) -- every access reconstructs a fresh Python list of `Vector3d` objects that each alias the correct C++ memory address, so `cs.target_positions[k] = some_vector3d` silently assigns into that throwaway list and is discarded; the underlying solver state never changes. The fix (see `bench_rbdl.py`'s `set_targets`) is to fetch the aliased `Vector3d` list once per frame and mutate each element in place via its own (real) `__setitem__`, e.g. `cs.target_positions[k][:] = target[k]`, rather than reassigning list entries.
