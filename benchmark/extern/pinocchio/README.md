# Pinocchio benchmark

Benchmarks [Pinocchio](https://github.com/stack-of-tasks/pinocchio) against fastik, on both bodies (see `../../README.md`). Methodology mirrors `../../fastik_python/bench.py` and `../../fastik_rust/src/perf.rs` exactly (same fixtures, same metrics, same config values) so the numbers are directly comparable. See the [Benchmarks docs page](../../../docs/benchmarks.md) for Pinocchio's modeling compromises and results.

## Running

Pinocchio's pip wheels don't support Python 3.13+, so a dedicated 3.12 venv is used:

```
cd /path/to/fastik/benchmark/extern/pinocchio
.venv312/bin/python bench_pinocchio.py
```

Prints a correctness cross-check (synthetic exact-fit frames) followed by the 3 performance numbers, and writes one `../../plot/results/pinocchio-<body>.json` per body.

## Native C++ benchmark

`bench_pinocchio_cpp.cpp` is a native C++ port of `bench_pinocchio.py`: same model construction and Gauss-Newton/LM math, ported line for line to Pinocchio's C++ API, with the outer-loop linear algebra done in Eigen instead of numpy. It exists to measure Pinocchio's own C++ speed on this workload without Python/numpy overhead.

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

`json.hpp` is a verbatim copy of `../rbdl/json.hpp` (dependency-free JSON reader), kept local so this directory builds standalone. Unlike the RBDL/KDL benchmarks, the JSON body plan is parsed directly in double precision here (not via `../rbdl/forward_kinematics.hpp`'s float-based `BodyPlan`), to match Python's float64 arrays exactly.
