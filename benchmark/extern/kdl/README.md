# KDL benchmark

Benchmarks Orocos KDL's tree-based inverse kinematics against fastik, on both bodies (see `../../README.md`). See the [Benchmarks docs page](../../../docs/benchmarks.md) for KDL's modeling compromises and results; `bench_kdl.cpp`'s header comment has the full write-up.

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
