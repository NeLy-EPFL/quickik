#!/usr/bin/env bash
# Runs every benchmark under benchmark/ (QuickIK's own Rust/Python/C++
# bindings, plus the external KDL/Pinocchio/RBDL comparisons and the
# quickik_scaling thread-count sweep), then regenerates the comparison and
# scaling charts. See ../README.md for what each of these compares.
#
# This script only *runs* things -- it doesn't build or install anything.
# Every one-time setup step (building the external libraries, creating the
# .venv312 venvs, installing quickik into devtools-pyenv, etc.) is a real,
# often slow, first-time cost documented in each subdirectory's own README;
# baking that into this script would make one missing prerequisite silently
# trigger a from-source build of RBDL or a fresh venv you didn't ask for.
# Instead this script checks that everything is already in place and fails
# loudly, with a pointer to the relevant README, if it isn't -- so a partial
# environment never produces a comparison chart that silently looks complete.
#
# Usage:
#   benchmark/scripts/run_all_benchmarks.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DEVTOOLS_VENV="$REPO_ROOT/devtools-pyenv/.venv"
KDL_DIR="$REPO_ROOT/benchmark/extern/kdl"
RBDL_DIR="$REPO_ROOT/benchmark/extern/rbdl"
PINOCCHIO_DIR="$REPO_ROOT/benchmark/extern/pinocchio"

missing=()

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || missing+=("$2")
}

check_path() {
    [ -e "$1" ] || missing+=("$2")
}

check_python_import() {
    [ -x "$1" ] || return 0 # already reported by the check_path for it
    "$1" -c "import $2" >/dev/null 2>&1 || missing+=("$3")
}

echo "== Checking prerequisites =="

check_cmd cargo \
    "cargo/rustc not on PATH -- install a Rust toolchain (https://rustup.rs)"

check_path "$REPO_ROOT/cpp/build/quickik_cpp_benchmark" \
    "cpp/build/quickik_cpp_benchmark missing -- build with:
      cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build -j"

check_path "$DEVTOOLS_VENV/bin/python" \
    "devtools-pyenv/.venv missing -- run: cd devtools-pyenv && uv sync"
check_python_import "$DEVTOOLS_VENV/bin/python" quickik \
    "quickik not installed in devtools-pyenv/.venv -- run:
      cd python && $DEVTOOLS_VENV/bin/maturin develop --release
      (see docs/installation.md)"

check_path "$KDL_DIR/bench_kdl" \
    "$KDL_DIR/bench_kdl missing -- see benchmark/extern/kdl/README.md's Build section"

check_path "$RBDL_DIR/bench_rbdl" \
    "$RBDL_DIR/bench_rbdl missing -- see benchmark/extern/rbdl/README.md's Build section"
check_path "$RBDL_DIR/.venv312/bin/python" \
    "$RBDL_DIR/.venv312 missing -- see benchmark/extern/rbdl/README.md's
      'Python bindings' section"
check_path "$RBDL_DIR/rbdl-src/build-python/python/rbdl.so" \
    "RBDL's Python module (rbdl.so) not built -- see benchmark/extern/rbdl/README.md's
      'Building the wrapper' section"

check_path "$PINOCCHIO_DIR/.venv312/bin/python" \
    "$PINOCCHIO_DIR/.venv312 missing -- see benchmark/extern/pinocchio/README.md"
check_path "$PINOCCHIO_DIR/bench_pinocchio_cpp" \
    "$PINOCCHIO_DIR/bench_pinocchio_cpp missing -- see benchmark/extern/pinocchio/README.md's
      'Native C++ benchmark' section"

if [ "${#missing[@]}" -gt 0 ]; then
    echo "Missing prerequisites -- not running anything:" >&2
    for m in "${missing[@]}"; do
        echo "  - $m" >&2
    done
    exit 1
fi

echo "All prerequisites present."

echo
echo "== Running benchmarks (this takes a while -- KDL/RBDL's cold-latency"
echo "   sweeps alone are ~20000 solves per body) =="

echo; echo "-- QuickIK (Rust) --"
cargo run --release -p quickik-benchmark

echo; echo "-- QuickIK (C++) --"
"$REPO_ROOT/cpp/build/quickik_cpp_benchmark"

echo; echo "-- QuickIK (Python) --"
"$DEVTOOLS_VENV/bin/python" "$REPO_ROOT/benchmark/quickik_python/bench.py"

echo; echo "-- KDL --"
(cd "$KDL_DIR" && ./bench_kdl)

echo; echo "-- RBDL (C++) --"
(cd "$RBDL_DIR" && ./bench_rbdl)

echo; echo "-- RBDL (Python) --"
(cd "$RBDL_DIR" && .venv312/bin/python bench_rbdl.py)

echo; echo "-- Pinocchio (Python) --"
(cd "$PINOCCHIO_DIR" && .venv312/bin/python bench_pinocchio.py)

echo; echo "-- Pinocchio (C++) --"
CMEEL="$PINOCCHIO_DIR/.venv312/lib/python3.12/site-packages/cmeel.prefix"
(cd "$PINOCCHIO_DIR" && LD_LIBRARY_PATH="$CMEEL/lib" ./bench_pinocchio_cpp)

echo; echo "-- QuickIK scaling sweep (Rust, thread count) --"
"$REPO_ROOT/benchmark/quickik_scaling/run_sweep.sh"

echo; echo "== Aggregating results into charts =="
"$DEVTOOLS_VENV/bin/python" "$REPO_ROOT/benchmark/plot/plot_comparison.py"
"$DEVTOOLS_VENV/bin/python" "$REPO_ROOT/benchmark/plot/plot_scaling.py"
# 2D (XYView) vs. 3D, QuickIK's Rust benchmark only -- see plot_2d_comparison.py's
# own module docstring for why (2D observations are a QuickIK-only feature, and
# Python/C++ only carry a perf sanity test for it, not a full benchmark). Its
# input JSONs (quickik-rust-2d-xyview-neuromechfly.json, errors-neuromechfly.json)
# are already written above by the QuickIK (Rust) run.
"$DEVTOOLS_VENV/bin/python" "$REPO_ROOT/benchmark/plot/plot_2d_comparison.py"

echo; echo "Done. Results under benchmark/plot/results/, charts regenerated."
