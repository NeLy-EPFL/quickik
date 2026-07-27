#!/usr/bin/env bash
# Regenerates everything Zensical can't build itself -- the Rust/C++ API
# references (cargo doc / Doxygen) and the benchmark charts (copied from
# wherever benchmark/plot/*.py last wrote them) -- then builds or serves the
# site. The Python API reference (docs/api/python.md) is rendered natively by
# Zensical itself, via mkdocstrings; it just needs quickik built into the
# same venv Zensical runs under (devtools-pyenv), done below.
#
# Prerequisites on PATH: cargo, uv, doxygen (system package manager).
# zensical and maturin come from devtools-pyenv (`uv sync` there).
#
# Usage: docs/build.sh [build|serve]  (default: build)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- Rust API reference (cargo doc) ---
cargo doc --no-deps -p quickik
rm -rf docs/api/rust
mkdir -p docs/api/rust
cp -r target/doc/. docs/api/rust/

# --- Python API reference (mkdocstrings, via devtools-pyenv's quickik) ---
# Rebuilt through `uv sync --reinstall-package`, not a direct `maturin
# develop`: `uv run` (below) does its own implicit sync before running
# Zensical, and if that sync doesn't already agree the editable install is
# current, it silently reinstalls quickik from uv's cache -- discarding
# whatever `maturin develop` had just built from the current sources.
uv sync --reinstall-package quickik --project devtools-pyenv

# --- C++ API reference (Doxygen) ---
cargo build -p quickik-cpp
rm -rf docs/api/cpp-staging docs/api/cpp
mkdir -p docs/api/cpp-staging
cp docs/doxygen/rust_stub.h docs/api/cpp-staging/
python3 -c '
import re, pathlib
src = pathlib.Path("cpp/include/quickik.h").read_text()
# cxx emits Rust doc comments as plain "//" C++ comments; promote them to
# "///" so Doxygen (which only treats /** */, ///, and //! as doc comments)
# picks them up.
src = re.sub(r"^(\s*)// ", r"\1/// ", src, flags=re.MULTILINE)
# cxx names each include guard "CXXBRIDGE1_..._quickik$Name" -- the "$" desyncs
# Doxygen'"'"'s preprocessor across successive guards, silently dropping every
# quickik:: type after the first. The guards are meaningless for docs anyway.
src = re.sub(r"^\s*#(ifndef|define|endif)\b.*CXXBRIDGE1_.*$\n?", "", src, flags=re.MULTILINE)
# Keep only the quickik-facing declarations; rust_stub.h stands in for the
# runtime-support preamble (rust::Box/Slice/Str/Vec/Opaque/Error) that comes
# before it in the real generated header.
src = src[src.index("namespace quickik {"):]
pathlib.Path("docs/api/cpp-staging/quickik.h").write_text(src)
'
doxygen docs/doxygen/Doxyfile

# --- Benchmark charts ---
rm -rf docs/assets/benchmarks
mkdir -p docs/assets/benchmarks
if compgen -G "benchmark/plot/results/*.svg" > /dev/null; then
    cp benchmark/plot/results/*.svg docs/assets/benchmarks/
else
    echo "docs/build.sh: no charts in benchmark/plot/results/ yet -- see benchmark/README.md" >&2
fi

# `--clean` (build only; `serve` has no such flag) drops Zensical's own
# page-render cache (repo-root `.cache/`), which otherwise keys the Python
# API page purely off docs/api/python.md's own content -- oblivious to the
# fact that mkdocstrings' actual output depends on the freshly rebuilt
# quickik extension above, so a stale render survives every rebuild until
# something else happens to touch that .md file.
if [ "${1:-build}" = "build" ]; then
    uv run --project devtools-pyenv zensical build --clean
else
    uv run --project devtools-pyenv zensical "$1"
fi
