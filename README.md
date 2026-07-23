# QuickIK

Fast inverse kinematics library aimed for both high throughput and low latency.

QuickIK provides high-level APIs for processing consecutive frames (i.e. with warm starts) and multi-threaded batch processing, as well as low-level APIs for more specific use cases (e.g. real-time application). QuickIK is written in Rust but comes with Python and C++ bindings.

## Documentation

See the [docs site](https://nely-epfl.github.io/quickik/) for installation, usage examples, and benchmarks against KDL, Pinocchio, and RBDL.

Not yet published to crates.io or PyPI – see [`docs/installation.md`](docs/installation.md) for building from a local clone.

## For developers

A series of Python tools are used for development (Python bindings build tools, linting, benchmark plotting, unit testing, etc.). These are provided in a uv-managed environment under [`devtools-pyenv/`](devtools-pyenv/). This environment is unrelated to the QuickIK Python bindings, which lives under [`python/`](python/). The rest of the section assumes you have activated this environment:

```sh
cd devtools-pyenv && uv sync && source .venv/bin/activate
```

### Build

- **Rust core**: `cargo build` (add `--workspace` to also build the `python`/`cpp` bindings crates and the `benchmark/` crates).
- **Python bindings**: `cd python && maturin develop --release` – see [`docs/installation.md`](docs/installation.md) for prerequisites.
- **C++ bindings**: CMake, from `cpp/` – see [`docs/installation.md`](docs/installation.md).

### Lint/format

- **Rust**: `cargo fmt --check` (format) and `cargo clippy --workspace --all-targets` (lint).
- **Python**: with `devtools-pyenv` active, from that directory: `ruff check ..` (lint) and `ruff format ..` (format). Config is [`ruff.toml`](ruff.toml) at the repo root, auto-discovered – covers every `.py` file (`python/tests/`, `benchmark/`, `devtools-pyenv/` itself), but not the Python code blocks embedded in the docs site's Markdown pages.

### Tests

- **Rust**: `cargo test --workspace` (this also builds, but doesn't test, the `python`/`cpp` crates, since neither has its own `#[test]`s).
- **C++**: `./cpp/build/quickik_cpp_tests`, built as part of the C++ bindings step above.
- **Python**: with `devtools-pyenv` active and the bindings built (above), `pytest python/tests/`.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the exact commands CI runs.

### Benchmarks

See [`benchmark/README.md`](benchmark/README.md) for running QuickIK's own (Rust/Python/C++) and the external libraries' (KDL, Pinocchio, RBDL) benchmarks, and aggregating results into the comparison charts shown on the docs site.

### Docs site

```sh
docs/build.sh serve   # http://localhost:8000
```

Rebuilds the Rust/Python/C++ API references and benchmark charts from your local checkout, then serves the site locally.
