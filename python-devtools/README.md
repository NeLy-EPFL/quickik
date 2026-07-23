# python-devtools

A shared [uv](https://docs.astral.sh/uv/) environment for this repo's Python-side tooling: `ruff` (linting), `matplotlib`/`numpy`/`scipy` (benchmark plotting), and `pytest` (the Python bindings' own test suite).

This is a plain uv *application* project (`uv init --app --no-package`), not a library -- it exists only to pin these tools' versions in one lockfile (`uv.lock`) instead of re-declaring them ad hoc on every invocation. It's deliberately separate from `python/pyproject.toml` (the actual `quickik` package) so there's no ambiguity about which `pyproject.toml` governs what.

## Setup

```sh
cd python-devtools
uv sync
source .venv/bin/activate
```

## Using it

- **Lint**: `ruff check ..` (config: `../ruff.toml`, auto-discovered) lints every `.py` file in the repo.
- **Benchmark plotting**: with this venv active, `benchmark/plot/plot_comparison.py`, `plot_scaling.py`, and `render_video.py` all just run as `python <script>.py` -- see each one's own docstring.
- **Python bindings' tests**: `maturin develop --release` (from `python/`, with this venv active) builds `quickik` into this same environment, then `pytest ../python/tests/` runs its test suite.

Deliberately not included: `mujoco`/`flygym` (only `benchmark/scripts/generate_fixtures.py` needs them, and it already documents running with flygym's own venv) and `maturin` itself (a build tool for `quickik`, not a dependency of anything in this directory).
