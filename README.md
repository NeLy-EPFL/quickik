# FastIK

Fast inverse kinematics library aimed for both high throughput and low latency.

FastIK provides high-level APIs for processing consecutive frames (i.e. with warm starts) and multi-threaded batch processing, as well as low-level APIs for more specific use cases (e.g. real-time application). FastIK is written in Rust but comes with Python and C++ bindings.

## Documentation

See [`docs/`](docs/) (built with [Zensical](https://zensical.org)) for installation, usage examples, and benchmarks against KDL, Pinocchio, and RBDL:

```sh
docs/build.sh serve   # http://localhost:8000
```

Not yet published to crates.io or PyPI -- see [`docs/installation.md`](docs/installation.md) for building from a local clone.
