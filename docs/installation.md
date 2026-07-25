# Installation

## From local clone

```sh
git clone https://github.com/NeLy-EPFL/quickik
```

=== "Rust"

    Reference the clone by path in your `Cargo.toml`:

    ```toml
    [dependencies]
    quickik = { path = "../quickik" }
    ```

=== "Python"

    The Python bindings (`python/`) are a [PyO3](https://pyo3.rs)/[maturin](https://github.com/PyO3/maturin) extension module, so they build from source and require a Rust toolchain plus Python >= 3.8.

    ```sh
    cd quickik/python
    pip install maturin  # or use uv etc. if you're in a managed virtual env
    maturin develop --release

    # To run unit tests for the Python bindings:
    # Run `pip install pytest` if you don't have it already
    pytest tests/
    ```

    `maturin develop` builds the Rust extension and installs it into your active environment in editable mode, so it rebuilds in place as you change the Rust source. For linting or benchmark plotting rather than the bindings themselves, see `devtools-pyenv/` instead.

=== "C++"

    The C++ bindings (`cpp/`) are a [`cxx`](https://cxx.rs) bridge over the Rust core, so they also build from source and require a Rust toolchain, plus CMake and a C++17 compiler.

    ```sh
    cmake -S quickik/cpp -B quickik/cpp/build -DCMAKE_BUILD_TYPE=Release
    cmake --build quickik/cpp/build -j

    # To run unit tests for the C++ bindings:
    ./quickik/cpp/build/quickik_cpp_tests
    ```

    The `cmake --build` command above already builds everything needed to run the test suite – there's nothing extra to do for that. If you're linking QuickIK into your own CMake project instead, add `cpp/include/` to your include path and link against the `quickik_cpp` and `quickik_cpp_bridge` imported targets that `cpp/CMakeLists.txt` defines; both are produced automatically by the same build (a custom target runs `cargo build -p quickik-cpp` and copies the generated header and glue code into `cpp/include/`/`cpp/lib/`).
