# Installation

## From local clone

```sh
git clone https://github.com/sibocw/fastik
```

=== "Rust"

    Reference the clone by path in your `Cargo.toml`:

    ```toml
    [dependencies]
    fastik = { path = "../fastik" }
    ```

=== "Python"

    The Python bindings (`python/`) are a [PyO3](https://pyo3.rs)/[maturin](https://github.com/PyO3/maturin) extension module, so they build from source and require a Rust toolchain plus Python >= 3.8.

    ```sh
    cd fastik/python
    pip install maturin
    maturin develop --release
    ```

    `maturin develop` builds the Rust extension and installs it into your active environment in editable mode, so it rebuilds in place as you change the Rust source.

=== "C++"

    The C++ bindings (`cpp/`) are a [`cxx`](https://cxx.rs) bridge over the Rust core, so they also build from source and require a Rust toolchain, plus CMake and a C++17 compiler.

    ```sh
    cmake -S fastik/cpp -B fastik/cpp/build -DCMAKE_BUILD_TYPE=Release
    cmake --build fastik/cpp/build -j
    ./fastik/cpp/build/fastik_cpp_tests   # runs the binding's own test suite
    ```

    `cargo build -p fastik-cpp` (driven by `cpp/CMakeLists.txt` as a custom target) compiles the Rust side and copies the generated header and bridge glue into `cpp/include/` and `cpp/lib/`. A consuming project needs `cpp/include/` on its include path, and both `target/release/libfastik_cpp.a` (the crate itself) and `cpp/lib/libfastik-cpp-bridge.a` (the cxx-generated glue) linked in – see `cpp/CMakeLists.txt`'s `fastik_cpp`/`fastik_cpp_bridge` imported targets for the exact setup.
