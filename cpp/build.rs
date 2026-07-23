//! Compiles the `#[cxx::bridge]` module in `src/lib.rs`, then copies the
//! generated header and the bridge's compiled glue code out of Cargo's
//! internal (hashed, per-build) `OUT_DIR` into stable, documented paths --
//! `include/` and `lib/` -- so external, non-Cargo C++ builds (CMake, plain
//! g++) have a fixed location to point at. See `README.md` for the full
//! build instructions.

use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    cxx_build::bridge("src/lib.rs")
        .flag_if_supported("-std=c++17")
        .compile("quickik-cpp-bridge");

    let include_dir = manifest_dir.join("include");
    fs::create_dir_all(include_dir.join("rust")).unwrap();
    fs::copy(
        out_dir.join("cxxbridge/include/quickik-cpp/src/lib.rs.h"),
        include_dir.join("quickik.h"),
    )
    .unwrap();
    fs::copy(
        out_dir.join("cxxbridge/include/rust/cxx.h"),
        include_dir.join("rust/cxx.h"),
    )
    .unwrap();

    let lib_dir = manifest_dir.join("lib");
    fs::create_dir_all(&lib_dir).unwrap();
    fs::copy(
        out_dir.join("libquickik-cpp-bridge.a"),
        lib_dir.join("libquickik-cpp-bridge.a"),
    )
    .unwrap();

    println!("cargo:rerun-if-changed=src/lib.rs");
}
