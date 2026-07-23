# Builds the QuickIK C++ bindings' test suite. This is also the reference
# integration example for consuming QuickIK from an external CMake project --
# see the top-level README's "C++" section.
#
# The Rust side (the core crate + the cxx bridge glue) is built via `cargo`,
# invoked as a custom target; `cargo build -p quickik-cpp` also copies the
# generated header and the bridge's compiled glue code into `include/` and
# `lib/` (see `build.rs`), which is what this file points at.

cmake_minimum_required(VERSION 3.15)
project(quickik_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QUICKIK_CPP_RELEASE "Build the Rust side in release mode" ON)
if(QUICKIK_CPP_RELEASE)
  set(QUICKIK_CARGO_ARGS --release)
  set(QUICKIK_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../target/release)
else()
  set(QUICKIK_CARGO_ARGS "")
  set(QUICKIK_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../target/debug)
endif()

add_custom_target(
  quickik_cargo_build ALL
  COMMAND cargo build -p quickik-cpp ${QUICKIK_CARGO_ARGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  BYPRODUCTS
    ${QUICKIK_TARGET_DIR}/libquickik_cpp.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libquickik-cpp-bridge.a
    ${CMAKE_CURRENT_SOURCE_DIR}/include/quickik.h
  COMMENT "cargo build -p quickik-cpp (Rust core + cxx bridge glue)"
)

# Two static libs come out of the Rust side: the crate's own compiled code
# (libquickik_cpp.a) and the cxx-bridge-generated C++ glue (compiled by
# cxx_build/cc, libquickik-cpp-bridge.a). They reference each other's symbols
# (Rust calls into cxx's C++ runtime helpers; C++ calls the exported
# `extern "Rust"` functions), so both must be linked together.
add_library(quickik_cpp STATIC IMPORTED GLOBAL)
set_target_properties(quickik_cpp PROPERTIES IMPORTED_LOCATION ${QUICKIK_TARGET_DIR}/libquickik_cpp.a)
add_dependencies(quickik_cpp quickik_cargo_build)

add_library(quickik_cpp_bridge STATIC IMPORTED GLOBAL)
set_target_properties(quickik_cpp_bridge PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libquickik-cpp-bridge.a)
add_dependencies(quickik_cpp_bridge quickik_cargo_build)

add_executable(quickik_cpp_tests tests/test_main.cpp)
target_include_directories(quickik_cpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_dependencies(quickik_cpp_tests quickik_cargo_build)
target_link_libraries(
  quickik_cpp_tests PRIVATE
  -Wl,--start-group quickik_cpp_bridge quickik_cpp -Wl,--end-group
  pthread dl m
)

# Correctness cross-check and throughput/latency benchmark against
# flygym.ik -- see ../benchmark/README.md.
add_executable(quickik_cpp_benchmark ../benchmark/quickik_cpp/bench_cpp.cpp)
target_include_directories(quickik_cpp_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(quickik_cpp_benchmark PRIVATE QUICKIK_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../benchmark/assets")
target_compile_options(quickik_cpp_benchmark PRIVATE -O3)
add_dependencies(quickik_cpp_benchmark quickik_cargo_build)
target_link_libraries(
  quickik_cpp_benchmark PRIVATE
  -Wl,--start-group quickik_cpp_bridge quickik_cpp -Wl,--end-group
  pthread dl m
)
