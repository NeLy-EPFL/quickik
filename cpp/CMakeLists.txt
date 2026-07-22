# Builds the FastIK C++ bindings' test suite. This is also the reference
# integration example for consuming FastIK from an external CMake project --
# see the top-level README's "C++" section.
#
# The Rust side (the core crate + the cxx bridge glue) is built via `cargo`,
# invoked as a custom target; `cargo build -p fastik-cpp` also copies the
# generated header and the bridge's compiled glue code into `include/` and
# `lib/` (see `build.rs`), which is what this file points at.

cmake_minimum_required(VERSION 3.15)
project(fastik_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FASTIK_CPP_RELEASE "Build the Rust side in release mode" ON)
if(FASTIK_CPP_RELEASE)
  set(FASTIK_CARGO_ARGS --release)
  set(FASTIK_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../target/release)
else()
  set(FASTIK_CARGO_ARGS "")
  set(FASTIK_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../target/debug)
endif()

add_custom_target(
  fastik_cargo_build ALL
  COMMAND cargo build -p fastik-cpp ${FASTIK_CARGO_ARGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  BYPRODUCTS
    ${FASTIK_TARGET_DIR}/libfastik_cpp.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libfastik-cpp-bridge.a
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fastik.h
  COMMENT "cargo build -p fastik-cpp (Rust core + cxx bridge glue)"
)

# Two static libs come out of the Rust side: the crate's own compiled code
# (libfastik_cpp.a) and the cxx-bridge-generated C++ glue (compiled by
# cxx_build/cc, libfastik-cpp-bridge.a). They reference each other's symbols
# (Rust calls into cxx's C++ runtime helpers; C++ calls the exported
# `extern "Rust"` functions), so both must be linked together.
add_library(fastik_cpp STATIC IMPORTED GLOBAL)
set_target_properties(fastik_cpp PROPERTIES IMPORTED_LOCATION ${FASTIK_TARGET_DIR}/libfastik_cpp.a)
add_dependencies(fastik_cpp fastik_cargo_build)

add_library(fastik_cpp_bridge STATIC IMPORTED GLOBAL)
set_target_properties(fastik_cpp_bridge PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libfastik-cpp-bridge.a)
add_dependencies(fastik_cpp_bridge fastik_cargo_build)

add_executable(fastik_cpp_tests tests/test_main.cpp)
target_include_directories(fastik_cpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_dependencies(fastik_cpp_tests fastik_cargo_build)
target_link_libraries(
  fastik_cpp_tests PRIVATE
  -Wl,--start-group fastik_cpp_bridge fastik_cpp -Wl,--end-group
  pthread dl m
)

# Correctness cross-check and throughput/latency benchmark against
# flygym.ik -- see ../benchmark/README.md.
add_executable(fastik_cpp_benchmark ../benchmark/cpp/bench_cpp.cpp)
target_include_directories(fastik_cpp_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(fastik_cpp_benchmark PRIVATE FASTIK_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../benchmark/assets")
target_compile_options(fastik_cpp_benchmark PRIVATE -O3)
add_dependencies(fastik_cpp_benchmark fastik_cargo_build)
target_link_libraries(
  fastik_cpp_benchmark PRIVATE
  -Wl,--start-group fastik_cpp_bridge fastik_cpp -Wl,--end-group
  pthread dl m
)
