# Builds QuickIK's C++ bindings as the CMake target `quickik` (a static library
# plus headers), and, if this is the top-level project, their tests and the
# benchmark in ../benchmark/quickik_cpp/. To use
# QuickIK from another CMake project, add this directory with
# `add_subdirectory` and link `quickik`, as examples/cpp/CMakeLists.txt does.
#
# The library itself is built by cargo (`cargo build -p quickik-cpp`), which
# also generates the headers in include/quickik/ (see build.rs).

cmake_minimum_required(VERSION 3.21)
project(quickik_cpp LANGUAGES CXX)

option(QUICKIK_CPP_RELEASE "Build the Rust side in release mode" ON)
option(QUICKIK_CPP_BUILD_TESTS "Build the C++ bindings' tests and benchmark" ${PROJECT_IS_TOP_LEVEL})

if(QUICKIK_CPP_RELEASE)
  set(QUICKIK_CARGO_ARGS --release)
  set(QUICKIK_CARGO_PROFILE release)
else()
  set(QUICKIK_CARGO_ARGS "")
  set(QUICKIK_CARGO_PROFILE debug)
endif()
set(QUICKIK_REPO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(QUICKIK_LIBRARY ${QUICKIK_REPO_DIR}/target/${QUICKIK_CARGO_PROFILE}/libquickik_cpp.a)

# Always runs, leaving it to cargo to decide whether anything needs rebuilding.
add_custom_target(
  quickik_cargo_build
  COMMAND cargo build -p quickik-cpp ${QUICKIK_CARGO_ARGS}
  WORKING_DIRECTORY ${QUICKIK_REPO_DIR}
  BYPRODUCTS ${QUICKIK_LIBRARY}
  USES_TERMINAL
)

# The Rust standard library needs threads, dynamic loading and libm.
find_package(Threads REQUIRED)
add_library(quickik STATIC IMPORTED GLOBAL)
set_target_properties(
  quickik PROPERTIES
  IMPORTED_LOCATION ${QUICKIK_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
  INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m"
  INTERFACE_COMPILE_FEATURES cxx_std_20
)
add_dependencies(quickik quickik_cargo_build)

if(QUICKIK_CPP_BUILD_TESTS)
  add_executable(quickik_cpp_tests tests/test_main.cpp)
  target_link_libraries(quickik_cpp_tests PRIVATE quickik)

  set(QUICKIK_BENCHMARK_DIR ${QUICKIK_REPO_DIR}/benchmark)
  add_executable(quickik_cpp_benchmark ${QUICKIK_BENCHMARK_DIR}/quickik_cpp/bench_cpp.cpp)
  target_link_libraries(quickik_cpp_benchmark PRIVATE quickik)
  target_compile_definitions(quickik_cpp_benchmark PRIVATE QUICKIK_ASSETS_DIR="${QUICKIK_BENCHMARK_DIR}/assets")
  target_compile_options(quickik_cpp_benchmark PRIVATE -O3)
endif()
