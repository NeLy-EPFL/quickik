# Builds the C++ examples. Also an example of using QuickIK from another CMake
# project: add QuickIK's cpp/ directory and link the `quickik` target.
#
# From the repository root:
#   cmake -S examples/cpp -B examples/cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build examples/cpp/build -j

cmake_minimum_required(VERSION 3.21)
project(quickik_examples LANGUAGES CXX)

add_subdirectory(../../cpp quickik)

# For reading and writing the examples' JSON data
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

foreach(example from_3d ortho2d from_cameras)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE quickik nlohmann_json::nlohmann_json)
  # Lets the examples find their input data and output directory.
  target_compile_definitions(${example} PRIVATE EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
endforeach()
