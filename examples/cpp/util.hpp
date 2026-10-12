// Helpers shared by the C++ examples: loading the recorded trajectory, a few
// vector operations, and summarizing and saving results in the format
// examples/plot_results.py reads. Nothing here is specific to QuickIK beyond
// reading its results.

#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "quickik.hpp"

// =============================================================================
//  Vector operations
// =============================================================================

inline quickik::Vec3 operator+(quickik::Vec3 a, quickik::Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline quickik::Vec3 operator-(quickik::Vec3 a, quickik::Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline quickik::Vec3 operator-(quickik::Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline quickik::Vec3 operator*(float s, quickik::Vec3 a) {
  return {s * a.x, s * a.y, s * a.z};
}
inline float dot(quickik::Vec3 a, quickik::Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float norm(quickik::Vec3 a) { return std::sqrt(dot(a, a)); }
inline quickik::Vec3 normalize(quickik::Vec3 a) { return (1 / norm(a)) * a; }
inline quickik::Vec3 cross(quickik::Vec3 a, quickik::Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline quickik::Vec3 operator*(const quickik::Mat3& m, quickik::Vec3 v) {
  return {dot(m.row0, v), dot(m.row1, v), dot(m.row2, v)};
}

// =============================================================================
//  Inputs and outputs
// =============================================================================

// Directory of the examples' input data and outputs (set by CMakeLists.txt).
inline const std::filesystem::path kExamplesDir = EXAMPLES_DIR;

// Seconds elapsed since `start`.
inline double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// A recorded keypoint trajectory, as stored in `example_trajectory.json`.
struct Trajectory {
  std::string units;
  float fps;
  std::vector<std::string> joint_names;
  // `positions[t][i]` is the position of keypoint `joint_names[i]` in frame
  // `t`.
  std::vector<std::vector<quickik::Vec3>> positions;

  static Trajectory load(const std::filesystem::path& path) {
    nlohmann::json data = nlohmann::json::parse(std::ifstream(path));
    Trajectory trajectory{data["units"], data["fps"], data["joint_names"], {}};
    for (const auto& frame : data["positions"]) {
      auto& positions = trajectory.positions.emplace_back();
      for (const auto& p : frame) positions.push_back({p[0], p[1], p[2]});
    }
    return trajectory;
  }

  // The body plan's joint index for each recorded keypoint.
  std::vector<size_t> joint_indices(const quickik::KinematicTree& tree) const {
    std::vector<size_t> indices;
    for (const auto& name : joint_names)
      indices.push_back(tree.joint_index(name).value());
    return indices;
  }

  // Mean of all recorded positions.
  quickik::Vec3 mean_position() const {
    quickik::Vec3 sum{0, 0, 0};
    size_t n = 0;
    for (const auto& frame : positions) {
      for (const auto& p : frame) sum = sum + p;
      n += frame.size();
    }
    return (1.0f / n) * sum;
  }
};

// A camera's projection and what it observed.
struct CameraView {
  std::string name;
  std::array<float, 2> image_size;
  std::unique_ptr<quickik::Projection> projection;
  // `observed[t][i]` is the observed pixel position of the trajectory's i-th
  // keypoint in frame `t`.
  std::vector<std::vector<quickik::Vec2>> observed;
};

// Everything an example's IK run produces, for reporting and saving.
struct IkOutput {
  std::unique_ptr<quickik::KinematicTree> tree;
  Trajectory trajectory;
  // The body plan's joint index for each recorded keypoint.
  std::vector<size_t> joint_indices;
  // One result per frame.
  std::unique_ptr<quickik::SolverResultList> results;
  // Time spent in the solver, in seconds.
  double solve_time;
  // The cameras the observations came from (empty for 3D observations).
  std::vector<CameraView> views;

  // Fitted minus recorded position of every recorded keypoint in every frame.
  std::vector<quickik::Vec3> errors() const {
    std::vector<quickik::Vec3> errors;
    for (size_t t = 0; t < results->len(); t++) {
      auto fitted = (*results)[t]->keypoint_pos().value();
      for (size_t i = 0; i < joint_indices.size(); i++) {
        const float* p = &fitted[3 * joint_indices[i]];
        errors.push_back(quickik::Vec3{p[0], p[1], p[2]} -
                         trajectory.positions[t][i]);
      }
    }
    return errors;
  }

  // Mean distance between fitted and recorded keypoints, over all frames.
  float mean_error() const {
    float sum = 0;
    auto errors = this->errors();
    for (const auto& e : errors) sum += norm(e);
    return sum / errors.size();
  }
};

// Rounds every non-integer number in `value` to 3 decimal places.
inline void round_numbers(nlohmann::json& value) {
  if (value.is_number_float()) {
    value = std::round(value.get<double>() * 1e3) / 1e3;
  } else if (value.is_structured()) {
    for (auto& item : value) round_numbers(item);
  }
}

inline nlohmann::json view_json(const CameraView& view) {
  auto vec2 = [](quickik::Vec2 v) { return nlohmann::json{v.x, v.y}; };
  auto vec3 = [](quickik::Vec3 v) { return nlohmann::json{v.x, v.y, v.z}; };
  nlohmann::json projection;
  if (auto camera = view.projection->as_pinhole_camera()) {
    const auto& rot = camera->world2cam_rot_mat;
    projection = {
        {"type", "pinhole_camera"},
        {"fx", camera->fx},
        {"fy", camera->fy},
        {"cx", camera->cx},
        {"cy", camera->cy},
        {"world2cam_pos", vec3(camera->world2cam_pos)},
        {"world2cam_rot_mat", {vec3(rot.row0), vec3(rot.row1), vec3(rot.row2)}},
    };
  } else if (auto ortho = view.projection->as_orthographic_view_xy()) {
    projection = {{"type", "orthographic_view_xy"},
                  {"scale", ortho->scale},
                  {"offset", vec2(ortho->offset)}};
  }
  nlohmann::json observed = nlohmann::json::array();
  for (const auto& frame : view.observed) {
    auto& uvs = observed.emplace_back(nlohmann::json::array());
    for (const auto& uv : frame) uvs.push_back(vec2(uv));
  }
  return {{"name", view.name},
          {"image_size", view.image_size},
          {"projection", projection},
          {"observed", observed}};
}

// Writes the inputs and results to `examples/cpp/output/<example>_<body>.json`
// in the format `examples/plot_results.py` reads (shared by all languages),
// with every number rounded to 3 decimal places.
inline void save_output(const std::string& example, const std::string& body,
                        const IkOutput& output) {
  // Joint names and parent indices, from the body plan file (QuickIK's C++
  // bindings don't expose them).
  auto body_plan_path = kExamplesDir / "assets" / body / "body_plan.json";
  nlohmann::json body_plan =
      nlohmann::json::parse(std::ifstream(body_plan_path));
  nlohmann::json joint_names = nlohmann::json::array(),
                 parents = nlohmann::json::array();
  for (const auto& joint : body_plan["joints"]) {
    joint_names.push_back(joint["name"]);
    const auto& parent = joint["parent"];
    if (parent.is_null()) {
      parents.push_back(nullptr);
    } else {
      parents.push_back(*output.tree->joint_index(parent.get<std::string>()));
    }
  }
  nlohmann::json positions = nlohmann::json::array();
  for (const auto& frame : output.trajectory.positions) {
    auto& frame_json = positions.emplace_back(nlohmann::json::array());
    for (const auto& p : frame) frame_json.push_back({p.x, p.y, p.z});
  }
  nlohmann::json fit = {{"root_pos", nlohmann::json::array()},
                        {"root_quat_wxyz", nlohmann::json::array()},
                        {"dof_values", nlohmann::json::array()},
                        {"keypoint_positions", nlohmann::json::array()}};
  for (const auto& result : *output.results) {
    const auto& state = result.state();
    auto pos = state.root_pos();
    auto quat = state.root_quat();
    auto dof_values = state.dof_values();
    auto keypoint_pos = result.keypoint_pos().value();
    fit["root_pos"].push_back({pos.x, pos.y, pos.z});
    fit["root_quat_wxyz"].push_back({quat.w, quat.x, quat.y, quat.z});
    fit["dof_values"].push_back(
        std::vector<float>(dof_values.begin(), dof_values.end()));
    auto& keypoints =
        fit["keypoint_positions"].emplace_back(nlohmann::json::array());
    for (size_t j = 0; j < keypoint_pos.size(); j += 3) {
      keypoints.push_back(
          {keypoint_pos[j], keypoint_pos[j + 1], keypoint_pos[j + 2]});
    }
  }
  nlohmann::json views = nlohmann::json::array();
  for (const auto& view : output.views) views.push_back(view_json(view));

  nlohmann::json data = {
      {"example", example},
      {"body", body},
      {"language", "cpp"},
      {"units", output.trajectory.units},
      {"fps", output.trajectory.fps},
      {"joint_names", joint_names},
      {"parents", parents},
      {"observations",
       {{"joint_indices", output.joint_indices}, {"positions", positions}}},
      {"fit", fit},
      {"views", views},
      {"solve_time_s", output.solve_time},
  };
  round_numbers(data);
  auto path = kExamplesDir / "cpp/output" / (example + "_" + body + ".json");
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path) << data.dump();
  std::printf("Saved %s\n", path.lexically_normal().c_str());
}
