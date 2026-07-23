// An independent forward-kinematics replica, computed directly from the JSON
// body plan (not calling into quickik at all), mirroring
// scripts/bench_python.py's own `forward_kinematics()` -- used to cross-check
// the C++ bindings' solved `State`, since FK isn't exposed to C++ (same as
// Python; see cpp/src/lib.rs's module docs).

#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "json.hpp"

struct Vec3 {
  float x = 0, y = 0, z = 0;
  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  float norm() const { return std::sqrt(x * x + y * y + z * z); }
};

// Hamilton quaternion, (w, x, y, z), matching quickik's own convention
// (nalgebra::UnitQuaternion, serialized in JSON body plans as [w, x, y, z]).
struct Quat {
  float w = 1, x = 0, y = 0, z = 0;

  Quat operator*(const Quat &o) const {
    return {
        w * o.w - x * o.x - y * o.y - z * o.z,
        w * o.x + x * o.w + y * o.z - z * o.y,
        w * o.y - x * o.z + y * o.w + z * o.x,
        w * o.z + x * o.y - y * o.x + z * o.w,
    };
  }

  Vec3 apply(const Vec3 &v) const {
    Quat p{0, v.x, v.y, v.z};
    Quat conj{w, -x, -y, -z};
    Quat r = (*this * p) * conj;
    return {r.x, r.y, r.z};
  }

  static Quat from_axis_angle(const Vec3 &axis, float angle) {
    float half = angle * 0.5f;
    float s = std::sin(half);
    return {std::cos(half), axis.x * s, axis.y * s, axis.z * s};
  }
};

struct Dof {
  Vec3 axis;
};

struct Joint {
  std::string name;
  int parent = -1;  // -1 for the root
  Vec3 offset_pos;
  Quat offset_quat;
  std::vector<Dof> dofs;
  size_t dof_offset = 0;
};

struct BodyPlan {
  std::vector<Joint> joints;  // joints[0] is the root
};

inline BodyPlan load_body_plan(const std::string &path) {
  Json root = parse_json_file(path);
  BodyPlan plan;
  size_t cursor = 0;
  for (auto &j : root["joints"].as_array()) {
    Joint joint;
    joint.name = j["name"].as_string();
    if (j["parent"].is_null()) {
      joint.parent = -1;
    } else {
      const std::string &parent_name = j["parent"].as_string();
      for (size_t i = 0; i < plan.joints.size(); i++) {
        if (plan.joints[i].name == parent_name) {
          joint.parent = static_cast<int>(i);
          break;
        }
      }
    }
    auto &op = j["offset_pos"].as_array();
    joint.offset_pos = {static_cast<float>(op[0].as_number()), static_cast<float>(op[1].as_number()),
                         static_cast<float>(op[2].as_number())};
    auto &oq = j["offset_quat"].as_array();
    joint.offset_quat = {static_cast<float>(oq[0].as_number()), static_cast<float>(oq[1].as_number()),
                          static_cast<float>(oq[2].as_number()), static_cast<float>(oq[3].as_number())};
    joint.dof_offset = cursor;
    for (auto &d : j["dofs"].as_array()) {
      auto &axis = d["axis"].as_array();
      joint.dofs.push_back(
          {Vec3{static_cast<float>(axis[0].as_number()), static_cast<float>(axis[1].as_number()),
                static_cast<float>(axis[2].as_number())}});
      cursor++;
    }
    plan.joints.push_back(std::move(joint));
  }
  return plan;
}

/// Returns world positions of every joint (root included, at index 0),
/// matching `dof_angles`' flattened layout and `root_pos`/`root_rot` from a
/// solved `State`.
inline std::vector<Vec3> forward_kinematics(const BodyPlan &plan, const std::vector<float> &dof_angles,
                                             Vec3 root_pos, Quat root_rot) {
  std::vector<Vec3> world_pos(plan.joints.size());
  std::vector<Quat> world_rot(plan.joints.size());
  for (size_t i = 0; i < plan.joints.size(); i++) {
    const Joint &j = plan.joints[i];
    Vec3 origin;
    Quat rot;
    if (j.parent < 0) {
      origin = root_pos;
      rot = root_rot;
    } else {
      origin = world_pos[j.parent] + world_rot[j.parent].apply(j.offset_pos);
      rot = world_rot[j.parent] * j.offset_quat;
    }
    for (size_t d = 0; d < j.dofs.size(); d++) {
      rot = rot * Quat::from_axis_angle(j.dofs[d].axis, dof_angles[j.dof_offset + d]);
    }
    world_pos[i] = origin;
    world_rot[i] = rot;
  }
  return world_pos;
}
