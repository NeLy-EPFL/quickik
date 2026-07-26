"""Converts the Unitree G1 URDF into QuickIK's body-plan JSON schema (the same
schema as `benchmark/assets/neuromechfly_ypr_legs.json` -- see
`benchmark/scripts/generate_fixtures.py`'s docstring and `src/body_plan.rs`):

    {"fixed_base": bool, "x-name": ...,
     "joints": [{"name", "parent", "offset_pos": [x,y,z],
                 "offset_quat": [w,x,y,z], "dofs": [...]}]}

Unlike the fly's body plan (multi-axis joints collocated at a single point,
e.g. a 3-DOF "coxa"), every one of G1's 29 revolute joints has its own nonzero
offset from its parent link, so each becomes its own single-DOF node here --
matching the same convention the RBDL/Pinocchio/KDL benchmarks already use
when splitting a multi-DOF joint into a chain of single-DOF ones (see e.g.
../extern/rbdl/bench_rbdl.cpp's `build_model`).

Three zero-DOF leaf keypoints are added (mirroring the fly's leaf "claw"
nodes -- a keypoint with no DOF of its own, just a fixed offset from its
parent): `head`, `left_hand`, `right_hand`, taken from the URDF's own fixed
joints (`head_joint`, `left_hand_palm_joint`, `right_hand_palm_joint`) so the
whole-tree formulation has a target for the head and hands, not just the last
actuated joint in each chain.

All positions are the URDF's own real-world meters (no rescaling -- unlike
the fly's arbitrary "model units", there's no reason not to use G1's actual
dimensions). All `limits` are written as `null` (unbounded), matching every
other body plan in this repo: RBDL/KDL's own benchmarks specifically rely on
`limits: null` to skip joint-limit clamping (see ../extern/kdl/bench_kdl.cpp's
comment), and QuickIK's own `State` setter clamps to `limits` when present, so
introducing real bounds here would silently change solver behavior relative
to the other libraries' benchmarks.

Usage:

    python g1_body_plan.py
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

PREPROCESSING_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = PREPROCESSING_DIR.parent
URDF_PATH = BENCHMARK_DIR / "assets" / "g1_raw" / "g1_29dof.urdf"
OUT_PATH = BENCHMARK_DIR / "assets" / "g1_body_plan.json"

# (name, parent_link) for each revolute joint, in the URDF's own kinematic
# order (root to leaves, left leg/right leg/waist/left arm/right arm).
REVOLUTE_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Leaf keypoint name -> (fixed URDF joint providing its offset, DOF-bearing
# parent node it attaches to in this body plan -- body-plan node names, i.e.
# the parent *revolute* joint's name with "_joint" stripped, not the URDF's
# own fixed-joint parent link name).
LEAF_KEYPOINTS = {
    "head": ("head_joint", "waist_pitch"),
    "left_hand": ("left_hand_palm_joint", "left_wrist_yaw"),
    "right_hand": ("right_hand_palm_joint", "right_wrist_yaw"),
}

ROOT_LINK = "pelvis"

# Each DOF's `neutral` value -- the solver's regularization anchor
# (`SolverConfig::neutral_weight`) and `State::neutral_pose`'s starting guess. The
# URDF's own zero configuration (no retargeted motion data feeds this
# anymore -- see g1_fixtures.py's module docstring for why).
NEUTRAL_ANGLES = [0.0] * len(REVOLUTE_JOINT_ORDER)


def rpy_to_wxyz(rpy):
    """URDF <origin rpy="r p y"> (extrinsic X-Y-Z Euler, URDF's own
    convention) -> a [w, x, y, z] quaternion."""
    import math

    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y_, z]


def parse_joints(urdf_path):
    """Returns {joint_name: {"type", "parent", "child", "xyz", "rpy",
    "axis", "limits"}} for every <joint> in the URDF."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        origin = j.find("origin")
        xyz = (
            [float(v) for v in origin.get("xyz", "0 0 0").split()]
            if origin is not None
            else [0.0, 0.0, 0.0]
        )
        rpy = (
            [float(v) for v in origin.get("rpy", "0 0 0").split()]
            if origin is not None
            else [0.0, 0.0, 0.0]
        )
        axis_el = j.find("axis")
        axis = (
            [float(v) for v in axis_el.get("xyz").split()]
            if axis_el is not None
            else None
        )
        limit_el = j.find("limit")
        limits = (
            [float(limit_el.get("lower")), float(limit_el.get("upper"))]
            if limit_el is not None and limit_el.get("lower") is not None
            else None
        )
        joints[name] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "limits": limits,
        }
    return joints


def build_g1_body_plan(neutral_angles):
    joints = parse_joints(URDF_PATH)

    nodes = [
        {
            "name": ROOT_LINK,
            "parent": None,
            "offset_pos": [0.0, 0.0, 0.0],
            "offset_quat": [1.0, 0.0, 0.0, 0.0],
            "dofs": [],
        }
    ]

    # link name -> body-plan node name that link's frame is attached to
    # (needed since a revolute joint's *child link* is this node itself, but
    # a later joint parented on that same link should point at this node).
    link_to_node_name = {ROOT_LINK: ROOT_LINK}

    for joint_name, neutral_angle in zip(
        REVOLUTE_JOINT_ORDER, neutral_angles, strict=True
    ):
        j = joints[joint_name]
        parent_node = link_to_node_name[j["parent"]]
        node_name = joint_name.removesuffix("_joint")
        nodes.append(
            {
                "name": node_name,
                "parent": parent_node,
                "offset_pos": j["xyz"],
                "offset_quat": rpy_to_wxyz(j["rpy"]),
                "dofs": [
                    {
                        "name": joint_name,
                        "axis": j["axis"],
                        "type": "hinge",
                        "neutral": neutral_angle,
                        "limits": None,
                    }
                ],
            }
        )
        link_to_node_name[j["child"]] = node_name

    for leaf_name, (fixed_joint_name, parent_node) in LEAF_KEYPOINTS.items():
        j = joints[fixed_joint_name]
        nodes.append(
            {
                "name": leaf_name,
                "parent": parent_node,
                "offset_pos": j["xyz"],
                "offset_quat": rpy_to_wxyz(j["rpy"]),
                "dofs": [],
            }
        )

    return {"fixed_base": False, "x-name": "g1_29dof", "joints": nodes}


if __name__ == "__main__":
    body_plan = build_g1_body_plan(NEUTRAL_ANGLES)
    OUT_PATH.write_text(json.dumps(body_plan, indent=2) + "\n")
    n_dofs = sum(len(j["dofs"]) for j in body_plan["joints"])
    print(f"Wrote {len(body_plan['joints'])} joints ({n_dofs} DOFs) to {OUT_PATH}")
