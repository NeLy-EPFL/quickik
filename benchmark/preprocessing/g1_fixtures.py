"""Generates `benchmark/assets/fixtures_g1.json`, in the exact same schema as
the fly's `benchmark/assets/fixtures.json` (see
`benchmark/scripts/generate_fixtures.py`'s docstring): `synthetic_frames`
(exact-fit correctness check), `real_frames` (a sparse, diverse spread of
poses -- no cross-solver reconstruction fields, unlike the fly's, since G1
has no independent reference IK solver to check against), and
`native_rate_frames` (a contiguous window for the warm-start throughput
benchmark).

`real_frames`/`native_rate_frames` come from the *raw* LAFAN1 motion capture
(see `../assets/g1_raw/README.md` for where the BVH file comes from) --
retargeting a human skeleton onto a robot with different proportions is
itself an inverse-kinematics problem, so this benchmark does that retarget
with the same IK solvers it's comparing, rather than starting from a
third-party dataset that already solved it. `BVH_TO_G1` maps each raw BVH
joint to the G1 keypoint(s) anatomically nearest it (multiple keypoints often
share one BVH joint, since G1 expands a single anatomical joint -- e.g. the
hip -- into 3 single-DOF nodes that a human skeleton doesn't distinguish);
`compute_scale` rescales the human's positions by its overall leg-length
ratio to G1's so targets are roughly G1-sized. This is a coarse, uniform
rescale, not a proper retarget (which would need per-limb scaling and joint
limit awareness) -- the resulting residual is expected to be larger than the
fly's own real fixtures, precisely because it's real: a human's proportions
don't match G1's, so no solver will ever fit these targets exactly, the same
way none of them fits the fly's real mocap exactly.

`target_ego` here means the same thing as `generate_fixtures.py`'s own
`ego()`: a keypoint's world position expressed relative to one *fixed*
reference (root position + rotation), captured once per fixture group and
reused for every frame in that group -- not per-frame recentering (see that
script's `thorax_world_pos`/`thorax_world_mat`, captured once outside its
per-frame loop). G1's root pose genuinely varies frame to frame (a walking
human's pelvis really does translate and turn), whereas the fly benchmark's
synthetic construction happens to hold the thorax fixed throughout -- so this
benchmark exercises real floating-base tracking under warm-starting in a way
the fly one doesn't. The root keypoint itself is still always given `Missing`
at solve time (that's a harness-side convention, not a fixtures one -- see
`build_observations` in each of the 6 harnesses), matching the fly exactly.

Usage (with devtools-pyenv/'s shared venv active):

    python g1_fixtures.py
"""

import json
from pathlib import Path

import numpy as np
from g1_body_plan import NEUTRAL_ANGLES, build_g1_body_plan
from g1_kinematics import G1Kinematics
from lafan1_bvh import Lafan1Skeleton, parse_bvh
from scipy.spatial.transform import Rotation as R

PREPROCESSING_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = PREPROCESSING_DIR.parent
BVH_PATH = BENCHMARK_DIR / "assets" / "g1_raw" / "walk1_subject1.bvh"
OUT_PATH = BENCHMARK_DIR / "assets" / "fixtures_g1.json"

NATIVE_RATE_START = 800
NATIVE_RATE_LENGTH = 300  # consecutive frames, 10s at this dataset's 30 fps
REAL_FRAMES = list(range(100, 7700, 345))  # ~22 frames spread across the clip
N_SYNTHETIC_FRAMES = 8
SYNTHETIC_ANGLE_SPREAD = 0.3  # radians, uniform around each DOF's neutral_angle

BVH_UNIT_TO_METERS = 0.01  # LAFAN1's BVH offsets/positions are centimeters

# BVH is Y-up (X/Z horizontal); G1's URDF, like every other body plan here, is
# Z-up. A +90-degree rotation about X maps Y-up to Z-up correctly (verified
# against the URDF's own sign convention: G1's feet sit at negative Z relative
# to the pelvis, i.e. "up" is +Z), without mirroring left/right.
BVH_TO_G1_AXES = R.from_euler("x", 90, degrees=True)

# Each raw BVH joint -> the G1 keypoint(s) anatomically nearest it. G1 expands
# every anatomical joint into one single-DOF node per axis (see
# g1_body_plan.py's docstring), so several keypoints often share one BVH
# landmark -- e.g. the hip's 3 rotational DOFs are all close to "LeftUpLeg".
BVH_TO_G1 = {
    "LeftUpLeg": ["left_hip_pitch", "left_hip_roll", "left_hip_yaw"],
    "LeftLeg": ["left_knee"],
    "LeftFoot": ["left_ankle_pitch", "left_ankle_roll"],
    "RightUpLeg": ["right_hip_pitch", "right_hip_roll", "right_hip_yaw"],
    "RightLeg": ["right_knee"],
    "RightFoot": ["right_ankle_pitch", "right_ankle_roll"],
    "Spine": ["waist_yaw"],
    "Spine1": ["waist_roll"],
    "Spine2": ["waist_pitch"],
    "Head": ["head"],
    "LeftShoulder": ["left_shoulder_pitch"],
    "LeftArm": ["left_shoulder_roll", "left_shoulder_yaw"],
    "LeftForeArm": ["left_elbow"],
    "LeftHand": ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw", "left_hand"],
    "RightShoulder": ["right_shoulder_pitch"],
    "RightArm": ["right_shoulder_roll", "right_shoulder_yaw"],
    "RightForeArm": ["right_elbow"],
    "RightHand": [
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
        "right_hand",
    ],
}
G1_TO_BVH = {
    g1_name: bvh_name
    for bvh_name, g1_names in BVH_TO_G1.items()
    for g1_name in g1_names
}


def cumulative_offset(body_plan, node_name):
    """Sums `offset_pos` from the root down to (and including) `node_name`,
    i.e. that node's position in the body plan's own rest pose."""
    by_name = {j["name"]: j for j in body_plan["joints"]}
    pos = np.zeros(3)
    name = node_name
    while name is not None:
        node = by_name[name]
        pos += np.array(node["offset_pos"])
        name = node["parent"]
    return pos


def compute_scale(body_plan, bvh_joints):
    """Ratio of G1's own hip-to-ankle leg length to the BVH subject's, so BVH
    positions can be rescaled to roughly G1's proportions (see module
    docstring: a coarse global scale, not a proper per-limb retarget)."""
    g1_hip = cumulative_offset(body_plan, "left_hip_pitch")
    g1_ankle = cumulative_offset(body_plan, "left_ankle_pitch")
    g1_leg_length = np.linalg.norm(g1_ankle - g1_hip)

    bvh_leg_length = np.linalg.norm(
        bvh_joints["LeftLeg"].offset + bvh_joints["LeftFoot"].offset
    )
    bvh_leg_length *= BVH_UNIT_TO_METERS

    return g1_leg_length / bvh_leg_length


def ego(world_pos, ref_pos, ref_rot):
    """A keypoint's world position -> a fixed reference frame's own local
    coordinates. `ref_pos`/`ref_rot` are captured once per fixture group and
    reused for every frame in it (see module docstring)."""
    return ref_rot.inv().apply(world_pos - ref_pos)


def bvh_frame_positions(skeleton, motion_row, scale):
    """{bvh_joint_name: scaled, axis-corrected world position (meters), for
    one BVH motion frame. `scale` is the limb-length ratio from
    `compute_scale`; `BVH_UNIT_TO_METERS` converts the BVH's own centimeters
    first."""
    evaluated = skeleton.evaluate(motion_row)
    multiplier = BVH_UNIT_TO_METERS * scale
    return {
        name: BVH_TO_G1_AXES.apply(pos) * multiplier
        for name, (pos, _rot) in evaluated.items()
    }


def bvh_frame_root_rotation(skeleton, motion_row):
    """Hips' world rotation for one BVH motion frame, expressed in G1's
    axis convention (see `BVH_TO_G1_AXES`)."""
    _pos, rot = skeleton.evaluate(motion_row)["Hips"]
    return BVH_TO_G1_AXES * rot * BVH_TO_G1_AXES.inv()


def target_ego_for_frame(positions, keypoint_names, ref_pos, ref_rot):
    return [
        ego(positions[G1_TO_BVH[name]], ref_pos, ref_rot).tolist()
        for name in keypoint_names
    ]


def generate_synthetic_frames(kin, keypoint_names, neutral_angles, rng):
    frames = []
    for i in range(N_SYNTHETIC_FRAMES):
        dof_angles = [
            a + rng.uniform(-SYNTHETIC_ANGLE_SPREAD, SYNTHETIC_ANGLE_SPREAD)
            for a in neutral_angles
        ]
        root_pos, root_rot = np.zeros(3), R.identity()
        positions = kin.evaluate(root_pos, root_rot, dof_angles)
        target_ego = [
            ego(positions[name], root_pos, root_rot).tolist() for name in keypoint_names
        ]
        frames.append(
            {
                "frame": i,
                "target_ego": target_ego,
                "ground_truth_dof_angles_per_leg": [
                    dof_angles[
                        0:6
                    ],  # left leg: hip_pitch/roll/yaw, knee, ankle_pitch/roll
                    dof_angles[6:12],  # right leg
                    dof_angles[12:15],  # waist: yaw, roll, pitch
                    dof_angles[
                        15:22
                    ],  # left arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
                    dof_angles[22:29],  # right arm
                ],
            }
        )
    return frames


def generate_real_and_native_frames(skeleton, keypoint_names, motion, scale):
    def frames_for(indices):
        ref_positions = bvh_frame_positions(skeleton, motion[indices[0]], scale)
        ref_pos, ref_rot = (
            ref_positions["Hips"],
            bvh_frame_root_rotation(skeleton, motion[indices[0]]),
        )
        out = []
        for frame in indices:
            positions = bvh_frame_positions(skeleton, motion[frame], scale)
            target_ego = target_ego_for_frame(
                positions, keypoint_names, ref_pos, ref_rot
            )
            out.append({"frame": frame, "target_ego": target_ego})
        return out

    real_frames = frames_for(REAL_FRAMES)
    native_rate_frames = frames_for(
        list(range(NATIVE_RATE_START, NATIVE_RATE_START + NATIVE_RATE_LENGTH))
    )
    return real_frames, native_rate_frames


def main():
    rng = np.random.default_rng(seed=0)
    body_plan = build_g1_body_plan(NEUTRAL_ANGLES)
    kin = G1Kinematics(body_plan)
    keypoint_names = [
        j["name"] for j in body_plan["joints"][1:]
    ]  # everything but pelvis

    bvh_joints, motion, _frame_time = parse_bvh(BVH_PATH)
    skeleton = Lafan1Skeleton(bvh_joints)
    scale = compute_scale(body_plan, bvh_joints)

    synthetic_frames = generate_synthetic_frames(
        kin, keypoint_names, NEUTRAL_ANGLES, rng
    )
    real_frames, native_rate_frames = generate_real_and_native_frames(
        skeleton, keypoint_names, motion, scale
    )

    fixtures = {
        "metadata": {
            "bodyplan": "g1_body_plan.json",
            "source_recording": "LAFAN1/walk1_subject1.bvh",
            "note": (
                "target_ego lists one [x,y,z] per non-root body-plan joint, in the "
                "same order as g1_body_plan.json's joints[1:] (i.e. excluding the "
                "free-floating 'pelvis' root, which has no independent mocap "
                "keypoint -- feed it Missing, same convention as the fly's "
                "fixtures.json). real_frames/native_rate_frames come from a raw "
                "human motion capture recording rescaled onto G1's proportions "
                "(see g1_fixtures.py's module docstring), not from a third-party "
                f"retargeting -- expect a real, nonzero fit residual. scale={scale:.4f}."
            ),
        },
        "leg_joint_names": keypoint_names,
        "synthetic_frames": synthetic_frames,
        "real_frames": real_frames,
        "native_rate_frames": native_rate_frames,
    }
    OUT_PATH.write_text(json.dumps(fixtures))
    print(
        f"Wrote {len(synthetic_frames)} synthetic + {len(real_frames)} real + "
        f"{len(native_rate_frames)} native-rate frames to {OUT_PATH} (scale={scale:.4f})"
    )


if __name__ == "__main__":
    main()
