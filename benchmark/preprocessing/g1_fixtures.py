"""Generates `benchmark/assets/fixtures_g1.json`, in the exact same schema as
the fly's `benchmark/assets/fixtures.json` (see
`benchmark/scripts/generate_fixtures.py`'s docstring): `synthetic_frames`
(exact-fit correctness check), `real_frames` (a sparse, diverse spread of
poses -- no cross-solver reconstruction fields, unlike the fly's, since G1
has no independent reference IK solver to check against), and
`native_rate_frames` (a contiguous window for the warm-start throughput
benchmark).

Unlike the fly (whose fixtures come from real mocap fit by flygym's own IK),
G1's `real_frames`/`native_rate_frames` come from real motion capture that's
*already* retargeted to G1 by a third party: the LAFAN1_Retargeting_Dataset
(https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), which
solved the LAFAN1-to-G1 retargeting via proper numerical optimization
(Interaction Mesh + IK, with foot-slip correction) -- see
`assets/g1_raw/README.md` for how that CSV was fetched. Every one of G1's 29
DOFs has a real angle in this data, so unlike a from-scratch BVH retarget,
no "Missing" keypoints are needed here: `target_ego` is dense for every
non-root node, exactly like the fly's.

`target_ego` here means the same thing as `generate_fixtures.py`'s own
`ego()`: a keypoint's world position expressed relative to one *fixed*
reference (root position + rotation), captured once per fixture group and
reused for every frame in that group -- not per-frame recentering (see that
script's `thorax_world_pos`/`thorax_world_mat`, captured once outside its
per-frame loop). One real difference from the fly's fixtures: G1's actual
root pose genuinely varies frame to frame (a walking human's pelvis really
does translate and turn), whereas the fly benchmark's synthetic construction
happens to hold the thorax fixed throughout -- so this benchmark exercises
real floating-base tracking under warm-starting in a way the fly one
doesn't. The root keypoint itself is still always given `Missing` at solve
time (that's a harness-side convention, not a fixtures one -- see
`build_observations` in each of the 6 harnesses), matching the fly exactly.

Usage:

    uv run --with numpy --with scipy python g1_fixtures.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from g1_body_plan import RETARGETED_CSV_PATH, build_g1_body_plan, load_neutral_angles
from g1_kinematics import G1Kinematics

PREPROCESSING_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = PREPROCESSING_DIR.parent
OUT_PATH = BENCHMARK_DIR / "assets" / "fixtures_g1.json"

NATIVE_RATE_START = 800
NATIVE_RATE_LENGTH = 300  # consecutive frames, 10s at this dataset's 30 fps
REAL_FRAMES = list(range(100, 7700, 345))  # ~22 frames spread across the clip
N_SYNTHETIC_FRAMES = 8
SYNTHETIC_ANGLE_SPREAD = 0.3  # radians, uniform around each DOF's neutral_angle


def load_csv_rows(path):
    with open(path) as f:
        return np.array([[float(v) for v in row] for row in csv.reader(f)])


def row_to_root_and_angles(row):
    return np.array(row[0:3]), R.from_quat(row[3:7]), list(row[7:])


def ego(world_pos, ref_pos, ref_rot):
    """A keypoint's world position -> a fixed reference frame's own local
    coordinates. `ref_pos`/`ref_rot` are captured once per fixture group and
    reused for every frame in it (see module docstring)."""
    return ref_rot.inv().apply(world_pos - ref_pos)


def target_ego_for_frame(kin, keypoint_names, root_pos, root_rot, dof_angles, ref_pos, ref_rot):
    positions = kin.evaluate(root_pos, root_rot, dof_angles)
    return [ego(positions[name], ref_pos, ref_rot).tolist() for name in keypoint_names]


def generate_synthetic_frames(kin, keypoint_names, neutral_angles, rng):
    frames = []
    for i in range(N_SYNTHETIC_FRAMES):
        dof_angles = [a + rng.uniform(-SYNTHETIC_ANGLE_SPREAD, SYNTHETIC_ANGLE_SPREAD) for a in neutral_angles]
        root_pos, root_rot = np.zeros(3), R.identity()
        target_ego = target_ego_for_frame(kin, keypoint_names, root_pos, root_rot, dof_angles, root_pos, root_rot)
        frames.append(
            {
                "frame": i,
                "target_ego": target_ego,
                "ground_truth_dof_angles_per_leg": [
                    dof_angles[0:6],  # left leg: hip_pitch/roll/yaw, knee, ankle_pitch/roll
                    dof_angles[6:12],  # right leg
                    dof_angles[12:15],  # waist: yaw, roll, pitch
                    dof_angles[15:22],  # left arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
                    dof_angles[22:29],  # right arm
                ],
            }
        )
    return frames


def generate_real_and_native_frames(kin, keypoint_names, rows):
    def frames_for(indices):
        ref_row = rows[indices[0]]
        ref_pos, ref_rot, _ = row_to_root_and_angles(ref_row)
        out = []
        for frame in indices:
            root_pos, root_rot, dof_angles = row_to_root_and_angles(rows[frame])
            target_ego = target_ego_for_frame(kin, keypoint_names, root_pos, root_rot, dof_angles, ref_pos, ref_rot)
            out.append({"frame": frame, "target_ego": target_ego})
        return out

    real_frames = frames_for(REAL_FRAMES)
    native_rate_frames = frames_for(list(range(NATIVE_RATE_START, NATIVE_RATE_START + NATIVE_RATE_LENGTH)))
    return real_frames, native_rate_frames


def main():
    rng = np.random.default_rng(seed=0)
    neutral_angles = load_neutral_angles(RETARGETED_CSV_PATH)
    body_plan = build_g1_body_plan(neutral_angles)
    kin = G1Kinematics(body_plan)
    keypoint_names = [j["name"] for j in body_plan["joints"][1:]]  # everything but pelvis

    rows = load_csv_rows(RETARGETED_CSV_PATH)

    synthetic_frames = generate_synthetic_frames(kin, keypoint_names, neutral_angles, rng)
    real_frames, native_rate_frames = generate_real_and_native_frames(kin, keypoint_names, rows)

    fixtures = {
        "metadata": {
            "bodyplan": "g1_body_plan.json",
            "source_recording": "LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv",
            "note": (
                "target_ego lists one [x,y,z] per non-root body-plan joint, in the "
                "same order as g1_body_plan.json's joints[1:] (i.e. excluding the "
                "free-floating 'pelvis' root, which has no independent mocap "
                "keypoint -- feed it Missing, same convention as the fly's "
                "fixtures.json). Unlike the fly, every entry here is a real, "
                "densely-populated target (no Missing keypoints), since the "
                "source data has a real angle for all 29 of G1's DOFs."
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
        f"{len(native_rate_frames)} native-rate frames to {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
