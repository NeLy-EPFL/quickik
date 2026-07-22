"""Correctness cross-check and throughput/latency benchmark for fastik's
Python bindings, mirroring `benchmark/src/{correctness,perf}.rs` (the Rust
API benchmark) so the two are directly comparable.

Uses the same `assets/fixtures.json`/`assets/neuromechfly_ypr_legs.json` as
the Rust benchmark (see `generate_fixtures.py`). Requires:

  - fastik's Python extension built for this interpreter (see
    `python/README` or the top-level README's "Python" install section --
    this script does not build it for you).
  - flygym's own venv (mujoco, scipy, flygym) for the real-frame
    cross-solver check.

Run with flygym's own venv:

    cd /path/to/flygym && source .venv/bin/activate
    python /path/to/fastik/benchmark/scripts/bench_python.py

One thing the Rust benchmark can do that this one can't, since it's not
exposed to Python: forward kinematics on its own (`fastik::forward` isn't
bound). So instead of an independent FK re-check via fastik's own
(unexposed) `evaluate_fwdkin`, this script reimplements forward kinematics
directly from the JSON body plan (a fresh, independent computation, not
calling into fastik at all) for the keypoint comparisons below. The
performance benchmark's single-frame-latency target is one of the fixture's
own targets (`synthetic_frames[0]`), matching the Rust and C++ benchmarks
exactly, for a fair cross-language comparison.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import fastik

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BENCHMARK_DIR / "assets"

fixtures = json.loads((ASSETS_DIR / "fixtures.json").read_text())
bodyplan = json.loads((ASSETS_DIR / "neuromechfly_ypr_legs.json").read_text())
JOINTS = bodyplan["joints"]
LEG_JOINT_NAMES = fixtures["leg_joint_names"]
assert [j["name"] for j in JOINTS][1:] == LEG_JOINT_NAMES

# -----------------------------------------------------------------------------
# Independent forward kinematics, straight from the JSON body plan -- does not
# call into fastik at all, so it's a genuine cross-check of fastik's solved
# state, mirroring the Rust benchmark's use of fastik's own (Python-unexposed)
# evaluate_fwdkin for the same purpose.
# -----------------------------------------------------------------------------
_DOF_OFFSETS = {}
_cursor = 0
for _j in JOINTS:
    _DOF_OFFSETS[_j["name"]] = _cursor
    _cursor += len(_j["dofs"])


def forward_kinematics(dof_angles, root_pos, root_rot_wxyz):
    """Returns keypoint positions in `leg_joint_names` order (root excluded),
    matching `target_ego`'s layout in the fixtures."""
    w, x, y, z = root_rot_wxyz
    world_pos = {}
    world_rot = {}
    for j in JOINTS:
        name, parent = j["name"], j["parent"]
        if parent is None:
            origin, rot = np.array(root_pos, dtype=float), R.from_quat([x, y, z, w])
        else:
            p_origin, p_rot = world_pos[parent], world_rot[parent]
            origin = p_origin + p_rot.apply(j["offset_pos"])
            qw, qx, qy, qz = j["offset_quat"]
            rot = p_rot * R.from_quat([qx, qy, qz, qw])
        dof_start = _DOF_OFFSETS[name]
        for i, dof in enumerate(j["dofs"]):
            axis = np.array(dof["axis"])
            rot = rot * R.from_rotvec(axis * dof_angles[dof_start + i])
        world_pos[name], world_rot[name] = origin, rot
    return np.array([world_pos[name] for name in LEG_JOINT_NAMES])


def build_observations(target_ego):
    obs = [fastik.KeypointObservation.missing()]
    obs += [fastik.KeypointObservation.position_3d(list(p), 1.0) for p in target_ego]
    return obs


def angle_error_deg(solved, ground_truth):
    solved, ground_truth = np.asarray(solved), np.asarray(ground_truth)
    wrapped = (solved - ground_truth + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(np.abs(wrapped)).max()


# -----------------------------------------------------------------------------
# Correctness (mirrors correctness.rs)
# -----------------------------------------------------------------------------
def run_correctness():
    tree = fastik.KinematicTree.from_json_file(str(ASSETS_DIR / "neuromechfly_ypr_legs.json"))

    print("== Synthetic exact-fit frames (bug hunt) ==")
    print(f"{'frame':>6} {'kpt rms':>16} {'kpt max':>16} {'angle err deg':>18} {'angle err deg (w=0)':>20}")
    default_solver = fastik.Solver(tree, fastik.SolverConfig())
    zero_reg_solver = fastik.Solver(tree, fastik.SolverConfig(neutral_pose_weight=0.0))
    for i, frame in enumerate(fixtures["synthetic_frames"]):
        target = np.array(frame["target_ego"])
        ground_truth = np.array(frame["ground_truth_dof_angles_per_leg"]).flatten()
        obs = build_observations(target)

        state = fastik.State.neutral_pose(tree)
        default_solver.solve(state, obs)
        solved_pts = forward_kinematics(state.dof_angles, state.root_pos, state.root_rot)
        residual = np.linalg.norm(solved_pts - target, axis=1)
        angle_err = angle_error_deg(state.dof_angles, ground_truth)

        state0 = fastik.State.neutral_pose(tree)
        zero_reg_solver.solve(state0, obs)
        angle_err0 = angle_error_deg(state0.dof_angles, ground_truth)

        print(
            f"{i:>6} {np.sqrt((residual**2).mean()):>16.6f} {residual.max():>16.6f} "
            f"{angle_err:>18.4f} {angle_err0:>20.6f}"
        )
    print(
        "(kpt rms/max: 3D distance to target, via an independent from-JSON FK "
        "replica, model units. angle err: max abs error over all 42 DOFs, "
        "degrees, mod 2*pi. \"w=0\" = neutral_pose_weight=0.)\n"
    )

    print("== Real mocap frames (cross-solver vs. flygym.ik) ==")
    seq = fastik.SequenceSolver(tree, fastik.SolverConfig())
    # Per-frame (rms, max) first, then aggregate across frames -- matching
    # correctness.rs's residual_stats/rms_of/mean/max_of exactly, so the two
    # benchmarks' numbers are directly comparable (a flat rms/mean over every
    # keypoint pooled together is a *different*, not-quite-comparable
    # statistic from a per-frame-then-aggregated one).
    fastik_rms, fastik_max, cross_rms, cross_max = [], [], [], []
    for frame in fixtures["real_frames"]:
        target = np.array(frame["target_ego"])
        state = seq.solve_frame(build_observations(target))
        solved_pts = forward_kinematics(state.dof_angles, state.root_pos, state.root_rot)
        dists = np.linalg.norm(solved_pts - target, axis=1)
        fastik_rms.append(np.sqrt((dists**2).mean()))
        fastik_max.append(dists.max())
        cross_dists = np.linalg.norm(solved_pts - np.array(frame["flygym_ik_reconstructed_ego"]), axis=1)
        cross_rms.append(np.sqrt((cross_dists**2).mean()))
        cross_max.append(cross_dists.max())
    fastik_rms, fastik_max = np.array(fastik_rms), np.array(fastik_max)
    cross_rms, cross_max = np.array(cross_rms), np.array(cross_max)
    print(f"over {len(fixtures['real_frames'])} frames:")
    print(
        f"  fastik fit residual to target:      "
        f"rms={np.sqrt((fastik_rms**2).mean()):.5f}  mean={fastik_rms.mean():.5f}  max={fastik_max.max():.5f}"
    )
    print(
        f"  cross-solver agreement (vs flygym.ik): "
        f"rms={np.sqrt((cross_rms**2).mean()):.5f}  mean={cross_rms.mean():.5f}  max={cross_max.max():.5f}\n"
    )
    return tree


# -----------------------------------------------------------------------------
# Performance (mirrors perf.rs)
# -----------------------------------------------------------------------------
def summarize(label, samples_sec):
    samples_us = np.sort(np.array(samples_sec) * 1e6)
    n = len(samples_us)
    mean = samples_us.mean()
    print(
        f"{label:<42} n={n:<7} mean={mean:>9.3f}us  median={np.median(samples_us):>9.3f}us  "
        f"p95={np.percentile(samples_us, 95):>9.3f}us  p99={np.percentile(samples_us, 99):>9.3f}us  "
        f"min={samples_us.min():>9.3f}us  max={samples_us.max():>9.3f}us  "
        f"throughput={1e6 / mean:>10.1f} calls/s"
    )


def bench_single_frame_latency(tree, target, n_calls):
    """Single-frame latency: a fresh State.neutral_pose() solved against a
    fixed real target every call (no warm start), default config -- the same
    fixture-derived target used by the Rust and C++ benchmarks."""
    config = fastik.SolverConfig()
    solver = fastik.Solver(tree, config)
    obs = build_observations(target)
    for _ in range(1000):
        state = fastik.State.neutral_pose(tree)
        solver.solve(state, obs)

    samples = []
    for _ in range(n_calls):
        state = fastik.State.neutral_pose(tree)
        t0 = time.perf_counter()
        solver.solve(state, obs)
        samples.append(time.perf_counter() - t0)
    return samples


def bench_solve_sequence(tree, all_obs, config):
    seq = fastik.SequenceSolver(tree, config)
    for obs in all_obs:
        seq.solve_frame(obs)

    timed_seq = fastik.SequenceSolver(tree, config)
    samples = []
    for obs in all_obs:
        t0 = time.perf_counter()
        timed_seq.solve_frame(obs)
        samples.append(time.perf_counter() - t0)
    return samples


# Frames per segment/thread, matching perf.rs exactly (same stride, so a
# `n_segments`-thread run gets exactly one segment per thread).
SEGMENT_LEN = 200
OVERLAP_LEN = 20


def frames_for_n_segments(n_segments):
    return SEGMENT_LEN + max(n_segments - 1, 0) * (SEGMENT_LEN - OVERLAP_LEN)


def tiled_native_rate_sequence(length):
    base = [build_observations(f["target_ego"]) for f in fixtures["native_rate_frames"]]
    return [base[i % len(base)] for i in range(length)]


def bench_multithread_sequence_throughput(tree, sequence):
    config = fastik.SolverConfig()
    segmented_config = fastik.SegmentedSolveConfig(SEGMENT_LEN, OVERLAP_LEN, 0.05)
    fastik.solve_sequence_segmented_parallel(tree, config, sequence, segmented_config)  # warm up
    t0 = time.perf_counter()
    fastik.solve_sequence_segmented_parallel(tree, config, sequence, segmented_config)
    return time.perf_counter() - t0


def run_performance(tree):
    print(f"fastik Python-bindings benchmark (state_dim={tree.n_dofs + 6})\n")

    # Same fixture-derived target used by the Rust and C++ benchmarks, so
    # this number is directly comparable across all three.
    target = np.array(fixtures["synthetic_frames"][0]["target_ego"])
    print("-- single-frame time (latency), default config (adaptive early stop) --")
    summarize("solve()", bench_single_frame_latency(tree, target, 20_000))

    print("\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --")
    native_obs = [build_observations(f["target_ego"]) for f in fixtures["native_rate_frames"]]
    summarize(
        "SequenceSolver.solve_frame",
        bench_solve_sequence(tree, native_obs, fastik.SolverConfig()),
    )

    print("\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop) --")
    try:
        n_threads = len(os.sched_getaffinity(0))
    except AttributeError:
        n_threads = os.cpu_count() or 1
    sequence = tiled_native_rate_sequence(frames_for_n_segments(n_threads))
    elapsed = bench_multithread_sequence_throughput(tree, sequence)
    print(
        f"solve_sequence_segmented_parallel   n_frames={len(sequence):<6} elapsed={elapsed * 1e3:>9.3f}ms  "
        f"throughput={len(sequence) / elapsed:>10.1f} frames/s"
    )


if __name__ == "__main__":
    tree = run_correctness()
    run_performance(tree)
