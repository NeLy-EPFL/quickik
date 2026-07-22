"""Correctness cross-check and throughput/latency benchmark for Pinocchio,
mirroring `benchmark/fastik_python/bench.py` / `benchmark/fastik_rust/src/perf.rs`
so all three are directly comparable.

Pinocchio has no built-in general-purpose IK solver, so this script:

  1. Builds the full neuromechfly body plan (thorax free-flyer + 6 legs x 7
     DOFs each = 42 DOFs) directly via Pinocchio's `model.addJoint`/`addFrame`
     API (no URDF), generalizing `poc_one_leg.py` to all 6 legs. See
     `build_full_model` below.
  2. Implements its own Gauss-Newton/Levenberg-Marquardt IK loop on top of
     `pin.computeJointJacobians`/`pin.getFrameJacobian`, matching fastik's
     own solver (`src/solver.rs`) as closely as possible: position-only 3-row
     residuals per keypoint, LM diagonal damping, a small neutral-pose
     Tikhonov prior, and the same early-stopping rule (position/angle delta
     tolerances), capped at `n_iterations`. See `solve_ik` below.

Run with the dedicated Python 3.12 venv (Pinocchio's wheels don't support
3.13+):

    cd /path/to/fastik/benchmark/extern/pinocchio
    .venv312/bin/python bench_pinocchio.py

See README.md for modeling compromises (mirrored-leg axis handling,
multiprocessing standing in for fastik's in-process thread pool) and the
numbers from the last run.
"""

import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parents[1]
ASSETS_DIR = BENCHMARK_DIR / "assets"
MODEL_JSON = ASSETS_DIR / "neuromechfly_ypr_legs.json"
FIXTURES_JSON = ASSETS_DIR / "fixtures.json"

# Gauss-Newton/LM config, matching fastik's SolverConfig::default() exactly
# (src/solver.rs) for a fair comparison.
N_ITERATIONS = 10
DAMPING = 1e-6
NEUTRAL_POSE_WEIGHT = 1e-3
POSITION_TOLERANCE = 1e-3
ANGLE_TOLERANCE = 1e-3

# -----------------------------------------------------------------------------
# Model construction: thorax free-flyer + 6 legs x (3+2+1+1) revolute DOFs,
# generalizing poc_one_leg.py to all 6 legs and handling mirrored (right-side)
# legs' signed axes (e.g. [-1, 0, 0]).
# -----------------------------------------------------------------------------
_AXIS_JOINTS = [pin.JointModelRX, pin.JointModelRY, pin.JointModelRZ]


def _axis_to_joint(axis):
    """Maps a (possibly negative) unit axis like [-1, 0, 0] to a Pinocchio
    RX/RY/RZ joint (always defined about the *positive* axis) plus a sign.

    Rotating by `sign * theta` about the positive axis is identical to
    rotating by `theta` about `sign * (positive axis)`, since
    R(-n, -t) = R(n, t). So a mirrored-leg DOF with axis [-1, 0, 0] and
    JSON angle `theta` is reproduced exactly by a `JointModelRX` driven with
    angle `-theta` -- no extra rotation needs to be baked into the joint
    placement. Verified against a finite-difference Jacobian check (see
    README.md).
    """
    axis = np.asarray(axis, dtype=float)
    idx = int(np.argmax(np.abs(axis)))
    sign = float(np.sign(axis[idx]))
    return _AXIS_JOINTS[idx](), sign


def build_full_model(body_plan_path=MODEL_JSON):
    """Builds the full thorax + 6-leg Pinocchio model from the JSON body plan.

    Returns:
        model: the `pin.Model`.
        keypoint_frame_ids: list of 30 operational-frame ids, one per leg
            joint node (thorax excluded), in the same order as
            `fixtures.json`'s `leg_joint_names` / `target_ego` -- verified by
            an assertion below.
        q_neutral: `model.nq`-sized neutral configuration (root at the
            origin/identity orientation, each DOF at its own JSON
            `neutral_angle`, sign-adjusted per `_axis_to_joint`).
        dof_signs: length-42 array of +-1, one per DOF, in JSON DOF-flatten
            order (matches `state.dof_angles` / `ground_truth_dof_angles_per_leg`
            convention used by fastik's own benchmarks) -- needed to convert
            a solved Pinocchio angle back to the JSON's signed-axis
            convention for cross-checks.
        dof_names: DOF names, same order as `dof_signs`.
    """
    with open(body_plan_path) as f:
        body_plan = json.load(f)
    joints = body_plan["joints"]

    model = pin.Model()
    thorax_id = model.addJoint(
        0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "thorax"
    )
    model.appendBodyToJoint(thorax_id, pin.Inertia.Zero(), pin.SE3.Identity())

    parent_joint_id = {"thorax": thorax_id}
    keypoint_frame_ids = []
    keypoint_names = []
    dof_signs = []
    dof_neutral_json = []
    dof_names = []

    for node in joints:
        name = node["name"]
        if name == "thorax":
            continue
        parent_id = parent_joint_id[node["parent"]]
        qw, qx, qy, qz = node["offset_quat"]
        offset = pin.SE3(
            pin.Quaternion(qw, qx, qy, qz).matrix(),
            np.array(node["offset_pos"], dtype=float),
        )

        if not node["dofs"]:
            # Leaf keypoint with no DOFs (claw tip): fixed operational frame.
            frame = pin.Frame(name, parent_id, 0, offset, pin.FrameType.OP_FRAME)
            keypoint_frame_ids.append(model.addFrame(frame))
            keypoint_names.append(name)
            continue

        # One single-DOF revolute joint per scalar DOF; only the first
        # carries the translational offset from the parent keypoint, the
        # rest are collocated (identity placement).
        current_parent = parent_id
        placement = offset
        last_joint_id = None
        for dof in node["dofs"]:
            joint_model, sign = _axis_to_joint(dof["axis"])
            joint_id = model.addJoint(
                current_parent, joint_model, placement, dof["name"]
            )
            model.appendBodyToJoint(joint_id, pin.Inertia.Zero(), pin.SE3.Identity())
            dof_signs.append(sign)
            dof_neutral_json.append(dof["neutral_angle"])
            dof_names.append(dof["name"])
            current_parent = joint_id
            placement = pin.SE3.Identity()
            last_joint_id = joint_id

        parent_joint_id[name] = last_joint_id
        # Operational frame at this node's own keypoint (tip of its DOF
        # chain -- position is independent of that chain's own rotations).
        frame = pin.Frame(name, last_joint_id, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME)
        keypoint_frame_ids.append(model.addFrame(frame))
        keypoint_names.append(name)

    fixtures = json.loads(FIXTURES_JSON.read_text())
    assert keypoint_names == fixtures["leg_joint_names"], (
        "keypoint order must match fixtures.json's leg_joint_names for "
        "target_ego indexing to line up"
    )

    dof_signs = np.array(dof_signs)
    dof_neutral_pin = dof_signs * np.array(dof_neutral_json)
    q_neutral = pin.neutral(model)
    q_neutral[7:] = dof_neutral_pin

    return model, keypoint_frame_ids, q_neutral, dof_signs, dof_names


# -----------------------------------------------------------------------------
# Gauss-Newton/LM inverse kinematics, matching src/solver.rs's math shape.
# -----------------------------------------------------------------------------
def solve_ik(model, data, keypoint_frame_ids, target, q0, neutral_q, disable_early_stop=False):
    """Runs up to `N_ITERATIONS` Gauss-Newton steps from `q0` toward `target`
    (an (n_keypoints, 3) array of world positions, one per
    `keypoint_frame_ids` entry -- the thorax has no residual term, matching
    fastik's `Missing` convention for the root keypoint).

    `disable_early_stop=True` always runs the full `N_ITERATIONS`, ignoring
    `POSITION_TOLERANCE`/`ANGLE_TOLERANCE` -- the worst case if a frame never
    converges early.

    Returns the converged configuration `q` (size `model.nq`).
    """
    q = q0.copy()
    nv = model.nv
    idx_root = np.arange(6)
    idx_dofs = np.arange(6, nv)

    for _ in range(N_ITERATIONS):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        jtj = np.zeros((nv, nv))
        jtr = np.zeros(nv)
        for k, fid in enumerate(keypoint_frame_ids):
            residual = target[k] - data.oMf[fid].translation
            jac = pin.getFrameJacobian(
                model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3, :]
            jtj += jac.T @ jac
            jtr += jac.T @ residual

        # Neutral-pose Tikhonov prior on the leg DOFs only (not the root),
        # matching accumulate_neutral_pose_prior in solver.rs.
        jtj[idx_dofs, idx_dofs] += NEUTRAL_POSE_WEIGHT
        jtr[idx_dofs] += NEUTRAL_POSE_WEIGHT * (neutral_q[7:] - q[7:])

        # Levenberg-Marquardt relative damping on the full diagonal.
        diag = np.diagonal(jtj)
        jtj[np.diag_indices(nv)] += DAMPING * np.maximum(diag, 1.0)

        try:
            delta = np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
            delta = np.zeros(nv)

        q = pin.integrate(model, q, delta)

        if disable_early_stop:
            continue
        max_pos = np.abs(delta[idx_root[:3]]).max()
        max_ang = np.abs(delta[idx_root[3:].tolist() + idx_dofs.tolist()]).max()
        if max_pos <= POSITION_TOLERANCE and max_ang <= ANGLE_TOLERANCE:
            break

    return q


# -----------------------------------------------------------------------------
# Correctness cross-check (mirrors bench.py's run_correctness, abbreviated).
# -----------------------------------------------------------------------------
def run_correctness(model, data, keypoint_frame_ids, q_neutral, dof_signs, fixtures):
    print("== Synthetic exact-fit frames (bug hunt) ==")
    print(f"{'frame':>6} {'kpt rms':>16} {'kpt max':>16} {'angle err deg':>18}")
    for i, frame in enumerate(fixtures["synthetic_frames"]):
        target = np.array(frame["target_ego"])
        ground_truth = np.array(frame["ground_truth_dof_angles_per_leg"]).flatten()

        q = solve_ik(model, data, keypoint_frame_ids, target, q_neutral, q_neutral)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        solved_pts = np.array([data.oMf[fid].translation for fid in keypoint_frame_ids])
        residual = np.linalg.norm(solved_pts - target, axis=1)

        solved_angles_json = dof_signs * q[7:]
        wrapped = (solved_angles_json - ground_truth + np.pi) % (2 * np.pi) - np.pi
        angle_err_deg = np.degrees(np.abs(wrapped)).max()

        print(
            f"{i:>6} {np.sqrt((residual**2).mean()):>16.6f} {residual.max():>16.6f} "
            f"{angle_err_deg:>18.4f}"
        )
    print(
        "(kpt rms/max: 3D distance to target, model units. angle err: max abs "
        "error over all 42 DOFs, degrees, mod 2*pi, converted back to the "
        "JSON's signed-axis convention via dof_signs.)\n"
    )


# -----------------------------------------------------------------------------
# Performance (mirrors perf.rs / fastik_python/bench.py).
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
    return mean


def bench_single_frame_latency(model, data, keypoint_frame_ids, q_neutral, target, n_calls, disable_early_stop=False):
    """One IK solve from the fixed neutral configuration against a fixed
    target every call (no warm start) -- the same fixture-derived target
    used by the Rust/C++/Python fastik benchmarks."""
    for _ in range(1000):
        solve_ik(model, data, keypoint_frame_ids, target, q_neutral, q_neutral, disable_early_stop)

    samples = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        solve_ik(model, data, keypoint_frame_ids, target, q_neutral, q_neutral, disable_early_stop)
        samples.append(time.perf_counter() - t0)
    return samples


def bench_single_thread_sequence(model, data, keypoint_frame_ids, q_neutral, targets):
    """Warm-started sequence solve: frame i+1's initial q is frame i's
    converged q, matching SequenceSolver's own warm-starting."""
    q = q_neutral.copy()
    for target in targets:
        q = solve_ik(model, data, keypoint_frame_ids, target, q, q_neutral)

    q = q_neutral.copy()
    samples = []
    for target in targets:
        t0 = time.perf_counter()
        q = solve_ik(model, data, keypoint_frame_ids, target, q, q_neutral)
        samples.append(time.perf_counter() - t0)
    return samples


# Multi-thread ("multi-process") sequence throughput -----------------------
# Python's GIL means CPU-bound numpy/Pinocchio code can't run in real
# parallel threads, so this uses multiprocessing (separate processes)
# instead of fastik's in-process thread pool -- see README.md.
MULTITHREAD_N_PROCESSES = 8
CHUNK_LEN = 300  # frames per process, tiled from native_rate_frames if needed

_worker_state = {}


def _init_worker():
    model, keypoint_frame_ids, q_neutral, _, _ = build_full_model()
    _worker_state["model"] = model
    _worker_state["data"] = model.createData()
    _worker_state["keypoint_frame_ids"] = keypoint_frame_ids
    _worker_state["q_neutral"] = q_neutral


def _solve_chunk(chunk_targets):
    model = _worker_state["model"]
    data = _worker_state["data"]
    keypoint_frame_ids = _worker_state["keypoint_frame_ids"]
    q_neutral = _worker_state["q_neutral"]
    q = q_neutral.copy()  # cold start at the chunk's beginning
    for target in chunk_targets:
        q = solve_ik(model, data, keypoint_frame_ids, target, q, q_neutral)
    return None


def bench_multithread_sequence_throughput(chunks):
    """Solves `len(chunks)` chunks in parallel processes, each warm-started
    within itself but cold at its own start. Measures wall-clock time from
    dispatch to all chunks done."""
    with multiprocessing.Pool(MULTITHREAD_N_PROCESSES, initializer=_init_worker) as pool:
        pool.map(_solve_chunk, chunks)  # warm up (spawns + JITs numpy, etc.)
        t0 = time.perf_counter()
        pool.map(_solve_chunk, chunks)
        elapsed = time.perf_counter() - t0
    return elapsed


def write_results_json(
    single_frame_latency_us, single_frame_latency_max_us, single_thread_throughput_fps, multi_thread_throughput_fps
):
    results = {
        "name": "pinocchio",
        "language": "python",
        "formulation": "whole-tree",
        "single_frame_latency_us": single_frame_latency_us,
        "single_frame_latency_max_us": single_frame_latency_max_us,
        "single_thread_throughput_fps": single_thread_throughput_fps,
        "multi_thread_throughput_fps": multi_thread_throughput_fps,
        "multi_thread_n_threads": MULTITHREAD_N_PROCESSES,
        "notes": (
            "Pinocchio has no built-in IK solver; this benchmarks a hand-written "
            "Gauss-Newton/LM loop (position-only residuals, LM damping, neutral-"
            "pose prior, early stopping) on top of pin.computeJointJacobians/"
            "getFrameJacobian, matching fastik's SolverConfig::default() math "
            "shape. 'multi-thread' throughput uses multiprocessing (8 separate "
            "processes), not an in-process thread pool, since Python's GIL "
            "prevents real multi-threaded parallelism for CPU-bound numpy/"
            "Pinocchio code -- not directly comparable to fastik's in-process "
            "segmented solve."
        ),
    }
    out_dir = BENCHMARK_DIR / "plot" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pinocchio.json").write_text(json.dumps(results, indent=2))


def run_performance(model, data, keypoint_frame_ids, q_neutral, fixtures):
    print(f"Pinocchio benchmark (nq={model.nq}, nv={model.nv})\n")

    target = np.array(fixtures["synthetic_frames"][0]["target_ego"])
    print("-- single-frame time (latency), no warm start --")
    single_frame_latency_us = summarize(
        "solve_ik()",
        bench_single_frame_latency(model, data, keypoint_frame_ids, q_neutral, target, 20_000),
    )

    # Early stop disabled, so every call runs the full N_ITERATIONS -- the
    # worst case if a frame never converges early.
    print(f"\n-- single-frame time (latency), early stop disabled ({N_ITERATIONS} iterations) --")
    single_frame_latency_max_us = summarize(
        "solve_ik() (forced max iterations)",
        bench_single_frame_latency(model, data, keypoint_frame_ids, q_neutral, target, 20_000, disable_early_stop=True),
    )

    print("\n-- single-thread sequence throughput (native-rate frames, warm start) --")
    native_targets = [np.array(f["target_ego"]) for f in fixtures["native_rate_frames"]]
    single_thread_mean_us = summarize(
        "solve_ik() (warm-started)",
        bench_single_thread_sequence(model, data, keypoint_frame_ids, q_neutral, native_targets),
    )

    print(
        f"\n-- multi-thread sequence throughput (multiprocessing, "
        f"{MULTITHREAD_N_PROCESSES} processes x {CHUNK_LEN} frames each) --"
    )
    chunks = [
        [native_targets[i % len(native_targets)] for i in range(p * CHUNK_LEN, (p + 1) * CHUNK_LEN)]
        for p in range(MULTITHREAD_N_PROCESSES)
    ]
    total_frames = sum(len(c) for c in chunks)
    elapsed = bench_multithread_sequence_throughput(chunks)
    multithread_fps = total_frames / elapsed
    print(
        f"solve (multiprocess)                n_frames={total_frames:<6} elapsed={elapsed * 1e3:>9.3f}ms  "
        f"throughput={multithread_fps:>10.1f} frames/s"
    )

    write_results_json(single_frame_latency_us, single_frame_latency_max_us, 1e6 / single_thread_mean_us, multithread_fps)


def main():
    fixtures = json.loads(FIXTURES_JSON.read_text())
    model, keypoint_frame_ids, q_neutral, dof_signs, dof_names = build_full_model()
    data = model.createData()

    run_correctness(model, data, keypoint_frame_ids, q_neutral, dof_signs, fixtures)
    run_performance(model, data, keypoint_frame_ids, q_neutral, fixtures)


if __name__ == "__main__":
    main()
