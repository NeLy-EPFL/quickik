"""Correctness cross-check and throughput/latency benchmark for quickik's
Python bindings, mirroring `benchmark/src/{correctness,perf}.rs` (the Rust
API benchmark) so the two are directly comparable.

Runs against every body listed in `BODIES` below, each with its own
body-plan + fixtures JSON pair under `assets/` (see `generate_fixtures.py`).
The real-mocap frames' flygym.ik cross-check reference values are baked into
that fixtures JSON at generation time -- this script only reads them, so
unlike `generate_fixtures.py`, it needs neither flygym nor mujoco itself, just:

  - quickik's Python extension built for this interpreter (see
    `python/README` or the top-level README's "Python" install section --
    this script does not build it for you).
  - numpy and scipy.

Run with any venv that has both, e.g. `devtools-pyenv/` (see the top-level
README) with quickik additionally installed into it:

    cd devtools-pyenv && uv sync && source .venv/bin/activate
    cd ../python && maturin develop --release
    python ../benchmark/quickik_python/bench.py

One thing the Rust benchmark can do that this one can't, since it's not
exposed to Python: forward kinematics on its own (`quickik::forward` isn't
bound). So instead of an independent FK re-check via quickik's own
(unexposed) `evaluate_fwdkin`, this script reimplements forward kinematics
directly from the JSON body plan (a fresh, independent computation, not
calling into quickik at all) for the keypoint comparisons below. The
performance benchmark's single-frame-latency target is one of the fixture's
own targets (`synthetic_frames[0]`), matching the Rust and C++ benchmarks
exactly, for a fair cross-language comparison.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quickik
from scipy.spatial.transform import Rotation as R

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BENCHMARK_DIR / "assets"

BODIES = [
    {
        "name": "neuromechfly",
        "body_plan": "neuromechfly_ypr_legs.json",
        "fixtures": "fixtures.json",
    },
    {
        "name": "g1",
        "body_plan": "g1_body_plan.json",
        "fixtures": "fixtures_g1.json",
    },
]


@dataclass
class BodyContext:
    """Everything derived from one body's body-plan + fixtures JSON pair,
    built once per body and threaded explicitly through the functions below
    instead of living in module globals."""

    name: str
    tree: "quickik.KinematicTree"
    fixtures: dict
    joints: list
    leg_joint_names: list
    dof_offsets: dict


def build_body_context(name, body_plan, fixtures):
    body_plan_path = ASSETS_DIR / body_plan
    fixtures = json.loads((ASSETS_DIR / fixtures).read_text())
    joints = json.loads(body_plan_path.read_text())["joints"]
    leg_joint_names = fixtures["leg_joint_names"]
    assert [j["name"] for j in joints][1:] == leg_joint_names

    dof_offsets = {}
    cursor = 0
    for j in joints:
        dof_offsets[j["name"]] = cursor
        cursor += len(j["dofs"])

    tree = quickik.KinematicTree.from_json_file(str(body_plan_path))
    return BodyContext(name, tree, fixtures, joints, leg_joint_names, dof_offsets)


# -----------------------------------------------------------------------------
# Independent forward kinematics, straight from the JSON body plan -- does not
# call into quickik at all, so it's a genuine cross-check of quickik's solved
# state, mirroring the Rust benchmark's use of quickik's own (Python-unexposed)
# evaluate_fwdkin for the same purpose.
# -----------------------------------------------------------------------------
def forward_kinematics(ctx, dof_angles, root_pos, root_rot_wxyz):
    """Returns keypoint positions in `ctx.leg_joint_names` order (root
    excluded), matching `target_ego`'s layout in the fixtures."""
    w, x, y, z = root_rot_wxyz
    world_pos = {}
    world_rot = {}
    for j in ctx.joints:
        name, parent = j["name"], j["parent"]
        if parent is None:
            origin, rot = np.array(root_pos, dtype=float), R.from_quat([x, y, z, w])
        else:
            p_origin, p_rot = world_pos[parent], world_rot[parent]
            origin = p_origin + p_rot.apply(j["offset_pos"])
            qw, qx, qy, qz = j["offset_quat"]
            rot = p_rot * R.from_quat([qx, qy, qz, qw])
        dof_start = ctx.dof_offsets[name]
        for i, dof in enumerate(j["dofs"]):
            axis = np.array(dof["axis"])
            rot = rot * R.from_rotvec(axis * dof_angles[dof_start + i])
        world_pos[name], world_rot[name] = origin, rot
    return np.array([world_pos[name] for name in ctx.leg_joint_names])


def build_observations(target_ego):
    obs = [quickik.KeypointObservation.missing()]
    obs += [quickik.KeypointObservation.position_3d(list(p), 1.0) for p in target_ego]
    return obs


def angle_error_deg(solved, ground_truth):
    solved, ground_truth = np.asarray(solved), np.asarray(ground_truth)
    wrapped = (solved - ground_truth + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(np.abs(wrapped)).max()


# -----------------------------------------------------------------------------
# 2D observations (mirrors quickik_rust's twod.rs): a synthetic bottom-view
# pinhole Camera, fixed once per body, plus the trivial XYView. Camera isn't
# exposed to Python for projecting points (only for use as a Solver mapper),
# so project_to_2d below reimplements its position-only projection formula
# directly -- matching quickik_core::observation::Camera::project_3d_to_2d's
# position half exactly, just without the Jacobian half this doesn't need.
# -----------------------------------------------------------------------------
def synthetic_camera(points):
    """Builds a fixed pinhole camera framing every point in `points` ((N, 3)
    array): looks straight up (+Z) from below -- a "bottom view" -- at a
    distance chosen so the whole bounding sphere (with margin) stays inside a
    60-degree field of view. fx=fy=distance (not a literal focal length):
    keeps 2D coordinates/Jacobians at the same O(1) scale as the 3D case
    instead of real pixel units, which would swamp the solver's neutral-pose
    regularization (tuned for model-unit-scale residuals) -- see
    quickik_rust's twod.rs for the full derivation. Mirrors that module's
    synthetic_camera() exactly, so both produce the same task."""
    centroid = points.mean(axis=0)
    radius = np.linalg.norm(points - centroid, axis=1).max()

    fov_deg = 60.0
    margin = 1.5
    half_fov = np.radians(fov_deg / 2.0)
    distance = float(radius * margin / np.sin(half_fov))

    forward = np.array([0.0, 0.0, 1.0])
    cam_pos = centroid - forward * distance
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.stack([right, up, forward])  # rows: right, up, forward
    world2cam_pos = -(rot @ cam_pos)

    return quickik.Camera(
        fx=distance,
        fy=distance,
        cx=0.0,
        cy=0.0,
        world2cam_pos=[float(v) for v in world2cam_pos],
        world2cam_rot_mat=[float(v) for v in rot.flatten()],  # row-major
    )


def project_to_2d(camera, pos_world3d):
    rot = np.array(camera.world2cam_rot_mat).reshape(3, 3)
    pos_cam3d = rot @ pos_world3d + np.array(camera.world2cam_pos)
    return np.array(
        [
            camera.fx * pos_cam3d[0] / pos_cam3d[2] + camera.cx,
            camera.fy * pos_cam3d[1] / pos_cam3d[2] + camera.cy,
        ]
    )


def build_observations_2d_camera(target_ego, camera):
    obs = [quickik.KeypointObservation.missing()]
    for p in target_ego:
        pos2d = project_to_2d(camera, np.asarray(p))
        obs.append(quickik.KeypointObservation.position_2d(list(pos2d), 1.0))
    return obs


def build_observations_2d_xyview(target_ego):
    obs = [quickik.KeypointObservation.missing()]
    obs += [
        quickik.KeypointObservation.position_2d([p[0], p[1]], 1.0) for p in target_ego
    ]
    return obs


# -----------------------------------------------------------------------------
# Correctness (mirrors correctness.rs)
# -----------------------------------------------------------------------------
def run_correctness(ctx):
    tree = ctx.tree

    print("== Synthetic exact-fit frames (bug hunt) ==")
    print(
        f"{'frame':>6} {'kpt rms':>16} {'kpt max':>16} {'angle err deg':>18} {'angle err deg (w=0)':>20}"
    )
    default_solver = quickik.Solver(tree, quickik.SolverConfig())
    zero_reg_solver = quickik.Solver(tree, quickik.SolverConfig(weight=0.0))
    for i, frame in enumerate(ctx.fixtures["synthetic_frames"]):
        target = np.array(frame["target_ego"])
        ground_truth = np.concatenate(
            [
                np.asarray(g, dtype=float)
                for g in frame["ground_truth_dof_angles_per_leg"]
            ]
        )
        obs = build_observations(target)

        state = quickik.State.neutral_pose(tree)
        default_solver.solve(state, obs)
        solved_pts = forward_kinematics(
            ctx, state.dof_angles, state.root_pos, state.root_rot
        )
        residual = np.linalg.norm(solved_pts - target, axis=1)
        angle_err = angle_error_deg(state.dof_angles, ground_truth)

        state0 = quickik.State.neutral_pose(tree)
        zero_reg_solver.solve(state0, obs)
        angle_err0 = angle_error_deg(state0.dof_angles, ground_truth)

        print(
            f"{i:>6} {np.sqrt((residual**2).mean()):>16.6f} {residual.max():>16.6f} "
            f"{angle_err:>18.4f} {angle_err0:>20.6f}"
        )
    print(
        "(kpt rms/max: 3D distance to target, via an independent from-JSON FK "
        f"replica, model units. angle err: max abs error over all {tree.n_dofs} "
        'DOFs, degrees, mod 2*pi. "w=0" = weight=0.)\n'
    )

    print("== Real mocap frames (cross-solver vs. flygym.ik) ==")
    seq = quickik.SequenceSolver(tree, quickik.SolverConfig())
    # Per-frame (rms, max) first, then aggregate across frames -- matching
    # correctness.rs's residual_stats/rms_of/mean/max_of exactly, so the two
    # benchmarks' numbers are directly comparable (a flat rms/mean over every
    # keypoint pooled together is a *different*, not-quite-comparable
    # statistic from a per-frame-then-aggregated one).
    quickik_rms, quickik_max, cross_rms, cross_max = [], [], [], []
    for frame in ctx.fixtures["real_frames"]:
        target = np.array(frame["target_ego"])
        state = seq.solve_frame(build_observations(target))
        solved_pts = forward_kinematics(
            ctx, state.dof_angles, state.root_pos, state.root_rot
        )
        dists = np.linalg.norm(solved_pts - target, axis=1)
        quickik_rms.append(np.sqrt((dists**2).mean()))
        quickik_max.append(dists.max())
        reconstructed = frame.get("flygym_ik_reconstructed_ego")
        if reconstructed is not None:
            cross_dists = np.linalg.norm(solved_pts - np.array(reconstructed), axis=1)
            cross_rms.append(np.sqrt((cross_dists**2).mean()))
            cross_max.append(cross_dists.max())
    quickik_rms, quickik_max = np.array(quickik_rms), np.array(quickik_max)
    print(f"over {len(ctx.fixtures['real_frames'])} frames:")
    print(
        f"  quickik fit residual to target:      "
        f"rms={np.sqrt((quickik_rms**2).mean()):.5f}  mean={quickik_rms.mean():.5f}  max={quickik_max.max():.5f}"
    )
    if cross_rms:
        cross_rms, cross_max = np.array(cross_rms), np.array(cross_max)
        print(
            f"  cross-solver agreement (vs flygym.ik): "
            f"rms={np.sqrt((cross_rms**2).mean()):.5f}  mean={cross_rms.mean():.5f}  max={cross_max.max():.5f}\n"
        )
    else:
        print(
            "  cross-solver agreement (vs flygym.ik): n/a (no reference in fixtures)\n"
        )


def run_correctness_2d_for_mapper(ctx, label, mapper, to_2d):
    """Same checks as run_correctness, but every observation is `to_2d`'s
    projection of the fixture's usual 3D target. Fit quality is still
    measured in 3D -- the distance between the solved pose's FK output and
    the *original* 3D target -- since that's the physical quantity that
    matters, even though the solver never saw it."""
    tree = ctx.tree

    print(f"== Synthetic exact-fit frames (bug hunt), 2D via {label} ==")
    print(
        f"{'frame':>6} {'kpt rms':>16} {'kpt max':>16} {'angle err deg':>18} {'angle err deg (w=0)':>20}"
    )
    default_solver = quickik.Solver(tree, quickik.SolverConfig(), mapper)
    zero_reg_solver = quickik.Solver(tree, quickik.SolverConfig(weight=0.0), mapper)
    for i, frame in enumerate(ctx.fixtures["synthetic_frames"]):
        target = np.array(frame["target_ego"])
        ground_truth = np.concatenate(
            [
                np.asarray(g, dtype=float)
                for g in frame["ground_truth_dof_angles_per_leg"]
            ]
        )
        obs = to_2d(target)

        state = quickik.State.neutral_pose(tree)
        default_solver.solve(state, obs)
        solved_pts = forward_kinematics(
            ctx, state.dof_angles, state.root_pos, state.root_rot
        )
        residual = np.linalg.norm(solved_pts - target, axis=1)
        angle_err = angle_error_deg(state.dof_angles, ground_truth)

        state0 = quickik.State.neutral_pose(tree)
        zero_reg_solver.solve(state0, obs)
        angle_err0 = angle_error_deg(state0.dof_angles, ground_truth)

        print(
            f"{i:>6} {np.sqrt((residual**2).mean()):>16.6f} {residual.max():>16.6f} "
            f"{angle_err:>18.4f} {angle_err0:>20.6f}"
        )
    print(
        "(kpt rms/max: 3D distance between solved FK output and the *original 3D* "
        f"target -- the solver itself never sees 3D, only its {label} projection. "
        f"angle err: max abs error over all {tree.n_dofs} DOFs, degrees, mod 2*pi. "
        '"w=0" = weight=0.)\n'
    )

    print(f"== Real mocap frames (cross-solver vs. flygym.ik), 2D via {label} ==")
    seq = quickik.SequenceSolver(tree, quickik.SolverConfig(), mapper)
    quickik_rms, quickik_max, cross_rms, cross_max = [], [], [], []
    for frame in ctx.fixtures["real_frames"]:
        target = np.array(frame["target_ego"])
        state = seq.solve_frame(to_2d(target))
        solved_pts = forward_kinematics(
            ctx, state.dof_angles, state.root_pos, state.root_rot
        )
        dists = np.linalg.norm(solved_pts - target, axis=1)
        quickik_rms.append(np.sqrt((dists**2).mean()))
        quickik_max.append(dists.max())
        reconstructed = frame.get("flygym_ik_reconstructed_ego")
        if reconstructed is not None:
            cross_dists = np.linalg.norm(solved_pts - np.array(reconstructed), axis=1)
            cross_rms.append(np.sqrt((cross_dists**2).mean()))
            cross_max.append(cross_dists.max())
    quickik_rms, quickik_max = np.array(quickik_rms), np.array(quickik_max)
    print(f"over {len(ctx.fixtures['real_frames'])} frames:")
    print(
        f"  quickik fit residual to target:      "
        f"rms={np.sqrt((quickik_rms**2).mean()):.5f}  mean={quickik_rms.mean():.5f}  max={quickik_max.max():.5f}"
    )
    if cross_rms:
        cross_rms, cross_max = np.array(cross_rms), np.array(cross_max)
        print(
            f"  cross-solver agreement (vs flygym.ik): "
            f"rms={np.sqrt((cross_rms**2).mean()):.5f}  mean={cross_rms.mean():.5f}  max={cross_max.max():.5f}\n"
        )
    else:
        print(
            "  cross-solver agreement (vs flygym.ik): n/a (no reference in fixtures)\n"
        )


def run_correctness_2d(ctx, camera):
    run_correctness_2d_for_mapper(
        ctx, "Camera", camera, lambda t: build_observations_2d_camera(t, camera)
    )
    run_correctness_2d_for_mapper(
        ctx, "XYView", quickik.XYView(), build_observations_2d_xyview
    )


# -----------------------------------------------------------------------------
# Performance (mirrors perf.rs)
# -----------------------------------------------------------------------------
def summarize(label, samples_sec):
    """Prints the usual latency/throughput summary and returns the mean in
    microseconds, for callers that also want the number for
    results/quickik-python-<body>.json."""
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


def bench_single_frame_latency(tree, obs, n_calls, config, mapper=None):
    """Single-frame latency: a fresh State.neutral_pose() solved against a
    fixed observation set every call (no warm start) -- the same
    fixture-derived target used by the Rust and C++ benchmarks."""
    solver = quickik.Solver(tree, config, mapper)
    for _ in range(500):
        state = quickik.State.neutral_pose(tree)
        solver.solve(state, obs)

    samples = []
    for _ in range(n_calls):
        state = quickik.State.neutral_pose(tree)
        t0 = time.perf_counter()
        solver.solve(state, obs)
        samples.append(time.perf_counter() - t0)
    return samples


def build_observation_arrays(ctx, frames):
    """positions: (n_frames, n_keypoints, 3) float32, weights: (n_frames,
    n_keypoints) float32 -- a weight <= 0 is treated as missing, matching
    build_observations' "root is never observed" convention. n_keypoints
    includes the root (index 0).

    Feeds solve_sequence_segmented_parallel directly, instead of
    building one KeypointObservation Python object per keypoint per frame
    (see build_observations) -- avoids that per-object construction and the
    matching per-object unwrapping on the Rust side.
    """
    n_keypoints = len(ctx.joints)
    targets = np.array([f["target_ego"] for f in frames], dtype=np.float32)
    positions = np.zeros((len(frames), n_keypoints, 3), dtype=np.float32)
    positions[:, 1:, :] = targets
    weights = np.zeros((len(frames), n_keypoints), dtype=np.float32)
    weights[:, 1:] = 1.0
    return positions, weights


def tile_arrays(positions, weights, length):
    idx = np.arange(length) % positions.shape[0]
    return positions[idx], weights[idx]


def bench_solve_sequence(tree, positions, weights, config):
    """Single-thread sequence throughput: one bulk call to
    solve_sequence_segmented_parallel with n_workers=1 (a single
    segment spanning the whole sequence), instead of looping solve_frame()
    once per frame from Python -- avoids paying Python/PyO3 call overhead on
    every single frame."""
    parallel_config = quickik.ParallelSolveConfig(len(positions), 0, 0.05, 1)
    quickik.solve_sequence_segmented_parallel(
        tree, config, positions, weights, parallel_config
    )  # warm up
    t0 = time.perf_counter()
    quickik.solve_sequence_segmented_parallel(
        tree, config, positions, weights, parallel_config
    )
    return time.perf_counter() - t0


# Frames per segment/worker, matching perf.rs exactly (same stride, so a
# `n_segments`-worker run gets exactly one segment per worker).
SEGMENT_LEN = 200
OVERLAP_LEN = 20
# Worker count for the main "multi-thread sequence throughput" metric,
# passed explicitly via ParallelSolveConfig.n_workers -- fixed rather than
# detected, matching perf.rs (see its comment).
MULTITHREAD_N_THREADS = 8
# Frame count for the single-thread sequence-throughput metric, tiled from
# the 300-frame native-rate fixture -- larger than the multi-thread metric's
# per-worker segment since this one has no worker count to divide by.
SINGLE_THREAD_N_FRAMES = 1000


def frames_for_n_segments(n_segments):
    return SEGMENT_LEN + max(n_segments - 1, 0) * (SEGMENT_LEN - OVERLAP_LEN)


def bench_multithread_sequence_throughput(tree, positions, weights, n_workers):
    config = quickik.SolverConfig()
    parallel_config = quickik.ParallelSolveConfig(
        SEGMENT_LEN, OVERLAP_LEN, 0.05, n_workers
    )
    quickik.solve_sequence_segmented_parallel(
        tree, config, positions, weights, parallel_config
    )  # warm up
    t0 = time.perf_counter()
    quickik.solve_sequence_segmented_parallel(
        tree, config, positions, weights, parallel_config
    )
    return time.perf_counter() - t0


def write_results_json(
    body,
    single_frame_latency_us,
    single_frame_latency_max_us,
    single_thread_throughput_fps,
    multi_thread_throughput_fps,
):
    """Writes plot/results/quickik-python-<body>.json for
    plot/plot_comparison.py to pick up."""
    results = {
        "name": "quickik-python",
        "body": body,
        "language": "python",
        "formulation": "whole-tree",
        "single_frame_latency_us": single_frame_latency_us,
        "single_frame_latency_max_us": single_frame_latency_max_us,
        "single_thread_throughput_fps": single_thread_throughput_fps,
        "multi_thread_throughput_fps": multi_thread_throughput_fps,
        "multi_thread_n_threads": MULTITHREAD_N_THREADS,
        "notes": None,
    }
    out_dir = BENCHMARK_DIR / "plot" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"quickik-python-{body}.json").write_text(json.dumps(results, indent=2))


def run_performance(ctx):
    tree = ctx.tree
    print(f"quickik Python-bindings benchmark (state_dim={tree.n_dofs + 6})\n")

    # Same fixture-derived target used by the Rust and C++ benchmarks, so
    # this number is directly comparable across all three.
    target = np.array(ctx.fixtures["synthetic_frames"][0]["target_ego"])
    target_obs = build_observations(target)
    print("-- single-frame time (latency), default config (adaptive early stop) --")
    single_frame_latency_us = summarize(
        "solve()",
        bench_single_frame_latency(tree, target_obs, 10_000, quickik.SolverConfig()),
    )

    # Early stop disabled (tolerances = 0), so every call runs the full
    # n_iterations -- the worst case if a frame never converges early.
    max_iterations_config = quickik.SolverConfig(
        position_tolerance=0.0, angle_tolerance=0.0
    )
    print(
        f"\n-- single-frame time (latency), early stop disabled ({max_iterations_config.n_iterations} iterations) --"
    )
    single_frame_latency_max_us = summarize(
        "solve() (forced max iterations)",
        bench_single_frame_latency(tree, target_obs, 10_000, max_iterations_config),
    )

    print(
        "\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --"
    )
    base_positions, base_weights = build_observation_arrays(
        ctx, ctx.fixtures["native_rate_frames"]
    )
    single_positions, single_weights = tile_arrays(
        base_positions, base_weights, SINGLE_THREAD_N_FRAMES
    )
    elapsed = bench_solve_sequence(
        tree, single_positions, single_weights, quickik.SolverConfig()
    )
    single_thread_fps = single_positions.shape[0] / elapsed
    print(
        f"solve_sequence_segmented_parallel (n_workers=1)   "
        f"n_frames={single_positions.shape[0]:<6} elapsed={elapsed * 1e3:>9.3f}ms  "
        f"throughput={single_thread_fps:>10.1f} frames/s"
    )

    print(
        f"\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop, "
        f"{MULTITHREAD_N_THREADS} threads) --"
    )
    n_frames = frames_for_n_segments(MULTITHREAD_N_THREADS)
    mt_positions, mt_weights = tile_arrays(base_positions, base_weights, n_frames)
    elapsed = bench_multithread_sequence_throughput(
        tree, mt_positions, mt_weights, MULTITHREAD_N_THREADS
    )
    multithread_fps = n_frames / elapsed
    print(
        f"solve_sequence_segmented_parallel   n_frames={n_frames:<6} "
        f"elapsed={elapsed * 1e3:>9.3f}ms  throughput={multithread_fps:>10.1f} frames/s"
    )

    write_results_json(
        ctx.name,
        single_frame_latency_us,
        single_frame_latency_max_us,
        single_thread_fps,
        multithread_fps,
    )


def tiled_native_rate_sequence_2d(frames, length, to_2d):
    """List-of-observation-lists counterpart to tile_arrays/build_observation_arrays,
    for mappers whose observations don't fit that array-based path's 3D shape."""
    base = [to_2d(f["target_ego"]) for f in frames]
    return [base[i % len(base)] for i in range(length)]


def bench_solve_sequence_2d(tree, sequence, config, mapper):
    """Single-thread sequence throughput, 2D counterpart of
    bench_solve_sequence: one bulk call to
    solve_sequence_segmented_parallel_from_observations with n_workers=1."""
    parallel_config = quickik.ParallelSolveConfig(len(sequence), 0, 0.05, 1)
    quickik.solve_sequence_segmented_parallel_from_observations(
        tree, config, sequence, parallel_config, mapper
    )  # warm up
    t0 = time.perf_counter()
    quickik.solve_sequence_segmented_parallel_from_observations(
        tree, config, sequence, parallel_config, mapper
    )
    return time.perf_counter() - t0


def bench_multithread_sequence_throughput_2d(tree, sequence, n_workers, mapper):
    config = quickik.SolverConfig()
    parallel_config = quickik.ParallelSolveConfig(
        SEGMENT_LEN, OVERLAP_LEN, 0.05, n_workers
    )
    quickik.solve_sequence_segmented_parallel_from_observations(
        tree, config, sequence, parallel_config, mapper
    )  # warm up
    t0 = time.perf_counter()
    quickik.solve_sequence_segmented_parallel_from_observations(
        tree, config, sequence, parallel_config, mapper
    )
    return time.perf_counter() - t0


def write_results_json_2d(
    body,
    observation,
    single_frame_latency_us,
    single_frame_latency_max_us,
    single_thread_throughput_fps,
    multi_thread_throughput_fps,
):
    """Writes plot/results/quickik-python-2d-<observation>-<body>.json for
    plot/plot_2d_comparison.py to pick up. Same schema as write_results_json
    plus an "observation" field ("camera"/"xyview")."""
    results = {
        "name": "quickik-python",
        "body": body,
        "language": "python",
        "formulation": "whole-tree",
        "observation": observation,
        "single_frame_latency_us": single_frame_latency_us,
        "single_frame_latency_max_us": single_frame_latency_max_us,
        "single_thread_throughput_fps": single_thread_throughput_fps,
        "multi_thread_throughput_fps": multi_thread_throughput_fps,
        "multi_thread_n_threads": MULTITHREAD_N_THREADS,
        "notes": None,
    }
    out_dir = BENCHMARK_DIR / "plot" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"quickik-python-2d-{observation}-{body}.json").write_text(
        json.dumps(results, indent=2)
    )


def run_performance_2d_for_mapper(ctx, observation, mapper, label, to_2d):
    tree = ctx.tree
    print(
        f"quickik Python-bindings benchmark, 2D via {label} (state_dim={tree.n_dofs + 6})\n"
    )

    target = np.array(ctx.fixtures["synthetic_frames"][0]["target_ego"])
    target_obs = to_2d(target)
    print("-- single-frame time (latency), default config (adaptive early stop) --")
    single_frame_latency_us = summarize(
        "solve()",
        bench_single_frame_latency(
            tree, target_obs, 10_000, quickik.SolverConfig(), mapper
        ),
    )

    max_iterations_config = quickik.SolverConfig(
        position_tolerance=0.0, angle_tolerance=0.0
    )
    print(
        f"\n-- single-frame time (latency), early stop disabled ({max_iterations_config.n_iterations} iterations) --"
    )
    single_frame_latency_max_us = summarize(
        "solve() (forced max iterations)",
        bench_single_frame_latency(
            tree, target_obs, 10_000, max_iterations_config, mapper
        ),
    )

    print(
        "\n-- single-thread sequence throughput (native-rate frames, adaptive early stop) --"
    )
    single_sequence = tiled_native_rate_sequence_2d(
        ctx.fixtures["native_rate_frames"], SINGLE_THREAD_N_FRAMES, to_2d
    )
    elapsed = bench_solve_sequence_2d(
        tree, single_sequence, quickik.SolverConfig(), mapper
    )
    single_thread_fps = len(single_sequence) / elapsed
    print(
        f"solve_sequence_segmented_parallel_from_observations (n_workers=1)   "
        f"n_frames={len(single_sequence):<6} elapsed={elapsed * 1e3:>9.3f}ms  "
        f"throughput={single_thread_fps:>10.1f} frames/s"
    )

    print(
        f"\n-- multi-thread sequence throughput (segmented parallel, adaptive early stop, "
        f"{MULTITHREAD_N_THREADS} threads) --"
    )
    n_frames = frames_for_n_segments(MULTITHREAD_N_THREADS)
    mt_sequence = tiled_native_rate_sequence_2d(
        ctx.fixtures["native_rate_frames"], n_frames, to_2d
    )
    elapsed = bench_multithread_sequence_throughput_2d(
        tree, mt_sequence, MULTITHREAD_N_THREADS, mapper
    )
    multithread_fps = n_frames / elapsed
    print(
        f"solve_sequence_segmented_parallel_from_observations   n_frames={n_frames:<6} "
        f"elapsed={elapsed * 1e3:>9.3f}ms  throughput={multithread_fps:>10.1f} frames/s\n"
    )

    write_results_json_2d(
        ctx.name,
        observation,
        single_frame_latency_us,
        single_frame_latency_max_us,
        single_thread_fps,
        multithread_fps,
    )


def run_performance_2d(ctx, camera):
    run_performance_2d_for_mapper(
        ctx,
        "camera",
        camera,
        "Camera",
        lambda t: build_observations_2d_camera(t, camera),
    )
    run_performance_2d_for_mapper(
        ctx, "xyview", quickik.XYView(), "XYView", build_observations_2d_xyview
    )


if __name__ == "__main__":
    for body in BODIES:
        print(f"===== body: {body['name']} =====\n")
        ctx = build_body_context(body["name"], body["body_plan"], body["fixtures"])
        run_correctness(ctx)
        run_performance(ctx)

        # Same task, but observed only in 2D (a synthetic pinhole camera,
        # bottom view, fixed once per body, plus the trivial XYView) --
        # mirrors quickik_rust's twod.rs. NeuroMechFly only for now, matching
        # the Rust benchmark's scope while the 2D fit is still being
        # validated for G1.
        if body["name"] == "neuromechfly":
            all_points = np.concatenate(
                [
                    np.array(
                        [f["target_ego"] for f in ctx.fixtures["synthetic_frames"]]
                    ).reshape(-1, 3),
                    np.array(
                        [f["target_ego"] for f in ctx.fixtures["native_rate_frames"]]
                    ).reshape(-1, 3),
                    np.array(
                        [f["target_ego"] for f in ctx.fixtures["real_frames"]]
                    ).reshape(-1, 3),
                ]
            )
            camera = synthetic_camera(all_points)

            run_correctness_2d(ctx, camera)
            run_performance_2d(ctx, camera)
        print()
