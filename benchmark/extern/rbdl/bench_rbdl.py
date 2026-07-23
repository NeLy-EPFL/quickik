"""Throughput/latency benchmark for RBDL's Python (Cython) bindings, mirroring
`bench_rbdl.cpp` (the native-C++ RBDL benchmark) and `../pinocchio/bench_pinocchio.py`
(the closest existing Python benchmark in this repo) so all are directly
comparable.

Unlike `bench_pinocchio.py` (which hand-writes a Gauss-Newton loop because
Pinocchio has no built-in IK solver), this benchmarks RBDL's own
`InverseKinematicsConstraintSet` solver -- the exact same solver
`bench_rbdl.cpp` benchmarks -- called through RBDL's Cython wrapper
(`rbdl-src/python/rbdl-wrapper.pyx`), which is not built by default upstream
and had to be built from source for this benchmark. See `README.md`'s
"Python bindings" section for the exact build commands and where the
compiled `rbdl.so` module lives.

Model construction (`build_model` below) is a line-for-line port of
`bench_rbdl.cpp`'s `build_model`/`neutral_q`, including the same modeling
compromise: RBDL's native `JointTypeFloatingBase` (quaternion joint) crashes
`InverseKinematicsConstraintSet` (an upstream bug, see `bench_rbdl.cpp`'s
header comment), so the free-floating thorax root is built as
`JointTypeTranslationXYZ` + `JointTypeEulerZYX` in series instead. Solver
tuning (`lambda=1e-6, max_steps=10, step_tol=1e-3`) is copied verbatim from
`bench_rbdl.cpp` -- literally QuickIK's own `SolverConfig::default()` values.
RBDL's own defaults (`max_steps=300, step_tol=1e-10`) are far tighter than
this problem needs and burn far more time for the same accuracy on real,
imperfectly-fittable mocap data.

Run with the dedicated Python 3.12 venv built for the RBDL Cython wrapper
(see README.md):

    cd /path/to/quickik/benchmark/extern/rbdl
    .venv312/bin/python bench_rbdl.py
"""

import json
import multiprocessing
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parents[1]
ASSETS_DIR = BENCHMARK_DIR / "assets"

# One body to benchmark: its body plan and matching fixtures file.
BODIES = [
    {
        "name": "neuromechfly",
        "body_plan": "neuromechfly_ypr_legs.json",
        "fixtures": "fixtures.json",
    },
    {"name": "g1", "body_plan": "g1_body_plan.json", "fixtures": "fixtures_g1.json"},
]

# The Cython wrapper is built (via RBDL's own CMake, RBDL_BUILD_PYTHON_WRAPPER=ON)
# into rbdl-src/build-python/python/rbdl.so -- not pip-installed, so it must be
# put on sys.path explicitly. See README.md's "Python bindings" section.
RBDL_PYTHON_DIR = HERE / "rbdl-src" / "build-python" / "python"
sys.path.insert(0, str(RBDL_PYTHON_DIR))

import numpy as np  # noqa: E402
import rbdl  # noqa: E402

# Damped Levenberg-Marquardt tuning for InverseKinematicsConstraintSet --
# copied verbatim from bench_rbdl.cpp (see that file's header comment and
# README.md for why).
LAMBDA = 1e-6
MAX_STEPS = 10
STEP_TOL = 1e-3


# -----------------------------------------------------------------------------
# Model construction: thorax (TranslationXYZ + EulerZYX in series) + 6 legs x
# (3+2+1+1) revolute DOFs, expanding each multi-dof JSON joint into a chain of
# 1-dof JointTypeRevolute bodies -- a line-for-line port of bench_rbdl.cpp's
# build_model/neutral_q.
# -----------------------------------------------------------------------------
def build_model(body_plan_path):
    """Builds the thorax + 6-leg RBDL model from the JSON body plan.

    Returns:
        model: the `rbdl.Model`.
        keypoint_body: length-31 list of RBDL body ids, one per JSON joint
            (index 0 is the thorax root -- unused as an IK target, matching
            QuickIK's `Missing`-root convention).
        keypoint_point: length-31 list of length-3 numpy arrays, the local
            point on `keypoint_body[i]` representing joint i's own
            (pre-own-rotation) position.
        q_neutral: `model.q_size`-sized neutral configuration (root at the
            origin, each leg DOF at its own JSON `neutral_angle`).
    """
    body_plan = json.loads(Path(body_plan_path).read_text())
    joints = body_plan["joints"]
    name_to_idx = {j["name"]: i for i, j in enumerate(joints)}

    model = rbdl.Model()
    model.gravity = np.zeros(3)
    null_body = rbdl.Body()

    # Floating thorax root: TranslationXYZ + EulerZYX in series (see module
    # docstring / bench_rbdl.cpp for why not JointTypeFloatingBase).
    trans_id = model.AddBody(
        0,
        rbdl.SpatialTransform(),
        rbdl.Joint(joint_type="JointTypeTranslationXYZ"),
        null_body,
    )
    thorax_id = model.AddBody(
        trans_id,
        rbdl.SpatialTransform(),
        rbdl.Joint(joint_type="JointTypeEulerZYX"),
        null_body,
    )

    n = len(joints)
    keypoint_body = [0] * n
    keypoint_point = [np.zeros(3)] * n
    tip_body = [0] * n  # tip_body[i]: RBDL body id to hook joint i's children onto
    keypoint_body[0] = thorax_id
    tip_body[0] = thorax_id

    dof_q_index = []
    neutral_angles = []

    for i in range(1, n):
        node = joints[i]
        hook = tip_body[name_to_idx[node["parent"]]]
        offset = np.array(node["offset_pos"], dtype=float)

        if not node["dofs"]:
            # Leaf keypoint (a claw): no RBDL body of its own.
            keypoint_body[i] = hook
            keypoint_point[i] = offset
            tip_body[i] = hook
            continue

        b = hook
        first_body = None
        for k, dof in enumerate(node["dofs"]):
            # Only the first dof in the chain carries the joint's own
            # translational offset; later dofs use a zero-offset frame.
            off = offset if k == 0 else np.zeros(3)
            frame = rbdl.SpatialTransform()
            frame.r = off
            axis = np.array(dof["axis"], dtype=float)
            # A 1-dof revolute joint about an arbitrary axis is a single
            # spatial-vector joint (angular part = axis, linear part = 0) --
            # crbdl.pxd doesn't expose RBDL's Joint(JointType, Vector3d)
            # convenience constructor, only this more general one.
            spatial_axis = np.array([axis[0], axis[1], axis[2], 0.0, 0.0, 0.0])
            joint = rbdl.Joint(axes=[spatial_axis])
            body = rbdl.Body()
            b = model.AddBody(b, frame, joint, body)
            if first_body is None:
                first_body = b
            dof_q_index.append(model.mJoints[b].q_index)
            neutral_angles.append(dof["neutral_angle"])

        keypoint_body[i] = first_body
        keypoint_point[i] = np.zeros(3)
        tip_body[i] = b

    q_neutral = np.zeros(model.q_size)
    for d, q_index in enumerate(dof_q_index):
        q_neutral[q_index] = neutral_angles[d]

    return model, keypoint_body, keypoint_point, q_neutral


def build_cs(keypoint_body, keypoint_point, target, step_tol=STEP_TOL):
    """Builds an IK constraint set for one target frame: one point constraint
    per non-root joint, in joint order (matching target's order 1:1).
    `step_tol=0` disables early stopping, forcing every solve to run the full
    `MAX_STEPS`."""
    cs = rbdl.InverseKinematicsConstraintSet()
    cs.dlambda = LAMBDA  # "lambda" is a Python keyword -> exposed as dlambda
    cs.max_steps = MAX_STEPS
    cs.step_tol = step_tol
    cs.constraint_tol = step_tol
    for k in range(target.shape[0]):
        cs.AddPointConstraint(keypoint_body[k + 1], keypoint_point[k + 1], target[k])
    return cs


def set_targets(cs, target):
    """Mutates an existing constraint set's targets in place (constraint
    body/point list is unchanged frame to frame) -- avoids rebuilding the
    whole constraint set every frame in the sequence benchmarks.

    `cs.target_positions` is a getter-only wrapper property (RBDL's Cython
    wrappergen only emits `__get__` for `VectorWrapperAddProperty` members,
    see rbdl-src/python/wrappergen.py): every access reconstructs a fresh
    Python list of Vector3d objects that each alias the live C++ memory
    (`address=&thisptr.target_positions[i]`). So `cs.target_positions[k] =
    ...` is a silent no-op -- it assigns into that throwaway list, not
    through to the C++ vector. The correct fix is to fetch the list once
    and mutate each aliased Vector3d in place via its own `__setitem__`.
    """
    tp = cs.target_positions
    for k in range(target.shape[0]):
        tp[k][:] = target[k]


def residual_stats(model, keypoint_body, keypoint_point, q, target):
    """3D distance rms/max between a solved q's keypoints and their targets,
    via CalcBodyToBaseCoordinates -- an independent check of cs.error_norm."""
    achieved = np.array(
        [
            rbdl.CalcBodyToBaseCoordinates(
                model, q, keypoint_body[k + 1], keypoint_point[k + 1], False
            )
            for k in range(target.shape[0])
        ]
    )
    d = np.linalg.norm(achieved - target, axis=1)
    return float(np.sqrt((d**2).mean())), float(d.max())


# -----------------------------------------------------------------------------
# Correctness sanity check (quick, mirrors bench_rbdl.cpp's run_correctness).
# -----------------------------------------------------------------------------
def run_correctness(model, keypoint_body, keypoint_point, q_neutral, fixtures):
    print("== Synthetic exact-fit frames (cold from neutral) ==")
    print(f"{'frame':>6} {'steps':>10} {'rms':>14} {'max':>14}")
    q_out = np.zeros(model.q_size)
    for frame in fixtures["synthetic_frames"]:
        target = np.array(frame["target_ego"], dtype=float)
        cs = build_cs(keypoint_body, keypoint_point, target)
        rbdl.InverseKinematicsCS(model, q_neutral, cs, q_out)
        rms, max_d = residual_stats(model, keypoint_body, keypoint_point, q_out, target)
        print(f"{frame['frame']:>6} {cs.num_steps:>10} {rms:>14.3e} {max_d:>14.3e}")

    print("\n== Real native-rate frames (warm-started, 300 frames) ==")
    native_frames = [
        np.array(f["target_ego"], dtype=float) for f in fixtures["native_rate_frames"]
    ]
    cs = build_cs(keypoint_body, keypoint_point, native_frames[0])
    q = q_neutral.copy()
    rms_all, max_all = [], []
    for target in native_frames:
        set_targets(cs, target)
        rbdl.InverseKinematicsCS(model, q, cs, q_out)
        rms, max_d = residual_stats(model, keypoint_body, keypoint_point, q_out, target)
        rms_all.append(rms)
        max_all.append(max_d)
        q = q_out.copy()
    print(
        f"residual to target (model units): mean_rms={np.mean(rms_all):.4e}  "
        f"max_rms={np.max(rms_all):.4e}  max={np.max(max_all):.4e}"
    )
    print(
        "(real mocap frames don't perfectly satisfy this exact rigid rotation-axis "
        "model -- see README.md.)\n"
    )


# -----------------------------------------------------------------------------
# Performance (mirrors bench_rbdl.cpp / ../pinocchio/bench_pinocchio.py).
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


def bench_single_frame_latency(
    model,
    keypoint_body,
    keypoint_point,
    q_neutral,
    target,
    n_calls,
    n_warmup,
    step_tol=STEP_TOL,
):
    """One IK solve from the fixed neutral configuration against a fixed
    target every call (no warm start)."""
    cs = build_cs(keypoint_body, keypoint_point, target, step_tol)
    q_out = np.zeros(model.q_size)
    for _ in range(n_warmup):
        rbdl.InverseKinematicsCS(model, q_neutral, cs, q_out)

    samples = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        rbdl.InverseKinematicsCS(model, q_neutral, cs, q_out)
        samples.append(time.perf_counter() - t0)
    return samples


def bench_single_thread_sequence(
    model, keypoint_body, keypoint_point, q_neutral, frames
):
    """Warm-started sequential solve: frame i's initial q is frame i-1's
    converged q."""
    cs = build_cs(keypoint_body, keypoint_point, frames[0])
    q_out = np.zeros(model.q_size)

    q = q_neutral.copy()
    for target in frames:  # untimed warmup pass
        set_targets(cs, target)
        rbdl.InverseKinematicsCS(model, q, cs, q_out)
        q = q_out.copy()

    q = q_neutral.copy()
    samples = []
    for target in frames:
        set_targets(cs, target)
        t0 = time.perf_counter()
        rbdl.InverseKinematicsCS(model, q, cs, q_out)
        samples.append(time.perf_counter() - t0)
        q = q_out.copy()
    return samples


# Multi-thread ("multi-process") sequence throughput -----------------------
# Python's GIL means CPU-bound Cython/RBDL code can't run in real parallel
# threads, so this uses multiprocessing (separate processes) instead of
# QuickIK's in-process thread pool -- same approach as
# ../pinocchio/bench_pinocchio.py's bench_multithread_sequence_throughput.
MULTITHREAD_N_PROCESSES = 8
CHUNK_LEN = 300  # frames per process, tiled from native_rate_frames if needed

_worker_state = {}


def _init_worker(body_plan_path):
    model, keypoint_body, keypoint_point, q_neutral = build_model(body_plan_path)
    _worker_state["model"] = model
    _worker_state["keypoint_body"] = keypoint_body
    _worker_state["keypoint_point"] = keypoint_point
    _worker_state["q_neutral"] = q_neutral


def _solve_chunk(chunk_targets):
    model = _worker_state["model"]
    keypoint_body = _worker_state["keypoint_body"]
    keypoint_point = _worker_state["keypoint_point"]
    q_neutral = _worker_state["q_neutral"]

    cs = build_cs(keypoint_body, keypoint_point, chunk_targets[0])
    q_out = np.zeros(model.q_size)
    q = q_neutral.copy()  # cold start at the chunk's beginning
    for target in chunk_targets:
        set_targets(cs, target)
        rbdl.InverseKinematicsCS(model, q, cs, q_out)
        q = q_out.copy()
    return None


def bench_multithread_sequence_throughput(chunks, body_plan_path):
    """Solves `len(chunks)` chunks in parallel processes, each warm-started
    within itself but cold at its own start. Measures wall-clock time from
    dispatch to all chunks done."""
    with multiprocessing.Pool(
        MULTITHREAD_N_PROCESSES, initializer=_init_worker, initargs=(body_plan_path,)
    ) as pool:
        pool.map(_solve_chunk, chunks)  # warm up (spawns processes, builds models)
        t0 = time.perf_counter()
        pool.map(_solve_chunk, chunks)
        elapsed = time.perf_counter() - t0
    return elapsed


def write_results_json(
    body,
    single_frame_latency_us,
    single_frame_latency_max_us,
    single_thread_throughput_fps,
    multi_thread_throughput_fps,
):
    results = {
        "name": "rbdl-python",
        "body": body,
        "language": "python",
        "formulation": "whole-tree",
        "single_frame_latency_us": single_frame_latency_us,
        "single_frame_latency_max_us": single_frame_latency_max_us,
        "single_thread_throughput_fps": single_thread_throughput_fps,
        "multi_thread_throughput_fps": multi_thread_throughput_fps,
        "multi_thread_n_threads": MULTITHREAD_N_PROCESSES,
        "notes": (
            "Calls RBDL's real InverseKinematicsConstraintSet solver (same solver "
            "bench_rbdl.cpp benchmarks) through RBDL's own Cython wrapper "
            "(rbdl-src/python/rbdl-wrapper.pyx), which is never built by default "
            "upstream (RBDL_BUILD_PYTHON_WRAPPER=OFF) and had to be built from "
            "source for this benchmark -- see README.md's 'Python bindings' "
            "section for the exact CMake invocation. Model construction "
            "(TranslationXYZ + EulerZYX in series for the floating thorax root, "
            "instead of RBDL's native JointTypeFloatingBase, which crashes "
            "InverseKinematicsConstraintSet -- an upstream bug) and "
            "solver tuning (lambda=1e-6, max_steps=10, step_tol=1e-3, literally "
            "QuickIK's own SolverConfig::default() values) are copied verbatim "
            "from bench_rbdl.cpp for an apples-to-apples Python-vs-C++ "
            "comparison. All 30 leg keypoints are fit jointly in one "
            "InverseKinematicsCS() call per frame, same as the C++ benchmark -- "
            "unlike pinocchio.json's hand-written Python IK loop (30 separate "
            "per-keypoint Jacobian calls per Gauss-Newton iteration), this "
            "benchmark makes exactly one Cython-wrapped solver call per frame, so "
            "Python/Cython call overhead is amortized over the whole solve "
            "rather than paid 30x per iteration. 'multi-thread' throughput uses "
            "multiprocessing (8 separate processes, each with its own Model "
            "instance, 300-frame chunks tiled from native_rate_frames as needed), "
            "not an in-process thread pool, since Python's GIL prevents real "
            "multi-threaded parallelism for this CPU-bound Cython code -- not "
            "directly comparable to rbdl.json's std::thread-based multi-thread "
            "metric, which shares the OS scheduler/cache differently."
        ),
    }
    out_dir = BENCHMARK_DIR / "plot" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rbdl-python-{body}.json").write_text(json.dumps(results, indent=2))


def run_performance(
    body_name, body_plan_path, model, keypoint_body, keypoint_point, q_neutral, fixtures
):
    n_dofs = model.q_size - 6
    print(
        f"RBDL model (Python/Cython bindings): q_size={model.q_size} (6 floating-base + {n_dofs} leg dofs)\n"
    )

    target = np.array(fixtures["synthetic_frames"][0]["target_ego"], dtype=float)
    print("-- single-frame time (latency), no warm start --")
    single_frame_latency_us = summarize(
        "InverseKinematicsCS() (cold)",
        bench_single_frame_latency(
            model, keypoint_body, keypoint_point, q_neutral, target, 20_000, 1000
        ),
    )

    # step_tol=0 disables early stopping, forcing every solve to run the full
    # MAX_STEPS -- the worst case if a frame never converges early.
    print(
        f"\n-- single-frame time (latency), early stop disabled ({MAX_STEPS} steps) --"
    )
    single_frame_latency_max_us = summarize(
        "InverseKinematicsCS() (forced max steps)",
        bench_single_frame_latency(
            model,
            keypoint_body,
            keypoint_point,
            q_neutral,
            target,
            20_000,
            1000,
            step_tol=0.0,
        ),
    )

    print(
        "\n-- single-thread sequence throughput (native-rate frames, warm-started) --"
    )
    native_targets = [
        np.array(f["target_ego"], dtype=float) for f in fixtures["native_rate_frames"]
    ]
    single_thread_mean_us = summarize(
        "InverseKinematicsCS() (warm)",
        bench_single_thread_sequence(
            model, keypoint_body, keypoint_point, q_neutral, native_targets
        ),
    )

    print(
        f"\n-- multi-thread sequence throughput (multiprocessing, "
        f"{MULTITHREAD_N_PROCESSES} processes x {CHUNK_LEN} frames each) --"
    )
    chunks = [
        [
            native_targets[i % len(native_targets)]
            for i in range(p * CHUNK_LEN, (p + 1) * CHUNK_LEN)
        ]
        for p in range(MULTITHREAD_N_PROCESSES)
    ]
    total_frames = sum(len(c) for c in chunks)
    elapsed = bench_multithread_sequence_throughput(chunks, body_plan_path)
    multithread_fps = total_frames / elapsed
    print(
        f"solve (multiprocess)                 n_frames={total_frames:<6} elapsed={elapsed * 1e3:>9.3f}ms  "
        f"throughput={multithread_fps:>10.1f} frames/s"
    )

    write_results_json(
        body_name,
        single_frame_latency_us,
        single_frame_latency_max_us,
        1e6 / single_thread_mean_us,
        multithread_fps,
    )
    print(f"\nWrote ../../plot/results/rbdl-python-{body_name}.json")


if __name__ == "__main__":
    for body in BODIES:
        print(f"\n########## body: {body['name']} ##########\n")

        body_plan_path = ASSETS_DIR / body["body_plan"]
        fixtures_path = ASSETS_DIR / body["fixtures"]

        fixtures = json.loads(fixtures_path.read_text())
        assert [j["name"] for j in json.loads(body_plan_path.read_text())["joints"]][
            1:
        ] == fixtures["leg_joint_names"], (
            f"joint order must match {fixtures_path.name}'s leg_joint_names for target_ego indexing to line up"
        )

        model, keypoint_body, keypoint_point, q_neutral = build_model(body_plan_path)

        run_correctness(model, keypoint_body, keypoint_point, q_neutral, fixtures)
        run_performance(
            body["name"],
            body_plan_path,
            model,
            keypoint_body,
            keypoint_point,
            q_neutral,
            fixtures,
        )
