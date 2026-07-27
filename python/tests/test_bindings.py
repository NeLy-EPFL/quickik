"""Tests for QuickIK's Python bindings, mirroring cpp/tests/test_main.cpp and
tests/solver_test.rs / tests/high_level_test.rs. Uses the same "two-joint
chain" fixture as those (see tests/common/mod.rs): a root, joint1 and joint2
(each with one Z-axis DOF, joint2 limited to [-0.5, 0.5]), and a trailing
fixed tip.

Forward kinematics isn't exposed to Python (same as C++), so
`two_link_positions` below computes the four keypoints' world positions
directly from the chain's known geometry, in [root, joint1, joint2, tip]
order, to build observations for a target pose.

Run with QuickIK's Python extension already built for this interpreter (see
docs/getting-started/installation.md):

    cd python && maturin develop --release
    pytest tests/
"""

import math
import time

import numpy as np
import pytest
import quickik

TWO_JOINT_CHAIN_JSON = """
{
    "joints": [
        {"name": "root", "parent": null, "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []},
        {"name": "joint1", "parent": "root", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": null}]},
        {"name": "joint2", "parent": "joint1", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": [-0.5, 0.5]}]},
        {"name": "tip", "parent": "joint2", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": []}
    ]
}
"""


@pytest.fixture
def tree():
    return quickik.KinematicTree.from_json_str(TWO_JOINT_CHAIN_JSON)


def two_link_positions(a1, a2):
    """Positions of [root, joint1, joint2, tip] when joint1/joint2 are at
    angles (a1, a2) about the shared Z axis -- see tests/common/mod.rs's doc
    comment for why joint1's own keypoint never moves with a1."""
    return [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0 + math.cos(a1), math.sin(a1), 0.0),
        (1.0 + math.cos(a1) + math.cos(a1 + a2), math.sin(a1) + math.sin(a1 + a2), 0.0),
    ]


def observations_for(a1, a2):
    return [
        quickik.KeypointObservation.position_3d(list(pos), 1.0)
        for pos in two_link_positions(a1, a2)
    ]


def no_prior_config():
    return quickik.SolverConfig(neutral_weight=0.0)


def test_malformed_json_raises():
    with pytest.raises(ValueError):
        quickik.KinematicTree.from_json_str("not valid json")


def test_recovers_pose_from_3d_observations(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, no_prior_config())
    solver.solve(state, observations_for(0.4, 0.3))

    assert state.dof_angles[0] == pytest.approx(0.4, abs=1e-3)
    assert state.dof_angles[1] == pytest.approx(0.3, abs=1e-3)


def test_position2d_observation_on_mapperless_solver_raises(tree):
    state = quickik.State.neutral_pose(tree)
    observations = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints)]
    observations[1] = quickik.KeypointObservation.position_2d([1.0, 0.0], 1.0)

    solver = quickik.Solver(tree, quickik.SolverConfig())
    with pytest.raises(ValueError):
        solver.solve(state, observations)


def test_solve_rejects_wrong_observation_count(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, quickik.SolverConfig())

    too_few = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints - 1)]
    with pytest.raises(ValueError):
        solver.solve(state, too_few)

    too_many = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints + 1)]
    with pytest.raises(ValueError):
        solver.solve(state, too_many)


def test_solve_frame_rejects_wrong_observation_count(tree):
    solver = quickik.SequenceSolver(tree, quickik.SolverConfig())
    too_few = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints - 1)]
    with pytest.raises(ValueError):
        solver.solve_frame(too_few)


def test_recovers_pose_from_xyview_observations(tree):
    positions = two_link_positions(0.35, -0.25)
    observations = [
        quickik.KeypointObservation.position_2d([p[0], p[1]], 1.0) for p in positions
    ]

    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, no_prior_config(), mapper=quickik.XYView())
    solver.solve(state, observations)

    assert state.dof_angles[0] == pytest.approx(0.35, abs=1e-3)
    assert state.dof_angles[1] == pytest.approx(-0.25, abs=1e-3)


def test_recovers_pose_from_camera_observations(tree):
    positions = two_link_positions(0.2, 0.15)
    camera = quickik.Camera(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        world2cam_pos=[0.0, 0.0, 5.0],
        world2cam_rot_mat=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )

    observations = []
    for x, y, z in positions:
        # Pinhole projection with world2cam_rot_mat = identity: cam == world.
        cam_z = z + camera.world2cam_pos[2]
        u = camera.fx * x / cam_z + camera.cx
        v = camera.fy * y / cam_z + camera.cy
        observations.append(quickik.KeypointObservation.position_2d([u, v], 1.0))

    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, no_prior_config(), mapper=camera)
    solver.solve(state, observations)

    assert state.dof_angles[0] == pytest.approx(0.2, abs=1e-3)
    assert state.dof_angles[1] == pytest.approx(0.15, abs=1e-3)


def test_xyview_latency_not_much_worse_than_3d(tree):
    """Sanity check, not a benchmark (see benchmark/ for real numbers):
    XYView's per-keypoint sparse-accumulation path (solver.rs's Position2D
    branch) shouldn't be dramatically slower than the Position3D path it
    mirrors. A generous factor -- this only needs to catch a gross
    regression (e.g. an accidental per-call allocation creeping back in),
    not assert precise parity, since single-frame timing on this tiny
    fixture is dominated by Python/FFI call overhead common to both paths."""

    def mean_solve_seconds(observations, mapper=None):
        solver = quickik.Solver(tree, quickik.SolverConfig(), mapper)
        state = quickik.State.neutral_pose(tree)
        solver.solve(state, observations)  # warm up

        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            state = quickik.State.neutral_pose(tree)
            solver.solve(state, observations)
        return (time.perf_counter() - t0) / n

    positions = two_link_positions(0.4, 0.3)
    observations_3d = observations_for(0.4, 0.3)
    observations_2d = [
        quickik.KeypointObservation.position_2d([p[0], p[1]], 1.0) for p in positions
    ]

    t_3d = mean_solve_seconds(observations_3d)
    t_2d = mean_solve_seconds(observations_2d, mapper=quickik.XYView())

    assert t_2d < t_3d * 5


def test_missing_observations_leave_state_at_neutral_prior(tree):
    state = quickik.State.neutral_pose(tree)
    observations = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints)]

    solver = quickik.Solver(tree, quickik.SolverConfig())
    solver.solve(state, observations)

    for angle in state.dof_angles:
        assert abs(angle) < 1e-6


def test_config_can_be_tuned_between_solve_calls(tree):
    state = quickik.State.neutral_pose(tree)
    observations = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints)]

    solver = quickik.Solver(tree, quickik.SolverConfig())
    solver.solve(state, observations)

    solver.config.n_iterations = 3
    solver.solve(state, observations)

    assert solver.config.n_iterations == 3


def test_solve_respects_joint_limits(tree):
    state = quickik.State.neutral_pose(tree)
    # Same unreachable target as tests/solver_test.rs's
    # solve_respects_joint_limits: joint2 would need ~1.2 rad, past its 0.5 cap.
    observations = [
        quickik.KeypointObservation.missing(),
        quickik.KeypointObservation.position_3d([1.0, 0.0, 0.0], 1.0),
        quickik.KeypointObservation.position_3d([2.0, 0.0, 0.0], 1.0),
        quickik.KeypointObservation.position_3d([2.3624, 0.9320, 0.0], 1.0),
    ]

    solver = quickik.Solver(tree, quickik.SolverConfig())
    solver.solve(state, observations)

    joint2_angle = state.dof_angles[1]
    assert -0.5 - 1e-6 <= joint2_angle <= 0.5 + 1e-6
    assert joint2_angle > 0.45


def test_sequence_solver_warm_start_converges_faster(tree):
    config = no_prior_config()
    config.n_iterations = 1
    target = observations_for(0.4, 0.3)

    cold = quickik.SequenceSolver(tree, config)
    cold.solve_frame(target)
    cold_error = abs(cold.state.dof_angles[0] - 0.4)

    warm = quickik.SequenceSolver(tree, config)
    warm.solve_frame(target)
    warm.solve_frame(target)
    warm_error = abs(warm.state.dof_angles[0] - 0.4)

    assert warm_error < cold_error


def test_solve_sequence_returns_one_state_per_frame(tree):
    solver = quickik.SequenceSolver(tree, quickik.SolverConfig())
    angles = [(0.1, 0.05), (0.2, 0.1), (0.3, 0.15)]
    positions = np.array(
        [two_link_positions(a1, a2) for a1, a2 in angles], dtype=np.float32
    )
    weights = np.ones((len(angles), tree.n_joints), dtype=np.float32)

    states = solver.solve_sequence(positions, weights)

    assert len(states) == 3
    last = states[2]
    assert last.dof_angles[0] == pytest.approx(0.3, abs=1e-2)
    assert last.dof_angles[1] == pytest.approx(0.15, abs=1e-2)


def test_solve_sequence_casts_float64_arrays_to_float32(tree):
    solver = quickik.SequenceSolver(tree, quickik.SolverConfig())
    angles = [(0.1, 0.05), (0.2, 0.1), (0.3, 0.15)]
    positions = np.array(
        [two_link_positions(a1, a2) for a1, a2 in angles], dtype=np.float64
    )
    weights = np.ones((len(angles), tree.n_joints), dtype=np.float64)

    states = solver.solve_sequence(positions, weights)

    assert len(states) == 3
    last = states[2]
    assert last.dof_angles[0] == pytest.approx(0.3, abs=1e-2)
    assert last.dof_angles[1] == pytest.approx(0.15, abs=1e-2)


def test_solve_sequence_rejects_wrong_keypoint_count(tree):
    positions = np.zeros((5, tree.n_joints - 1, 3), dtype=np.float32)
    weights = np.zeros((5, tree.n_joints - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="keypoints"):
        quickik.SequenceSolver(tree, quickik.SolverConfig()).solve_sequence(
            positions, weights
        )


def test_solve_sequence_treats_nan_weight_as_missing(tree):
    """Regression test: a NaN weight (a common "no confidence" sentinel from
    upstream pose-estimation pipelines) must be treated as missing, like a
    zero/negative weight, rather than poisoning the whole frame's solve (which
    would otherwise leave every DOF frozen at its prior every frame)."""
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)
    weights[:, 1] = np.nan  # joint1's own keypoint unobserved every frame

    states = quickik.SequenceSolver(tree, quickik.SolverConfig()).solve_sequence(
        positions, weights
    )

    # Losing one keypoint's worth of evidence in this whole-body joint solve
    # shifts the converged fit slightly system-wide (not just for the DOFs
    # that keypoint constrains), so this uses a looser tolerance than the
    # fully-observed tests above -- the point is to distinguish "converged
    # near the true trajectory" from "frozen at exactly zero" (what the NaN
    # bug caused), not to assert high precision.
    last, (a1, a2) = states[-1], true_angles[-1]
    assert last.dof_angles[0] == pytest.approx(a1, abs=5e-2)
    assert last.dof_angles[1] == pytest.approx(a2, abs=5e-2)


def sine_trajectory_2d_xyview_arrays(tree, n_frames):
    """2D (XYView-projected) counterpart to `sine_trajectory_arrays`: same
    sine trajectory, but `positions` drops each keypoint's Z coordinate,
    matching XYView's own (identity, Z-dropping) projection."""
    true_angles = []
    positions = np.zeros((n_frames, tree.n_joints, 2), dtype=np.float32)
    weights = np.ones((n_frames, tree.n_joints), dtype=np.float32)
    for t in range(n_frames):
        a = 0.3 * math.sin(t * 0.15)
        true_angles.append((a, a * 0.5))
        positions[t] = [(x, y) for x, y, _z in two_link_positions(a, a * 0.5)]
    return positions, weights, true_angles


def test_solve_sequence_xyview_reconstructs_trajectory(tree):
    positions, weights, true_angles = sine_trajectory_2d_xyview_arrays(
        tree, n_frames=10
    )

    solver = quickik.SequenceSolver(tree, no_prior_config(), mapper=quickik.XYView())
    states = solver.solve_sequence(positions, weights)

    assert len(states) == len(true_angles)
    last, (a1, a2) = states[-1], true_angles[-1]
    assert last.dof_angles[0] == pytest.approx(a1, abs=1e-2)
    assert last.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_sequence_rejects_3d_positions_when_mapper_set(tree):
    positions = np.zeros((3, tree.n_joints, 3), dtype=np.float32)
    weights = np.ones((3, tree.n_joints), dtype=np.float32)
    solver = quickik.SequenceSolver(
        tree, quickik.SolverConfig(), mapper=quickik.XYView()
    )
    with pytest.raises(ValueError, match="2"):
        solver.solve_sequence(positions, weights)


def test_solve_sequence_rejects_2d_positions_without_mapper(tree):
    positions = np.zeros((3, tree.n_joints, 2), dtype=np.float32)
    weights = np.ones((3, tree.n_joints), dtype=np.float32)
    solver = quickik.SequenceSolver(tree, quickik.SolverConfig())
    with pytest.raises(ValueError, match="3"):
        solver.solve_sequence(positions, weights)


def sine_trajectory_arrays(tree, n_frames):
    """positions/weights (all keypoints observed, weight 1.0) for a smooth
    sine trajectory of the two-joint chain, plus the true (a1, a2) angles
    used to generate it."""
    true_angles = []
    positions = np.zeros((n_frames, tree.n_joints, 3), dtype=np.float32)
    weights = np.ones((n_frames, tree.n_joints), dtype=np.float32)
    for t in range(n_frames):
        a = 0.3 * math.sin(t * 0.15)
        true_angles.append((a, a * 0.5))
        positions[t] = two_link_positions(a, a * 0.5)
    return positions, weights, true_angles


def test_solve_sequence_segmented_parallel_reconstructs_smooth_trajectory(tree):
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)

    parallel_config = quickik.ParallelSolveConfig(
        segment_len=10, overlap_len=3, overlap_tolerance=0.05, n_workers=-1
    )
    states = quickik.solve_sequence_segmented_parallel(
        tree, quickik.SolverConfig(), positions, weights, parallel_config
    )

    assert len(states) == len(true_angles)
    for state, (a1, a2) in zip(states, true_angles, strict=True):
        assert state.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert state.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_parallel_solve_config_for_recording_reconstructs_smooth_trajectory(tree):
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)

    parallel_config = quickik.ParallelSolveConfig.for_recording(len(true_angles))
    states = quickik.solve_sequence_segmented_parallel(
        tree, quickik.SolverConfig(), positions, weights, parallel_config
    )

    assert len(states) == len(true_angles)
    for state, (a1, a2) in zip(states, true_angles, strict=True):
        assert state.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert state.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_sequence_segmented_parallel_casts_float64_arrays_to_float32(tree):
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)
    positions = positions.astype(np.float64)
    weights = weights.astype(np.float64)

    parallel_config = quickik.ParallelSolveConfig(
        segment_len=10, overlap_len=3, overlap_tolerance=0.05, n_workers=-1
    )
    states = quickik.solve_sequence_segmented_parallel(
        tree, quickik.SolverConfig(), positions, weights, parallel_config
    )

    assert len(states) == len(true_angles)
    for state, (a1, a2) in zip(states, true_angles, strict=True):
        assert state.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert state.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_sequence_segmented_parallel_xyview_reconstructs_trajectory(tree):
    positions, weights, true_angles = sine_trajectory_2d_xyview_arrays(
        tree, n_frames=40
    )

    parallel_config = quickik.ParallelSolveConfig(
        segment_len=10, overlap_len=3, overlap_tolerance=0.05, n_workers=-1
    )
    states = quickik.solve_sequence_segmented_parallel(
        tree,
        quickik.SolverConfig(),
        positions,
        weights,
        parallel_config,
        mapper=quickik.XYView(),
    )

    assert len(states) == len(true_angles)
    for state, (a1, a2) in zip(states, true_angles, strict=True):
        assert state.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert state.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_sequence_segmented_parallel_rejects_wrong_keypoint_count(tree):
    positions = np.zeros((5, tree.n_joints - 1, 3), dtype=np.float32)
    weights = np.zeros((5, tree.n_joints - 1), dtype=np.float32)
    parallel_config = quickik.ParallelSolveConfig(
        segment_len=5, overlap_len=0, overlap_tolerance=0.05, n_workers=1
    )
    with pytest.raises(ValueError, match="keypoints"):
        quickik.solve_sequence_segmented_parallel(
            tree, quickik.SolverConfig(), positions, weights, parallel_config
        )


def test_solve_sequence_segmented_parallel_honors_explicit_n_workers(tree):
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)

    # n_workers=1 forces every segment through a single spawned thread,
    # exercising a different code path than the -1 (all available cores)
    # used above.
    parallel_config = quickik.ParallelSolveConfig(
        segment_len=10, overlap_len=3, overlap_tolerance=0.05, n_workers=1
    )
    states = quickik.solve_sequence_segmented_parallel(
        tree, quickik.SolverConfig(), positions, weights, parallel_config
    )

    assert len(states) == len(true_angles)
    for state, (a1, a2) in zip(states, true_angles, strict=True):
        assert state.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert state.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_invalid_parallel_solve_config_raises_value_error(tree):
    positions, weights, _ = sine_trajectory_arrays(tree, n_frames=5)

    # n_workers=0 is documented as invalid.
    zero_workers = quickik.ParallelSolveConfig(
        segment_len=5, overlap_len=0, overlap_tolerance=0.05, n_workers=0
    )
    with pytest.raises(ValueError):
        quickik.solve_sequence_segmented_parallel(
            tree, quickik.SolverConfig(), positions, weights, zero_workers
        )

    # overlap_len must be smaller than segment_len.
    bad_overlap = quickik.ParallelSolveConfig(
        segment_len=5, overlap_len=5, overlap_tolerance=0.05, n_workers=1
    )
    with pytest.raises(ValueError):
        quickik.solve_sequence_segmented_parallel(
            tree, quickik.SolverConfig(), positions, weights, bad_overlap
        )
