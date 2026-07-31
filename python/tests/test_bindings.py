"""Tests for QuickIK's Python bindings, mirroring cpp/tests/test_main.cpp and
tests/solver_test.rs / tests/sequential_solver_test.rs / tests/batched_solver_test.rs.
Uses the same "two-joint chain" fixture as those (see tests/common/mod.rs): a
root, joint1 and joint2 (each with one Z-axis DOF, joint2 limited to
[-0.5, 0.5]), and a trailing fixed tip.

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


FIXED_BASE_TWO_JOINT_CHAIN_JSON = """
{
    "fixed_base": true,
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


@pytest.fixture
def fixed_base_tree():
    return quickik.KinematicTree.from_json_str(FIXED_BASE_TWO_JOINT_CHAIN_JSON)


def two_link_positions(a1, a2):
    """Positions of [root, joint1, joint2, tip] when joint1/joint2 are at
    angles (a1, a2) about the shared Z axis. See tests/common/mod.rs's doc
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


def test_malformed_json_raises():
    with pytest.raises(ValueError):
        quickik.KinematicTree.from_json_str("not valid json")


def test_recovers_pose_from_3d_observations(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, neutral_weight=0.0)
    result = solver.solve(state, observations_for(0.4, 0.3))

    assert result.dof_angles[0] == pytest.approx(0.4, abs=1e-3)
    assert result.dof_angles[1] == pytest.approx(0.3, abs=1e-3)


def test_solve_with_fk_reports_keypoint_positions_matching_recovered_pose(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, neutral_weight=0.0)
    result = solver.solve(state, observations_for(0.4, 0.3), with_fk=True)

    expected = two_link_positions(0.4, 0.3)
    assert result.keypoint_pos.shape == (tree.n_joints, 3)
    for a, e in zip(result.keypoint_pos, expected, strict=True):
        assert a == pytest.approx(e, abs=1e-2)


def test_solve_without_with_fk_or_with_grad_leaves_optional_fields_none(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree)
    result = solver.solve(state, observations_for(0.4, 0.3))

    assert result.keypoint_pos is None
    assert result.jacobian is None
    assert result.cholesky_l is None


def test_result_state_matches_the_flat_dof_angles_root_pos_root_rot_properties(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, neutral_weight=0.0)
    result = solver.solve(state, observations_for(0.4, 0.3))

    assert result.state.dof_angles == pytest.approx(result.dof_angles)
    assert result.state.root_pos == pytest.approx(result.root_pos)
    assert result.state.root_rot == pytest.approx(result.root_rot)


def test_result_state_can_be_fed_into_another_solve_call(tree):
    """`result.state` should be a real, independent `State`, usable to
    warm-start a follow-up `Solver.solve` call, same as any other `State`."""
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, n_iterations=1, neutral_weight=0.0)
    target = observations_for(0.4, 0.3)

    cold_result = solver.solve(state, target)
    cold_error = abs(cold_result.dof_angles[0] - 0.4)

    warm_state = cold_result.state
    warm_result = solver.solve(warm_state, target)
    warm_error = abs(warm_result.dof_angles[0] - 0.4)

    assert warm_error < cold_error


def test_solve_with_grad_reports_jacobian_and_cholesky_l(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, neutral_weight=0.0)
    result = solver.solve(state, observations_for(0.4, 0.3), with_grad=True)

    n_joints = tree.n_joints
    state_dim = tree.n_dofs + 6
    assert result.jacobian.shape == (3 * n_joints, state_dim)
    assert result.cholesky_l.shape == (state_dim, state_dim)


def test_position2d_observation_on_mapperless_solver_raises(tree):
    state = quickik.State.neutral_pose(tree)
    observations = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints)]
    observations[1] = quickik.KeypointObservation.position_2d([1.0, 0.0], 1.0)

    solver = quickik.Solver(tree)
    with pytest.raises(ValueError):
        solver.solve(state, observations)


def test_solve_rejects_wrong_observation_count(tree):
    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree)

    too_few = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints - 1)]
    with pytest.raises(ValueError):
        solver.solve(state, too_few)

    too_many = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints + 1)]
    with pytest.raises(ValueError):
        solver.solve(state, too_many)


def test_recovers_pose_from_xyview_observations(tree):
    positions = two_link_positions(0.35, -0.25)
    observations = [
        quickik.KeypointObservation.position_2d([p[0], p[1]], 1.0) for p in positions
    ]

    state = quickik.State.neutral_pose(tree)
    solver = quickik.Solver(tree, mapper=quickik.XYView(), neutral_weight=0.0)
    result = solver.solve(state, observations)

    assert result.dof_angles[0] == pytest.approx(0.35, abs=1e-3)
    assert result.dof_angles[1] == pytest.approx(-0.25, abs=1e-3)


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
    solver = quickik.Solver(tree, mapper=camera, neutral_weight=0.0)
    result = solver.solve(state, observations)

    assert result.dof_angles[0] == pytest.approx(0.2, abs=1e-3)
    assert result.dof_angles[1] == pytest.approx(0.15, abs=1e-3)


def test_xyview_latency_not_much_worse_than_3d(tree):
    """Sanity check, not a benchmark (see benchmark/ for real numbers):
    XYView's per-keypoint sparse-accumulation path (solver.rs's Position2D
    branch) shouldn't be dramatically slower than the Position3D path it
    mirrors. A generous factor: this only needs to catch a gross
    regression (e.g. an accidental per-call allocation creeping back in),
    not assert precise parity, since single-frame timing on this tiny
    fixture is dominated by Python/FFI call overhead common to both paths."""

    def mean_solve_seconds(observations, mapper=None):
        solver = quickik.Solver(tree, mapper=mapper)
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

    solver = quickik.Solver(tree)
    result = solver.solve(state, observations)

    for angle in result.dof_angles:
        assert abs(angle) < 1e-6


def test_solver_fields_can_be_tuned_between_solve_calls(tree):
    state = quickik.State.neutral_pose(tree)
    observations = [quickik.KeypointObservation.missing() for _ in range(tree.n_joints)]

    solver = quickik.Solver(tree)
    solver.solve(state, observations)

    solver.n_iterations = 3
    solver.solve(state, observations)

    assert solver.n_iterations == 3


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

    solver = quickik.Solver(tree)
    result = solver.solve(state, observations)

    joint2_angle = result.dof_angles[1]
    assert -0.5 - 1e-6 <= joint2_angle <= 0.5 + 1e-6
    assert joint2_angle > 0.45


def test_sequence_solver_warm_starts_across_separate_calls(tree):
    target = np.array([two_link_positions(0.4, 0.3)], dtype=np.float32)
    weights = np.ones((1, tree.n_joints), dtype=np.float32)

    cold = quickik.SequenceSolver(tree, n_iterations=1, neutral_weight=0.0)
    cold_result = cold.solve(target, weights)[0]
    cold_error = abs(cold_result.dof_angles[0] - 0.4)

    warm = quickik.SequenceSolver(tree, n_iterations=1, neutral_weight=0.0)
    warm.solve(target, weights)
    warm_result = warm.solve(target, weights)[0]
    warm_error = abs(warm_result.dof_angles[0] - 0.4)

    assert warm_error < cold_error


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


def test_sequence_solver_solve_returns_one_result_per_frame(tree):
    solver = quickik.SequenceSolver(tree)
    angles = [(0.1, 0.05), (0.2, 0.1), (0.3, 0.15)]
    positions = np.array(
        [two_link_positions(a1, a2) for a1, a2 in angles], dtype=np.float32
    )
    weights = np.ones((len(angles), tree.n_joints), dtype=np.float32)

    results = solver.solve(positions, weights)

    assert len(results) == 3
    last = results[2]
    assert last.dof_angles[0] == pytest.approx(0.3, abs=1e-2)
    assert last.dof_angles[1] == pytest.approx(0.15, abs=1e-2)


def test_sequence_solver_solve_with_fk_matches_recovered_pose(tree):
    target = np.array([two_link_positions(0.4, 0.3)], dtype=np.float32)
    weights = np.ones((1, tree.n_joints), dtype=np.float32)

    solver = quickik.SequenceSolver(tree, neutral_weight=0.0)
    result = solver.solve(target, weights, with_fk=True)[0]

    expected = two_link_positions(0.4, 0.3)
    assert result.keypoint_pos.shape == (tree.n_joints, 3)
    for a, e in zip(result.keypoint_pos, expected, strict=True):
        assert a == pytest.approx(e, abs=1e-2)


def test_sequence_solver_casts_float64_arrays_to_float32(tree):
    solver = quickik.SequenceSolver(tree)
    angles = [(0.1, 0.05), (0.2, 0.1), (0.3, 0.15)]
    positions = np.array(
        [two_link_positions(a1, a2) for a1, a2 in angles], dtype=np.float64
    )
    weights = np.ones((len(angles), tree.n_joints), dtype=np.float64)

    results = solver.solve(positions, weights)

    assert len(results) == 3
    last = results[2]
    assert last.dof_angles[0] == pytest.approx(0.3, abs=1e-2)
    assert last.dof_angles[1] == pytest.approx(0.15, abs=1e-2)


def test_sequence_solver_solve_rejects_wrong_keypoint_count(tree):
    positions = np.zeros((5, tree.n_joints - 1, 3), dtype=np.float32)
    weights = np.zeros((5, tree.n_joints - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="keypoints"):
        quickik.SequenceSolver(tree).solve(positions, weights)


def test_sequence_solver_treats_nan_weight_as_missing(tree):
    """Regression test: a NaN weight (a common "no confidence" sentinel from
    upstream pose-estimation pipelines) must be treated as missing, like a
    zero/negative weight, rather than poisoning the whole frame's solve (which
    would otherwise leave every DOF frozen at its prior every frame)."""
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)
    weights[:, 1] = np.nan  # joint1's own keypoint unobserved every frame

    results = quickik.SequenceSolver(tree).solve(positions, weights)

    # Losing one keypoint's worth of evidence in this whole-body joint solve
    # shifts the converged fit slightly system-wide (not just for the DOFs
    # that keypoint constrains), so this uses a looser tolerance than the
    # fully-observed tests above. The point is to distinguish "converged
    # near the true trajectory" from "frozen at exactly zero" (what the NaN
    # bug caused), not to assert high precision.
    last, (a1, a2) = results[-1], true_angles[-1]
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


def test_sequence_solver_xyview_reconstructs_trajectory(tree):
    positions, weights, true_angles = sine_trajectory_2d_xyview_arrays(
        tree, n_frames=10
    )

    solver = quickik.SequenceSolver(tree, mapper=quickik.XYView(), neutral_weight=0.0)
    results = solver.solve(positions, weights)

    assert len(results) == len(true_angles)
    last, (a1, a2) = results[-1], true_angles[-1]
    assert last.dof_angles[0] == pytest.approx(a1, abs=1e-2)
    assert last.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_sequence_solver_solve_rejects_3d_positions_when_mapper_set(tree):
    positions = np.zeros((3, tree.n_joints, 3), dtype=np.float32)
    weights = np.ones((3, tree.n_joints), dtype=np.float32)
    solver = quickik.SequenceSolver(tree, mapper=quickik.XYView())
    with pytest.raises(ValueError, match="2"):
        solver.solve(positions, weights)


def test_sequence_solver_solve_rejects_2d_positions_without_mapper(tree):
    positions = np.zeros((3, tree.n_joints, 2), dtype=np.float32)
    weights = np.ones((3, tree.n_joints), dtype=np.float32)
    solver = quickik.SequenceSolver(tree)
    with pytest.raises(ValueError, match="3"):
        solver.solve(positions, weights)


def test_solve_segments_parallel_reconstructs_smooth_trajectory(tree):
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)

    solver = quickik.SequenceSolver(tree)
    results = solver.solve_segments_parallel(positions, weights, n_workers=4)

    assert len(results) == len(true_angles)
    for result, (a1, a2) in zip(results, true_angles, strict=True):
        assert result.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert result.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_segments_parallel_casts_float64_arrays_to_float32(tree):
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)
    positions = positions.astype(np.float64)
    weights = weights.astype(np.float64)

    solver = quickik.SequenceSolver(tree)
    results = solver.solve_segments_parallel(positions, weights, n_workers=4)

    assert len(results) == len(true_angles)
    for result, (a1, a2) in zip(results, true_angles, strict=True):
        assert result.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert result.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_segments_parallel_xyview_reconstructs_trajectory(tree):
    positions, weights, true_angles = sine_trajectory_2d_xyview_arrays(
        tree, n_frames=40
    )

    solver = quickik.SequenceSolver(tree, mapper=quickik.XYView())
    results = solver.solve_segments_parallel(positions, weights, n_workers=4)

    assert len(results) == len(true_angles)
    for result, (a1, a2) in zip(results, true_angles, strict=True):
        assert result.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert result.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_segments_parallel_rejects_wrong_keypoint_count(tree):
    positions = np.zeros((5, tree.n_joints - 1, 3), dtype=np.float32)
    weights = np.zeros((5, tree.n_joints - 1), dtype=np.float32)
    solver = quickik.SequenceSolver(tree)
    with pytest.raises(ValueError, match="keypoints"):
        solver.solve_segments_parallel(positions, weights, n_workers=1)


def test_solve_segments_parallel_honors_explicit_n_workers(tree):
    positions, weights, true_angles = sine_trajectory_arrays(tree, n_frames=40)

    # n_workers=1 forces the whole sequence through a single segment,
    # exercising a different code path than the >1 case used above.
    solver = quickik.SequenceSolver(tree)
    results = solver.solve_segments_parallel(positions, weights, n_workers=1)

    assert len(results) == len(true_angles)
    for result, (a1, a2) in zip(results, true_angles, strict=True):
        assert result.dof_angles[0] == pytest.approx(a1, abs=1e-2)
        assert result.dof_angles[1] == pytest.approx(a2, abs=1e-2)


def test_solve_segments_parallel_rejects_zero_workers(tree):
    positions, weights, _ = sine_trajectory_arrays(tree, n_frames=5)
    solver = quickik.SequenceSolver(tree)
    with pytest.raises(ValueError):
        solver.solve_segments_parallel(positions, weights, n_workers=0)


def joint_names(names):
    return list(names)


def test_batched_solver_matches_sequential_solve(tree):
    # A permutation of the tree's own joint order, so this actually exercises
    # name-based remapping rather than happening to pass only for the
    # identity order.
    keypoints_order = joint_names(["tip", "root", "joint2", "joint1"])
    order_joint_indices = [3, 0, 2, 1]
    targets = [(0.4, 0.3), (-0.2, 0.1), (0.3, -0.4), (0.15, 0.25)]

    expected_dof_angles = []
    for angles in targets:
        state = quickik.State.neutral_pose(tree)
        solver = quickik.Solver(tree, neutral_weight=0.0)
        result = solver.solve(state, observations_for(*angles))
        expected_dof_angles.append(result.dof_angles)

    positions = np.array(
        [
            [two_link_positions(*angles)[i] for i in order_joint_indices]
            for angles in targets
        ],
        dtype=np.float32,
    )
    weights = np.ones((len(targets), tree.n_joints), dtype=np.float32)

    batched_solver = quickik.BatchedSolver(tree, keypoints_order, neutral_weight=0.0)
    result = batched_solver.solve(positions, weights)

    assert result.joint_angles.shape == (len(targets), tree.n_dofs)
    for i in range(len(targets)):
        assert result.joint_angles[i] == pytest.approx(expected_dof_angles[i], abs=1e-4)


def test_batched_solver_with_grad_reports_jacobian_and_valid(tree):
    keypoints_order = joint_names(["root", "joint1", "joint2", "tip"])
    positions = np.array([two_link_positions(0.4, 0.3)], dtype=np.float32)
    weights = np.ones((1, tree.n_joints), dtype=np.float32)

    solver = quickik.BatchedSolver(tree, keypoints_order, neutral_weight=0.0)
    result = solver.solve(positions, weights, with_grad=True)

    n_joints = tree.n_joints
    state_dim = tree.n_dofs + 6
    assert result.valid[0]
    assert result.jacobian.shape == (1, 3 * n_joints, state_dim)
    assert result.cholesky_l.shape == (1, state_dim, state_dim)


def test_batched_solver_without_with_grad_or_with_fk_leaves_optional_fields_none(tree):
    keypoints_order = joint_names(["root", "joint1", "joint2", "tip"])
    positions = np.array(
        [two_link_positions(0.4, 0.3), two_link_positions(-0.1, 0.2)], dtype=np.float32
    )
    weights = np.ones((2, tree.n_joints), dtype=np.float32)

    solver = quickik.BatchedSolver(tree, keypoints_order)
    result = solver.solve(positions, weights)

    assert result.joint_angles.shape == (2, tree.n_dofs)
    assert result.keypoint_pos is None
    assert result.jacobian is None
    assert result.cholesky_l is None
    assert result.valid is None


def test_batched_solver_with_fk_reports_keypoint_positions(tree):
    keypoints_order = joint_names(["root", "joint1", "joint2", "tip"])
    positions = np.array([two_link_positions(0.4, 0.3)], dtype=np.float32)
    weights = np.ones((1, tree.n_joints), dtype=np.float32)

    solver = quickik.BatchedSolver(tree, keypoints_order)
    result = solver.solve(positions, weights, with_fk=True)

    expected = two_link_positions(0.4, 0.3)
    assert result.keypoint_pos.shape == (1, tree.n_joints, 3)
    for a, e in zip(result.keypoint_pos[0], expected, strict=True):
        assert a == pytest.approx(e, abs=1e-2)


def test_batched_solver_keypoint_to_joint_idx_matches_keypoints_order(tree):
    keypoints_order = joint_names(["tip", "root", "joint2", "joint1"])
    solver = quickik.BatchedSolver(tree, keypoints_order)
    assert solver.keypoint_to_joint_idx == [3, 0, 2, 1]


def test_batched_solver_rejects_unknown_joint_name(tree):
    with pytest.raises(ValueError):
        quickik.BatchedSolver(
            tree, joint_names(["root", "joint1", "joint2", "nonexistent"])
        )


def test_batched_solver_rejects_duplicate_joint_name(tree):
    with pytest.raises(ValueError):
        quickik.BatchedSolver(tree, joint_names(["root", "joint1", "joint1", "tip"]))


def test_batched_solver_rejects_fixed_base_tree(fixed_base_tree):
    with pytest.raises(ValueError):
        quickik.BatchedSolver(
            fixed_base_tree, joint_names(["root", "joint1", "joint2", "tip"])
        )


def test_batched_solver_n_workers_one_matches_default(tree):
    keypoints_order = joint_names(["root", "joint1", "joint2", "tip"])
    targets = [(0.4, 0.3), (-0.2, 0.1), (0.3, -0.4), (0.15, 0.25)]
    positions = np.array([two_link_positions(*a) for a in targets], dtype=np.float32)
    weights = np.ones((len(targets), tree.n_joints), dtype=np.float32)

    single_worker = quickik.BatchedSolver(
        tree, keypoints_order, neutral_weight=0.0, n_workers=1
    )
    default_workers = quickik.BatchedSolver(tree, keypoints_order, neutral_weight=0.0)

    single_result = single_worker.solve(positions, weights)
    default_result = default_workers.solve(positions, weights)

    assert single_result.joint_angles == pytest.approx(default_result.joint_angles)
    assert single_result.base_pos == pytest.approx(default_result.base_pos)
    assert single_result.base_quat == pytest.approx(default_result.base_quat)


def test_batched_solver_rejects_zero_workers(tree):
    keypoints_order = joint_names(["root", "joint1", "joint2", "tip"])
    with pytest.raises(ValueError):
        quickik.BatchedSolver(tree, keypoints_order, n_workers=0)
