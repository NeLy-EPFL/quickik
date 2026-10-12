"""Tests for QuickIK's Python bindings. The solver itself is tested in Rust
(`tests/`); these check what the binding layer adds: argument conversion,
error translation, and that results round-trip correctly.

Most tests use a two-joint planar chain: a free root, `joint1` and `joint2`
(one Z-axis hinge each, `joint2` limited to [-0.5, 0.5]), and a `tip`.

Run with QuickIK's Python extension built for this interpreter (see
docs/getting-started/installation.md):

    cd python && maturin develop --release
    pytest tests/
"""

import copy
import math

import numpy as np
import pytest
import quickik

TWO_JOINT_CHAIN_JSON = """
{
    "joints": [
        {"name": "root", "parent": null, "offset_pos": [0, 0, 0], "offset_quat": [1, 0, 0, 0], "dofs": []},
        {"name": "joint1", "parent": "root", "offset_pos": [1, 0, 0], "offset_quat": [1, 0, 0, 0],
         "dofs": [{"axis": [0, 0, 1], "type": "hinge", "neutral": 0.0, "limits": null}]},
        {"name": "joint2", "parent": "joint1", "offset_pos": [1, 0, 0], "offset_quat": [1, 0, 0, 0],
         "dofs": [{"axis": [0, 0, 1], "type": "hinge", "neutral": 0.0, "limits": [-0.5, 0.5]}]},
        {"name": "tip", "parent": "joint2", "offset_pos": [1, 0, 0], "offset_quat": [1, 0, 0, 0], "dofs": []}
    ]
}
"""  # noqa: E501


def two_joint_chain(fixed_base=False):
    """The same tree as `TWO_JOINT_CHAIN_JSON`, built from Python objects."""
    hinge = quickik.Dof(quickik.DofType.HINGE, axis=(0, 0, 1))
    limited = quickik.Dof(axis=(0, 0, 1), limits=(-0.5, 0.5))
    joints = [
        quickik.Joint("root"),
        quickik.Joint("joint1", parent=0, offset_pos=(1, 0, 0), dofs=[hinge]),
        quickik.Joint("joint2", parent=1, offset_pos=(1, 0, 0), dofs=[limited]),
        quickik.Joint("tip", parent=2, offset_pos=(1, 0, 0)),
    ]
    return quickik.KinematicTree(joints, fixed_base=fixed_base)


@pytest.fixture
def tree():
    return quickik.KinematicTree.from_json_str(TWO_JOINT_CHAIN_JSON)


def chain_positions(a1, a2):
    """`(4, 3)` keypoint positions of the chain with DOF values `(a1, a2)`. A
    joint's own DOFs only move its descendants, so `joint1` stays put."""
    return np.array(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1 + math.cos(a1), math.sin(a1), 0),
            (1 + math.cos(a1) + math.cos(a1 + a2), math.sin(a1) + math.sin(a1 + a2), 0),
        ]
    )


def observations_3d(positions):
    return [
        quickik.KeypointObservation.position_3d(j, pos)
        for j, pos in enumerate(positions)
    ]


def exact_config():
    """No priors and no early stopping, so solves converge to an exact fit."""
    return quickik.SolverConfig(
        n_iterations=20, position_tolerance=0, angle_tolerance=0
    )


def camera(x_offset=0.0):
    """A pinhole camera 5 units above the chain, looking down."""
    rot = np.diag([1.0, -1.0, -1.0])
    return quickik.PinholeCamera(500, 500, 320, 240, (-x_offset, 0, 5), rot)


def sine_trajectory(n_frames):
    """`(n_frames, 4, 3)` positions and the `(n_frames, 2)` DOF values that
    generate them."""
    dof_values = np.array(
        [(0.3 * math.sin(0.15 * t), 0.15 * math.sin(0.15 * t)) for t in range(n_frames)]
    )
    return np.array([chain_positions(*values) for values in dof_values]), dof_values


# ===== Body plan ==============================================================


def test_tree_from_objects_matches_json(tree):
    built = two_joint_chain()
    assert built.joint_names == tree.joint_names == ["root", "joint1", "joint2", "tip"]
    assert (built.n_joints, built.n_dofs, built.state_dim) == (4, 2, 8)
    assert built.joints[2].dofs[0].limits == pytest.approx((-0.5, 0.5))
    assert tree.joint_index("joint2") == 2
    assert tree.joint_dof_start_indices == [0, 0, 1, 2]


def test_tree_errors_are_translated():
    with pytest.raises(ValueError):
        quickik.KinematicTree.from_json_str("not valid json")
    with pytest.raises(ValueError, match="root"):
        quickik.KinematicTree([quickik.Joint("a"), quickik.Joint("b")])
    with pytest.raises(OSError):
        quickik.KinematicTree.from_json_file("does/not/exist.json")
    with pytest.raises(KeyError):
        two_joint_chain().joint_index("nonexistent")


def test_fixed_base_tree_has_no_root_dofs():
    tree = two_joint_chain(fixed_base=True)
    assert tree.fixed_base
    assert (tree.n_root_dofs, tree.state_dim) == (0, 2)


# ===== State and forward kinematics ===========================================


def test_state_arrays_are_read_only_copies(tree):
    state = quickik.State.neutral_pose(tree)
    with pytest.raises(ValueError, match="read-only"):
        state.dof_values[0] = 1.0

    state.dof_values = [0.4, 0.3]
    state.root_pos = np.array([1, 2, 3], dtype=np.float64)
    state.root_quat = (2, 0, 0, 0)  # normalized when set
    assert state.dof_values == pytest.approx([0.4, 0.3])
    assert state.root_pos == pytest.approx([1, 2, 3])
    assert state.root_quat == pytest.approx([1, 0, 0, 0])

    copied = copy.copy(state)
    copied.dof_values = [0, 0]
    assert state.dof_values == pytest.approx([0.4, 0.3])

    with pytest.raises(ValueError, match="2 elements"):
        state.dof_values = [0.0]
    with pytest.raises(ValueError, match="3 elements"):
        state.root_pos = (0, 0)


def test_forward_kinematics(tree):
    state = quickik.State.neutral_pose(tree)
    state.dof_values = [0.4, 0.3]
    workspace = quickik.ForwardKinematicsWorkspace(tree)
    quickik.forward_kinematics(workspace, state)

    assert workspace.keypoint_positions == pytest.approx(chain_positions(0.4, 0.3))
    assert workspace.jacobian().shape == (3 * tree.n_joints, tree.state_dim)


# ===== Projections and observations ===========================================


def test_projections():
    ortho = quickik.OrthographicViewXY(scale=2, offset=(10, 20))
    uv, d_uv_d_pos = ortho.project((1, 2, 3))
    assert uv == pytest.approx([12, 24])
    assert d_uv_d_pos == pytest.approx(np.array([[2, 0, 0], [0, 2, 0]]))

    cam = camera()
    assert isinstance(cam, quickik.Projection)
    assert (cam.fx, cam.cy) == (500, 240)
    assert cam.world2cam_rot_mat == pytest.approx(np.diag([1, -1, -1]))
    assert cam.project((1, 1, 0))[0] == pytest.approx([420, 140])

    with pytest.raises(TypeError):
        quickik.Projection()
    with pytest.raises(ValueError, match="world2cam_rot_mat"):
        quickik.PinholeCamera(500, 500, 320, 240, (0, 0, 5), np.eye(2))


def test_observation_properties():
    cam = camera()
    observation = quickik.KeypointObservation.position_2d(3, (1, 2), cam, weight=0.5)
    assert observation.joint_idx == 3
    assert observation.pos == pytest.approx([1, 2])
    assert observation.weight == 0.5
    assert isinstance(observation.projection, quickik.PinholeCamera)
    assert observation.projection.fx == cam.fx

    observation = quickik.KeypointObservation.position_3d(0, (1, 2, 3))
    assert observation.projection is None
    with pytest.raises(ValueError, match="3 elements"):
        quickik.KeypointObservation.position_3d(0, (1, 2))


def test_sequence_from_arrays_checks_shapes():
    pos_3d, weights_3d = np.zeros((5, 4, 3)), np.ones((5, 4))
    assert len(quickik.ObservationSequence.from_arrays(pos_3d, weights_3d)) == 5
    with pytest.raises(ValueError, match="weights_3d"):
        quickik.ObservationSequence.from_arrays(pos_3d, np.ones((5, 3)))
    with pytest.raises(ValueError, match="pos_2d"):
        quickik.ObservationSequence.from_arrays(
            projections=[camera()],
            pos_2d=np.zeros((5, 2, 4, 2)),
            weights_2d=np.ones((5, 2, 4)),
        )
    with pytest.raises(ValueError, match="together"):
        quickik.ObservationSequence.from_arrays(pos_3d)
    with pytest.raises(ValueError):
        quickik.ObservationSequence.from_arrays()


def test_sequence_from_flat_checks_frame_lengths():
    observations = observations_3d(chain_positions(0, 0))
    assert len(quickik.ObservationSequence.from_flat(observations, [1, 3])) == 2
    with pytest.raises(ValueError, match="frame_lengths"):
        quickik.ObservationSequence.from_flat(observations, [1, 2])


# ===== Solvers ================================================================


def test_single_frame_solver_from_3d(tree):
    solver = quickik.SingleFrameSolver(tree, exact_config())
    state = quickik.State.neutral_pose(tree)
    result = solver.solve(
        state, observations_3d(chain_positions(0.4, 0.3)), with_fk=True
    )

    assert result.state.dof_values == pytest.approx([0.4, 0.3], abs=1e-3)
    # Solves in place
    assert state.dof_values == pytest.approx(result.state.dof_values)
    assert result.keypoint_pos == pytest.approx(chain_positions(0.4, 0.3), abs=1e-3)
    assert result.jacobian is None and result.cholesky_l is None


def test_single_frame_solver_from_two_cameras(tree):
    """2D observations from two projections mixed in one frame."""
    positions = chain_positions(0.2, -0.3)
    cameras = [camera(-1), camera(1)]
    observations = [
        quickik.KeypointObservation.position_2d(j, cam.project(pos)[0], cam)
        for cam in cameras
        for j, pos in enumerate(positions)
    ]
    solver = quickik.SingleFrameSolver(tree, exact_config())
    state = quickik.State.neutral_pose(tree)
    result = solver.solve(state, observations, with_grad=True)

    assert result.state.dof_values == pytest.approx([0.2, -0.3], abs=1e-3)
    assert result.jacobian.shape == (2 * len(observations), tree.state_dim)
    assert result.cholesky_l.shape == (tree.state_dim, tree.state_dim)


def test_solver_errors_are_translated(tree):
    config = quickik.SolverConfig()
    config.add_dof_smoothness_prior(1.0)
    with pytest.raises(ValueError, match="smoothness"):
        quickik.SingleFrameSolver(tree, config)
    with pytest.raises(ValueError, match="already"):
        config.add_dof_smoothness_prior(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        quickik.SolverConfig().add_dof_neutral_prior(-1.0)

    solver = quickik.SingleFrameSolver(tree, quickik.SolverConfig())
    state = quickik.State.neutral_pose(tree)
    observation = quickik.KeypointObservation.position_3d(tree.n_joints, (0, 0, 0))
    with pytest.raises(ValueError, match="joint"):
        solver.solve(state, [observation])


def test_sequential_solver_step_solve_and_solve_parallel_agree(tree):
    positions, dof_values = sine_trajectory(40)
    config = quickik.SolverConfig()
    config.add_dof_smoothness_prior(1e-3)
    sequence = quickik.ObservationSequence.from_lists(
        [observations_3d(frame) for frame in positions]
    )

    solver = quickik.SequentialSolver(tree, config)
    solved = np.array([r.state.dof_values for r in solver.solve(sequence)])
    assert solver.state.dof_values == pytest.approx(solved[-1])

    solver.reset()
    stepped = [
        solver.step(observations_3d(frame)).state.dof_values for frame in positions
    ]
    assert np.array(stepped) == pytest.approx(solved)

    results = solver.solve_parallel(sequence, n_workers=4, n_overlap=0)
    assert np.array([r.state.dof_values for r in results]) == pytest.approx(
        dof_values, abs=1e-2
    )
    with pytest.raises(ValueError):
        solver.solve_parallel(sequence, n_workers=0)


def test_sequential_solver_initial_state(tree):
    initial_state = quickik.State.neutral_pose(tree)
    initial_state.root_pos = (0, 0, 7)
    solver = quickik.SequentialSolver(tree, quickik.SolverConfig(), initial_state)
    assert solver.state.root_pos == pytest.approx([0, 0, 7])


def test_from_arrays_matches_from_lists_and_skips_nan(tree):
    """`from_arrays` (with float64 input, cast to float32) gives the same
    result as `from_lists`, and a NaN weight counts as a missing keypoint."""
    positions, _ = sine_trajectory(10)
    weights = np.ones(positions.shape[:2])
    weights[:, 1] = np.nan
    positions[:, 1] = np.nan
    from_arrays = quickik.ObservationSequence.from_arrays(positions, weights)
    from_lists = quickik.ObservationSequence.from_lists(
        [
            [obs for obs in observations_3d(frame) if obs.joint_idx != 1]
            for frame in positions
        ]
    )

    config = quickik.SolverConfig()
    a = quickik.SequentialSolver(tree, config).solve(from_arrays)
    b = quickik.SequentialSolver(tree, config).solve(from_lists)
    a, b = (np.array([r.state.dof_values for r in results]) for results in (a, b))
    assert np.isfinite(a).all()
    assert a == pytest.approx(b, abs=1e-5)


def test_batched_solver_matches_single_frame_solves(tree):
    positions, dof_values = sine_trajectory(8)
    batch = quickik.ObservationSequence.from_arrays(positions, np.ones((8, 4)))
    solver = quickik.BatchedSolver(tree, exact_config(), n_workers=2)
    result = solver.solve(batch, with_grad=True, with_fk=True)

    assert result.dof_values == pytest.approx(dof_values, abs=1e-3)
    assert result.root_quat.shape == (8, 4)
    assert result.keypoint_pos == pytest.approx(positions, abs=1e-3)
    assert result.jacobian.shape == (8, 3 * tree.n_joints, tree.state_dim)
    assert result.cholesky_l.shape == (8, tree.state_dim, tree.state_dim)
    assert result.cholesky_valid.all()

    single_frame_solver = quickik.SingleFrameSolver(tree, exact_config())
    state = quickik.State.neutral_pose(tree)
    single = single_frame_solver.solve(
        state, observations_3d(positions[3]), with_grad=True
    )
    assert result.dof_values[3] == pytest.approx(single.state.dof_values)
    assert result.cholesky_l[3] == pytest.approx(single.cholesky_l)


def test_batched_solver_errors(tree):
    with pytest.raises(ValueError, match="free-floating"):
        quickik.BatchedSolver(two_joint_chain(fixed_base=True), quickik.SolverConfig())
    with pytest.raises(ValueError):
        quickik.BatchedSolver(tree, quickik.SolverConfig(), n_workers=0)

    # Items with different numbers of observations can be solved, but their
    # Jacobians can't be stacked.
    observations = observations_3d(chain_positions(0, 0))
    batch = quickik.ObservationSequence.from_lists([observations, observations[:2]])
    solver = quickik.BatchedSolver(tree, quickik.SolverConfig())
    assert solver.solve(batch).dof_values.shape == (2, 2)
    with pytest.raises(ValueError, match="stack"):
        solver.solve(batch, with_grad=True)
