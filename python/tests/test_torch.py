"""Tests for `quickik.torch` (the PyTorch `SolveIK` autograd.Function and its
`QuickIKSolve` nn.Module wrapper). Uses the same "two-joint chain" fixture as
tests/test_bindings.py (see that file's module docstring), but with a
deliberately *permuted* `keypoints_order` throughout: SolveIK's job is
exactly the internal-tree-order <-> caller-order remapping, so testing only
the identity order would miss bugs in that remapping.

Skips entirely if PyTorch isn't installed (it's an optional dependency; see
pyproject.toml's `[torch]` extra).

Run with QuickIK's Python extension already built for this interpreter (see
docs/getting-started/installation.md):

    cd python && maturin develop --release
    pytest tests/
"""

import math
import subprocess
import sys

import pytest
import quickik

torch = pytest.importorskip("torch")
import quickik.torch as qtorch

# gradcheck needs double precision to distinguish a real analytical-gradient
# bug from finite-difference roundoff.
torch.set_default_dtype(torch.float64)

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

# A joint has a non-1.0 weight_scaler, which SolveIK's gradient must fold in
# (see torch.py's `joint_weight_scalers` usage) -- the plain two-joint chain
# above (all scalers 1.0) wouldn't exercise that at all.
WEIGHTED_TWO_JOINT_CHAIN_JSON = """
{
    "joints": [
        {"name": "root", "parent": null, "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": [], "weight_scaler": 1.0},
        {"name": "joint1", "parent": "root", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": null}], "weight_scaler": 2.5},
        {"name": "joint2", "parent": "joint1", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0],
         "dofs": [{"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": [-0.5, 0.5]}], "weight_scaler": 1.0},
        {"name": "tip", "parent": "joint2", "offset_pos": [1.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "dofs": [], "weight_scaler": 0.3}
    ]
}
"""

# keypoints_order: a permutation of the tree's own joint order ("root",
# "joint1", "joint2", "tip"). ORDER_JOINT_INDICES[i] is the internal joint
# index that KEYPOINTS_ORDER[i] corresponds to.
KEYPOINTS_ORDER = ["joint2", "tip", "root", "joint1"]
ORDER_JOINT_INDICES = [2, 3, 0, 1]


@pytest.fixture
def tree():
    return quickik.KinematicTree.from_json_str(TWO_JOINT_CHAIN_JSON)


@pytest.fixture
def weighted_tree():
    return quickik.KinematicTree.from_json_str(WEIGHTED_TWO_JOINT_CHAIN_JSON)


def two_link_positions(a1, a2):
    """Positions of [root, joint1, joint2, tip] when joint1/joint2 are at
    angles (a1, a2) about the shared Z axis -- see
    tests/test_bindings.py's `two_link_positions` (same geometry)."""
    return [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0 + math.cos(a1), math.sin(a1), 0.0),
        (1.0 + math.cos(a1) + math.cos(a1 + a2), math.sin(a1) + math.sin(a1 + a2), 0.0),
    ]


def positions_in_keypoints_order(angles):
    """`(len(angles), 4, 3)` tensor of `two_link_positions`, permuted into
    KEYPOINTS_ORDER (rather than the tree's internal joint order)."""
    internal = torch.tensor(
        [two_link_positions(a1, a2) for a1, a2 in angles], dtype=torch.float64
    )
    return internal[:, ORDER_JOINT_INDICES, :].clone()


def positions_2d_in_keypoints_order(angles):
    """XYView counterpart to `positions_in_keypoints_order`: same positions,
    but with each keypoint's Z coordinate dropped, matching XYView's own
    (identity, Z-dropping) projection."""
    return positions_in_keypoints_order(angles)[..., :2].clone()


def batched_solver(tree, mapper=None, **kwargs):
    kwargs.setdefault("n_iterations", 20)
    kwargs.setdefault("neutral_weight", 1e-3)
    kwargs.setdefault("damping", 1e-6)
    kwargs.setdefault("position_tolerance", 0.0)
    kwargs.setdefault("angle_tolerance", 0.0)
    return quickik.BatchedSolver(tree, KEYPOINTS_ORDER, mapper=mapper, **kwargs)


def test_joint_names_and_weight_scalers(weighted_tree):
    assert weighted_tree.joint_names == ["root", "joint1", "joint2", "tip"]
    assert weighted_tree.joint_weight_scalers == pytest.approx([1.0, 2.5, 1.0, 0.3])


def test_batched_solver_rejects_camera_mapper(tree):
    camera = quickik.Camera(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        world2cam_pos=[0.0, 0.0, 5.0],
        world2cam_rot_mat=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    solver = batched_solver(tree, mapper=camera)
    positions = torch.zeros(1, 4, 2)
    weights = torch.ones(1, 4)
    with pytest.raises(ValueError, match="Camera"):
        qtorch.SolveIK.apply(solver, positions, weights)


def test_forward_recovers_pose_in_keypoints_order(tree):
    """SolveIK's forward pass should recover the same joint angles as plain
    `quickik.Solver.solve` given an equivalent, reachable target -- this
    catches the keypoints_order remapping being wrong in a way that gradcheck
    (which only checks *sensitivity*, not absolute correctness) wouldn't."""
    solver = batched_solver(tree, neutral_weight=0.0)
    positions = positions_in_keypoints_order([(0.4, 0.3)])
    weights = torch.ones(1, 4, dtype=torch.float64)

    joint_angles, base_pos, base_quat = qtorch.SolveIK.apply(solver, positions, weights)

    assert joint_angles[0, 0].item() == pytest.approx(0.4, abs=1e-2)
    assert joint_angles[0, 1].item() == pytest.approx(0.3, abs=1e-2)
    assert base_pos[0].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-2)
    assert base_quat[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-2)


def test_gradcheck_positions(tree):
    solver = batched_solver(tree, neutral_weight=0.0)
    angles = [(0.3, 0.2), (0.35, 0.15), (0.25, 0.25)]
    positions = positions_in_keypoints_order(angles).clone().requires_grad_(True)
    weights = torch.ones(len(angles), 4, dtype=torch.float64)

    def func(positions):
        return qtorch.SolveIK.apply(solver, positions, weights)

    # eps/atol/rtol looser than gradcheck's defaults: quickik_core computes
    # entirely in f32 regardless of this test's float64 tensors, so an eps
    # much below f32's own precision floor would just measure quantization
    # noise, not a real gradient mismatch.
    assert torch.autograd.gradcheck(func, (positions,), eps=1e-3, atol=2e-2, rtol=2e-2)


def test_gradcheck_positions_with_nonuniform_weights_and_weight_scaler(weighted_tree):
    """Same as test_gradcheck_positions, but with a non-1.0 joint
    weight_scaler and non-uniform per-keypoint weights, both of which the
    plain identity-weight_scaler case above doesn't exercise."""
    solver = batched_solver(weighted_tree, neutral_weight=0.0)
    angles = [(0.3, 0.2), (0.35, 0.15)]
    positions = positions_in_keypoints_order(angles).clone().requires_grad_(True)
    weights = torch.ones(len(angles), 4, dtype=torch.float64)
    weights[:, 0] = 0.7
    weights[:, 2] = 1.3

    def func(positions):
        return qtorch.SolveIK.apply(solver, positions, weights)

    assert torch.autograd.gradcheck(func, (positions,), eps=1e-3, atol=2e-2, rtol=2e-2)


def test_gradcheck_positions_xyview(tree):
    """Same as test_gradcheck_positions, but through the XYView mapper:
    regression coverage for slicing the raw 3D Jacobian down to its first
    two rows in `SolveIK.backward` (see `ctx.n_obs_dims`)."""
    solver = batched_solver(tree, mapper=quickik.XYView(), neutral_weight=0.0)
    angles = [(0.3, 0.2), (0.35, 0.15), (0.25, 0.25)]
    positions = positions_2d_in_keypoints_order(angles).clone().requires_grad_(True)
    weights = torch.ones(len(angles), 4, dtype=torch.float64)

    def func(positions):
        return qtorch.SolveIK.apply(solver, positions, weights)

    assert torch.autograd.gradcheck(func, (positions,), eps=1e-3, atol=2e-2, rtol=2e-2)


def test_invalid_item_gets_zero_gradient_not_nan(tree):
    """An item with no observed keypoints has no positive-definite
    linearization to differentiate through; its gradient should be exactly
    zero (not NaN/Inf), while a well-posed item in the same batch still gets
    a real gradient."""
    solver = batched_solver(
        tree, neutral_weight=0.0, position_tolerance=1e-3, angle_tolerance=1e-3
    )
    positions = torch.zeros(2, 4, 3, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        positions[1] = positions_in_keypoints_order([(0.3, 0.2)])[0]
    weights = torch.zeros(2, 4, dtype=torch.float64)
    weights[1] = 1.0  # only item 1 is observed

    joint_angles, base_pos, base_quat = qtorch.SolveIK.apply(solver, positions, weights)
    (joint_angles.sum() + base_pos.sum() + base_quat.sum()).backward()

    assert torch.all(positions.grad[0] == 0)
    assert torch.isfinite(positions.grad).all()
    assert positions.grad[1].abs().sum().item() > 0


def test_weights_requires_grad_raises(tree):
    solver = batched_solver(tree)
    positions = positions_in_keypoints_order([(0.3, 0.2)])
    weights = torch.ones(1, 4, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="weights"):
        qtorch.SolveIK.apply(solver, positions, weights)


def test_quickiksolve_module_matches_solveik_directly(tree):
    solver = batched_solver(tree, neutral_weight=0.0)
    positions = positions_in_keypoints_order([(0.4, 0.3)])
    weights = torch.ones(1, 4, dtype=torch.float64)

    module = qtorch.QuickIKSolve(solver)
    from_module = module(positions, weights)
    from_function = qtorch.SolveIK.apply(solver, positions, weights)

    for a, b in zip(from_module, from_function, strict=True):
        assert torch.equal(a, b)


def test_quickiksolve_module_with_xyview_mapper(tree):
    solver = batched_solver(tree, mapper=quickik.XYView(), neutral_weight=0.0)
    positions = positions_2d_in_keypoints_order([(0.4, 0.3)])
    weights = torch.ones(1, 4, dtype=torch.float64)

    module = qtorch.QuickIKSolve(solver)
    joint_angles, _, _ = module(positions, weights)

    assert joint_angles[0, 0].item() == pytest.approx(0.4, abs=1e-2)
    assert joint_angles[0, 1].item() == pytest.approx(0.3, abs=1e-2)


def test_quickiksolve_rejects_camera_mapper_at_construction(tree):
    camera = quickik.Camera(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        world2cam_pos=[0.0, 0.0, 5.0],
        world2cam_rot_mat=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    solver = batched_solver(tree, mapper=camera)
    with pytest.raises(ValueError, match="Camera"):
        qtorch.QuickIKSolve(solver)


def test_torch_py_raises_informative_error_without_pytorch():
    """quickik.torch should fail with an actionable message if PyTorch isn't
    installed, not an opaque ModuleNotFoundError. Runs in a subprocess with
    `torch` hidden via sys.modules, so it doesn't disturb the real torch
    import every other test in this file relies on."""
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "try:\n"
        "    import quickik.torch\n"
        "except ImportError as e:\n"
        "    assert 'torch' in str(e).lower()\n"
        "    assert 'install' in str(e).lower()\n"
        "    print('OK')\n"
        "    sys.exit(0)\n"
        "print('FAIL: did not raise ImportError')\n"
        "sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
