"""Tests for `quickik.torch` (the PyTorch `SolveIK` autograd.Function and its
`QuickIKSolve` nn.Module wrapper). Uses the two-joint chain from
tests/test_bindings.py.

Skips entirely if PyTorch isn't installed (it's an optional dependency; see
pyproject.toml's `[torch]` extra).

Run with QuickIK's Python extension built for this interpreter (see
docs/getting-started/installation.md):

    cd python && maturin develop --release
    pytest tests/
"""

import math
import subprocess
import sys

import numpy as np
import pytest
import quickik

torch = pytest.importorskip("torch")
import quickik.torch as qtorch  # noqa: E402

# gradcheck needs double precision to distinguish a real analytical-gradient
# bug from finite-difference roundoff.
torch.set_default_dtype(torch.float64)


def chain(joint_weight_multipliers=(1, 1, 1, 1)):
    """The two-joint chain from tests/test_bindings.py, with the given
    per-joint `weight_multiplier`s."""
    hinge = quickik.Dof(axis=(0, 0, 1))
    limited = quickik.Dof(axis=(0, 0, 1), limits=(-0.5, 0.5))
    names, dofs = ["root", "joint1", "joint2", "tip"], [[], [hinge], [limited], []]
    joints = [
        quickik.Joint(
            name,
            parent=None if i == 0 else i - 1,
            offset_pos=(0, 0, 0) if i == 0 else (1, 0, 0),
            dofs=joint_dofs,
            weight_multiplier=multiplier,
        )
        for i, (name, joint_dofs, multiplier) in enumerate(
            zip(names, dofs, joint_weight_multipliers, strict=True)
        )
    ]
    return quickik.KinematicTree(joints)


def batched_solver(tree, neutral_weight=0.0, **kwargs):
    kwargs = {
        "n_iterations": 20,
        "position_tolerance": 0,
        "angle_tolerance": 0,
    } | kwargs
    config = quickik.SolverConfig(**kwargs)
    config.add_dof_neutral_prior(neutral_weight)
    return quickik.BatchedSolver(tree, config)


def chain_positions(angles):
    """`(len(angles), 4, 3)` keypoint positions of the chain with DOF values
    `angles[i]` in item `i`."""
    return torch.tensor(
        [
            [
                (0, 0, 0),
                (1, 0, 0),
                (1 + math.cos(a1), math.sin(a1), 0),
                (
                    1 + math.cos(a1) + math.cos(a1 + a2),
                    math.sin(a1) + math.sin(a1 + a2),
                    0,
                ),
            ]
            for a1, a2 in angles
        ]
    )


def cameras():
    """Two pinhole cameras 5 units above the chain, looking down, offset along
    x. A pinhole projection depends on depth, unlike an orthographic one."""
    rot = np.diag([1.0, -1.0, -1.0])
    return [
        quickik.PinholeCamera(500, 500, 320, 240, (x, 0, 5), rot) for x in (-1.0, 1.0)
    ]


def project(positions, projections):
    """`(batch, n_views, 4, 2)` projections of `(batch, 4, 3)` positions."""
    projected = [
        [[projection.project(p)[0] for p in item] for projection in projections]
        for item in positions.numpy()
    ]
    return torch.tensor(np.array(projected), dtype=torch.float64)


ANGLES = [(0.3, 0.2), (0.35, 0.15), (0.25, 0.25)]


def test_forward_recovers_pose():
    positions = chain_positions([(0.4, 0.3)])
    dof_values, root_pos, root_quat = qtorch.SolveIK.apply(
        batched_solver(chain()), positions, torch.ones(1, 4)
    )
    assert dof_values[0].tolist() == pytest.approx([0.4, 0.3], abs=1e-3)
    assert root_pos[0].tolist() == pytest.approx([0, 0, 0], abs=1e-3)
    assert root_quat[0].tolist() == pytest.approx([1, 0, 0, 0], abs=1e-3)


def gradcheck(solver, positions, weights, projections=None, eps=1e-3):
    positions = positions.clone().requires_grad_(True)

    def func(positions):
        return qtorch.SolveIK.apply(solver, positions, weights, projections)

    # Looser than gradcheck's defaults: quickik computes in f32 regardless of
    # these float64 tensors, so a much smaller eps would just measure
    # quantization noise.
    return torch.autograd.gradcheck(func, (positions,), eps=eps, atol=2e-2, rtol=2e-2)


def test_gradcheck_3d():
    weights = torch.ones(len(ANGLES), 4)
    assert gradcheck(batched_solver(chain()), chain_positions(ANGLES), weights)


def test_gradcheck_with_nonuniform_weights_and_multipliers():
    """Both the observation weights and the joints' `weight_multiplier`s must
    be folded into the gradient."""
    tree = chain(joint_weight_multipliers=(1.0, 2.5, 1.0, 0.3))
    weights = torch.ones(len(ANGLES), 4)
    weights[:, 0] = 0.7
    weights[:, 2] = 1.3
    assert gradcheck(batched_solver(tree), chain_positions(ANGLES), weights)


def test_gradcheck_two_cameras():
    projections = cameras()
    positions = project(chain_positions(ANGLES), projections)
    weights = torch.ones(len(ANGLES), len(projections), 4)
    # Pixel observations are about 100 times larger than world positions, so
    # this eps is a comparable perturbation.
    solver = batched_solver(chain())
    assert gradcheck(solver, positions, weights, projections, eps=1e-1)


def test_invalid_item_gets_zero_gradient_not_nan():
    """An item with no observed keypoints has no positive-definite
    linearization to differentiate through; its gradient should be exactly
    zero (not NaN/Inf), while a well-posed item in the same batch still gets
    a real gradient."""
    positions = torch.zeros(2, 4, 3)
    positions[1] = chain_positions([(0.3, 0.2)])[0]
    positions.requires_grad_(True)
    weights = torch.zeros(2, 4)
    weights[1] = 1.0  # only item 1 is observed

    solver = batched_solver(chain())
    dof_values, root_pos, root_quat = qtorch.SolveIK.apply(solver, positions, weights)
    (dof_values.sum() + root_pos.sum() + root_quat.sum()).backward()

    assert torch.all(positions.grad[0] == 0)
    assert torch.isfinite(positions.grad).all()
    assert positions.grad[1].abs().sum().item() > 0


def test_weights_requires_grad_raises():
    weights = torch.ones(1, 4, requires_grad=True)
    with pytest.raises(ValueError, match="weights"):
        qtorch.SolveIK.apply(
            batched_solver(chain()), chain_positions(ANGLES[:1]), weights
        )


def test_quickiksolve_module_matches_solveik():
    solver, projections = batched_solver(chain()), cameras()
    positions = project(chain_positions([(0.4, 0.3)]), projections)
    weights = torch.ones(1, len(projections), 4)

    from_module = qtorch.QuickIKSolve(solver, projections)(positions, weights)
    from_function = qtorch.SolveIK.apply(solver, positions, weights, projections)

    assert from_module[0][0].tolist() == pytest.approx([0.4, 0.3], abs=1e-3)
    for a, b in zip(from_module, from_function, strict=True):
        assert torch.equal(a, b)


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
