"""PyTorch integration for QuickIK.

`SolveIK` is a `torch.autograd.Function` wrapping `BatchedSolver.solve`: it
differentiates through the *converged* pose of the batched Gauss-Newton solve
via implicit differentiation of the fixed point (adjoint solve against the
solve's own Cholesky factor), rather than unrolling the solver's iterations.
`QuickIKSolve` is a thin `nn.Module` wrapper around it.

Observations are either 3D positions, or 2D positions in any number of views
(e.g. calibrated cameras), laid out as in `ObservationSequence.from_arrays`.
`BatchedSolver.solve` returns the *residual* Jacobian, already mapped through
each observation's projection, so the adjoint solve below never needs to know
which projections were used.

Constraints inherited from `BatchedSolver`:
- The kinematic tree must be free-floating (not fixed-base), and the solver
  config can't have smoothness priors; `BatchedSolver` already enforces both.
- Gradients are only computed for positions, not weights (that would need the
  solve's residual vector, which isn't exposed) and not a projection's own
  parameters.
- Items whose last Gauss-Newton iteration wasn't positive-definite get a
  zeroed gradient (their forward values are still meaningful; there just
  isn't a usable linearization to differentiate through).
"""

try:
    import torch
except ImportError as e:
    raise ImportError(
        "quickik.torch requires PyTorch, which isn't installed in this environment. "
        "Install it with `pip install quickik[torch]`, or install PyTorch yourself."
    ) from e

import quickik

__all__ = ["QuickIKSolve", "SolveIK"]


class SolveIK(torch.autograd.Function):
    """Differentiable batched inverse-kinematics solve.

    `forward(batched_solver, positions, weights, projections=None)` returns
    `(dof_values, root_pos, root_quat)`:
    - `dof_values`: `(batch_size, n_dofs)`.
    - `root_pos`: `(batch_size, 3)`.
    - `root_quat`: `(batch_size, 4)`, `(w, x, y, z)`.

    Inputs, with every joint of `batched_solver.kinematic_tree` in joint order:
    - 3D observations (`projections=None`): `positions` is
      `(batch_size, n_joints, 3)` and `weights` is `(batch_size, n_joints)`.
    - 2D observations: `projections` is a list of `n_views` projections (e.g.
      `quickik.PinholeCamera`), `positions` is
      `(batch_size, n_views, n_joints, 2)` and `weights` is
      `(batch_size, n_views, n_joints)`.

    A weight that isn't positive (including NaN) marks a missing keypoint.
    `batched_solver` and `projections` are treated as constants (no gradient);
    only `positions` gets one back.
    """

    @staticmethod
    def forward(ctx, batched_solver, positions, weights, projections=None):
        if weights.requires_grad:
            raise ValueError(
                "SolveIK doesn't support gradients w.r.t. weights (only positions): "
                "pass weights.detach(), or a tensor that never had requires_grad set"
            )

        positions_np = positions.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        if projections is None:
            batch = quickik.ObservationSequence.from_arrays(
                pos_3d=positions_np, weights_3d=weights_np
            )
        else:
            batch = quickik.ObservationSequence.from_arrays(
                projections=projections, pos_2d=positions_np, weights_2d=weights_np
            )
        result = batched_solver.solve(batch, with_grad=True)

        device, dtype = positions.device, positions.dtype

        def to_tensor(array):
            return torch.from_numpy(array).to(device=device, dtype=dtype)

        # The weight the solve actually used for each observation: missing
        # keypoints count as 0, and every weight is multiplied by its joint's
        # `weight_multiplier`. Joints are the last axis in both layouts.
        joints = batched_solver.kinematic_tree.joints
        multipliers = torch.tensor(
            [joint.weight_multiplier for joint in joints], device=device, dtype=dtype
        )
        effective_weights = torch.where(weights > 0, weights, 0) * multipliers

        root_quat = to_tensor(result.root_quat)
        # One entry per observation, in the solve's (and Jacobian rows') order.
        ctx.effective_weights = effective_weights.reshape(weights.shape[0], -1)
        ctx.positions_shape = positions.shape
        ctx.jacobian = to_tensor(result.jacobian)
        ctx.cholesky_l = to_tensor(result.cholesky_l)
        ctx.cholesky_valid = torch.from_numpy(result.cholesky_valid).to(device=device)
        ctx.root_quat = root_quat

        return to_tensor(result.dof_values), to_tensor(result.root_pos), root_quat

    @staticmethod
    def backward(ctx, grad_dof_values, grad_root_pos, grad_root_quat):
        weights = ctx.effective_weights  # (batch, n_observations)
        # (batch, obs_dim * n_observations, state_dim), already projected through
        # each observation's projection for 2D observations.
        jacobian = ctx.jacobian
        cholesky_l = ctx.cholesky_l  # (batch, state_dim, state_dim)
        valid = ctx.cholesky_valid  # (batch,) bool
        root_quat = ctx.root_quat  # (batch, 4), (w, x, y, z)

        batch_size, n_rows, state_dim = jacobian.shape
        n_observations = weights.shape[1]
        obs_dim = n_rows // n_observations

        # root_quat's incoming gradient is w.r.t. the quaternion (w, x, y, z)
        # itself, but the linear system operates in the root's 3-dim rotation
        # *tangent* space (`State::apply_delta` left-multiplies a small-angle
        # quaternion: root_quat_new = exp(delta) * root_quat). For q = (w, vec),
        # d(q)/d(delta)|_0 = 0.5 * [-vec^T; w*I + skew(vec)], so
        # d(Loss)/d(delta) = 0.5 * (w*grad_vec - grad_w*vec - cross(vec, grad_vec)).
        qw, qvec = root_quat[..., 0], root_quat[..., 1:4]
        gw, gvec = grad_root_quat[..., 0], grad_root_quat[..., 1:4]
        v_rot_tangent = 0.5 * (
            qw.unsqueeze(-1) * gvec
            - gw.unsqueeze(-1) * qvec
            - torch.linalg.cross(qvec, gvec, dim=-1)
        )

        # State order: root position, root rotation, then every DOF.
        v = torch.cat([grad_root_pos, v_rot_tangent, grad_dof_values], dim=-1)

        # Substitute the identity for invalid (non-positive-definite) items so
        # cholesky_solve doesn't produce NaN/Inf; their contribution is zeroed
        # out below regardless (there's no meaningful linearization to solve
        # against for them).
        eye = torch.eye(state_dim, dtype=cholesky_l.dtype, device=cholesky_l.device)
        safe_cholesky_l = torch.where(valid.view(-1, 1, 1), cholesky_l, eye)
        # `mu = jtj^-1 @ v` (jtj is the solver's normal-equations matrix, i.e.
        # what `cholesky_l` factors). The implicit-function-theorem adjoint
        # variable is `lambda = H^-1 @ v` for `H = dg/dx`, but `H = -jtj` here:
        # `g = J^T W r + prior_grad` with `r = obs_pos - fk_pos(x)`, so
        # `dr/dx = -J` and (dropping the Gauss-Newton second-order term)
        # `dg/dx ~= -J^T W J - prior_hessian = -jtj`. So `lambda = -mu`, which
        # cancels against the leading `-` in `d(Loss)/dtheta = -lambda^T @
        # dg/dtheta`, leaving a net `+`.
        mu = torch.cholesky_solve(v.unsqueeze(-1), safe_cholesky_l, upper=False)
        mu = mu.squeeze(-1) * valid.unsqueeze(-1).to(mu.dtype)

        # d(Loss)/d(obs_pos_k) = weight_k * J_k @ mu, per observation.
        jac_per_obs = jacobian.view(batch_size, n_observations, obs_dim, state_dim)
        grad_positions = weights.unsqueeze(-1) * torch.einsum(
            "bnij,bj->bni", jac_per_obs, mu
        )

        return None, grad_positions.reshape(ctx.positions_shape), None, None


class QuickIKSolve(torch.nn.Module):
    """Thin `nn.Module` wrapper around `SolveIK`, for dropping into a model.

    Holds `batched_solver` and `projections` (fixed for this module's lifetime;
    `None` for 3D observations). `forward(positions, weights)` returns
    `(dof_values, root_pos, root_quat)`. See `SolveIK` for the input layout and
    constraints.
    """

    def __init__(self, batched_solver, projections=None):
        super().__init__()
        self.batched_solver = batched_solver
        self.projections = projections

    def forward(self, positions, weights):
        return SolveIK.apply(self.batched_solver, positions, weights, self.projections)
