"""PyTorch integration for QuickIK.

`SolveIK` is a `torch.autograd.Function` wrapping `quickik.solve_batch_with_grad`:
it differentiates through the *converged* pose of the batched Gauss-Newton
solve via implicit differentiation of the fixed point (adjoint solve against
the solve's own Cholesky factor), rather than unrolling the solver's
iterations. `QuickIKSolve` is a thin `nn.Module` wrapper around it.

Constraints inherited from `quickik.solve_batch_with_grad`:
- `mapper` must be `None` (3D observations) or an `XYView` (2D); a `Camera`
  isn't supported. The returned Jacobian is always the raw 3D
  keypoint-position Jacobian; for `XYView`, whose 2D projection is exactly
  that Jacobian's first two rows (a fixed, position-independent linear
  map), that's enough to differentiate correctly. A `Camera`'s projection
  Jacobian genuinely depends on position and isn't retained, so gradients
  through one would be wrong.
- `kinematic_tree` must be free-floating (not fixed-base).
- Gradients are only computed for `positions`, not `weights` (that would
  need the solve's residual vector, which isn't currently exposed).
- Items whose last Gauss-Newton iteration wasn't positive-definite get a
  zeroed gradient (their forward values are still meaningful; there just
  isn't a usable linearization to differentiate through).
"""

try:
    import torch
except ImportError as e:
    raise ImportError(
        "quickik.torch requires PyTorch, which isn't installed in this environment. "
        "Install it with `pip install quickik[torch]`, or install PyTorch yourself: "
        "https://pytorch.org/get-started/locally/"
    ) from e

import quickik

__all__ = ["SolveIK", "QuickIKSolve"]


def _resolve_keypoint_to_joint_idx(kinematic_tree, keypoints_order):
    """`result[i]` is `kinematic_tree`'s internal joint index that
    `keypoints_order[i]` (and so `positions`/`weights`'s keypoint axis
    position `i`) corresponds to. Mirrors `quickik.solve_batch_with_grad`'s
    own (Rust-side) name resolution, since that internal mapping isn't
    itself exposed to Python -- only `KinematicTree.joint_names` is."""
    joint_names = kinematic_tree.joint_names
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    if len(name_to_idx) != len(joint_names):
        raise ValueError(
            "kinematic_tree has duplicate joint names, so keypoints_order can't "
            "unambiguously refer to them by name"
        )
    try:
        return [name_to_idx[name] for name in keypoints_order]
    except KeyError as e:
        raise ValueError(f"keypoints_order: unknown joint name {e.args[0]!r}") from e


class SolveIK(torch.autograd.Function):
    """Differentiable batched inverse-kinematics solve.

    `forward(kinematic_tree, config, keypoints_order, mapper, positions,
    weights)` returns `(joint_angles, base_pos, base_quat)`:
    - `joint_angles`: `(batch_size, n_dofs)`, in `kinematic_tree`'s own DOF
      order.
    - `base_pos`: `(batch_size, 3)`.
    - `base_quat`: `(batch_size, 4)`, `(w, x, y, z)`.

    `mapper` is `None` (3D observations) or a `quickik.XYView` instance (2D);
    see the module docstring for why a `Camera` isn't supported. `positions`
    is `(batch_size, n_joints, 3)` if `mapper` is `None`, or `(batch_size,
    n_joints, 2)` if it's an `XYView`; `weights` is `(batch_size, n_joints)`;
    both are in `keypoints_order` (a list of joint names; see
    `KinematicTree.joint_names`). `kinematic_tree`, `config`,
    `keypoints_order`, and `mapper` are treated as constants (no gradient);
    only `positions` gets one back.
    """

    @staticmethod
    def forward(ctx, kinematic_tree, config, keypoints_order, mapper, positions, weights):
        if weights.requires_grad:
            raise ValueError(
                "SolveIK doesn't support gradients w.r.t. weights (only positions): pass "
                "weights.detach(), or a tensor that never had requires_grad set"
            )

        positions_np = positions.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        joint_angles, base_pos, base_quat, jacobian, cholesky_l, valid = (
            quickik.solve_batch_with_grad(
                kinematic_tree, config, keypoints_order, positions_np, weights_np, mapper=mapper
            )
        )

        device, dtype = positions.device, positions.dtype

        def to_tensor(array):
            return torch.from_numpy(array).to(device=device, dtype=dtype)

        keypoint_to_joint_idx = torch.tensor(
            _resolve_keypoint_to_joint_idx(kinematic_tree, keypoints_order),
            dtype=torch.long,
            device=device,
        )
        # Inverse permutation (internal joint index -> external keypoint
        # index): needed to bring `weights` into the Jacobian's internal
        # joint order for `backward`. See the module docstring on why the
        # Jacobian itself stays in internal order.
        internal_to_keypoint_idx = torch.argsort(keypoint_to_joint_idx)

        # The effective per-keypoint weight the solve actually used is
        # `weight * joint.weight_scaler` (see `Solver`'s own normal-equation
        # accumulation), not just `weight` alone.
        joint_weight_scalers = torch.tensor(
            kinematic_tree.joint_weight_scalers, device=device, dtype=dtype
        )
        weights_tensor = to_tensor(weights_np)
        effective_weights_internal = (
            weights_tensor.index_select(1, internal_to_keypoint_idx) * joint_weight_scalers
        )

        base_quat_tensor = to_tensor(base_quat)

        # Rust always returns the raw 3D Jacobian regardless of `mapper` (see
        # `quickik.solve_batch_with_grad`'s docs); for `XYView`, its 2D
        # projected Jacobian is exactly that Jacobian's first two rows, so
        # `backward` just needs to know how many rows to keep per keypoint.
        ctx.n_obs_dims = 2 if isinstance(mapper, quickik.XYView) else 3

        ctx.keypoint_to_joint_idx = keypoint_to_joint_idx
        ctx.effective_weights_internal = effective_weights_internal
        ctx.jacobian = to_tensor(jacobian)
        ctx.cholesky_l = to_tensor(cholesky_l)
        ctx.valid = torch.from_numpy(valid).to(device=device)
        ctx.base_quat = base_quat_tensor

        return to_tensor(joint_angles), to_tensor(base_pos), base_quat_tensor

    @staticmethod
    def backward(ctx, grad_joint_angles, grad_base_pos, grad_base_quat):
        weights = ctx.effective_weights_internal  # (batch, n_joints), internal order
        jacobian = ctx.jacobian  # (batch, 3 * n_joints, state_dim), internal order
        cholesky_l = ctx.cholesky_l  # (batch, state_dim, state_dim)
        valid = ctx.valid  # (batch,) bool
        base_quat = ctx.base_quat  # (batch, 4), (w, x, y, z)

        batch_size, n_joints_times_3, state_dim = jacobian.shape
        n_joints = n_joints_times_3 // 3

        # base_quat's incoming gradient is w.r.t. the quaternion (w, x, y,
        # z) itself, but the linear system operates in the root's 3-dim
        # rotation *tangent* space (State::apply_delta left-multiplies a
        # small-angle quaternion: root_rot_new = exp(delta) * root_rot).
        # For q = (w, vec), d(q)/d(delta)|_0 = 0.5 * [-vec^T; w*I + skew(vec)],
        # so d(Loss)/d(delta) = 0.5 * (w*grad_vec - grad_w*vec -
        # cross(vec, grad_vec)).
        qw, qvec = base_quat[..., 0], base_quat[..., 1:4]
        gw, gvec = grad_base_quat[..., 0], grad_base_quat[..., 1:4]
        v_rot_tangent = 0.5 * (
            qw.unsqueeze(-1) * gvec
            - gw.unsqueeze(-1) * qvec
            - torch.linalg.cross(qvec, gvec, dim=-1)
        )

        v = torch.cat([grad_base_pos, v_rot_tangent, grad_joint_angles], dim=-1)

        # Substitute the identity for invalid (non-positive-definite) items
        # so cholesky_solve doesn't produce NaN/Inf; their contribution is
        # zeroed out below regardless (there's no meaningful linearization
        # to solve against for them).
        eye = torch.eye(state_dim, dtype=cholesky_l.dtype, device=cholesky_l.device)
        safe_cholesky_l = torch.where(valid.view(-1, 1, 1), cholesky_l, eye)
        # `mu = jtj^-1 @ v` (jtj is `Solver`'s normal-equations matrix, i.e.
        # what `cholesky_l` actually factors). The implicit-function-theorem
        # adjoint variable is `lambda = H^-1 @ v` for `H = dg/dx`, but
        # `H = -jtj` here: `g = J^T W r + prior_grad` with `r = obs_pos -
        # fwdkin_pos(x)`, so `dr/dx = -J` and (dropping the Gauss-Newton
        # second-order term) `dg/dx ~= -J^T W J + prior_hessian = -jtj` (the
        # prior term's sign in `accumulate_neutral_pose_prior` -- `+weight`
        # into `jtj` vs. `d(g_prior)/dx = -weight` -- confirms this
        # independently). So `lambda = -mu`, which flips the sign again
        # below and cancels against the leading `-` in `d(Loss)/dtheta =
        # -lambda^T @ dg/dtheta`, leaving a net `+`.
        mu = torch.cholesky_solve(v.unsqueeze(-1), safe_cholesky_l, upper=False).squeeze(-1)
        mu = mu * valid.unsqueeze(-1).to(mu.dtype)

        # d(Loss)/d(obs_pos_k) = weight_k * J_k @ mu, per keypoint, in the
        # tree's internal keypoint order. `J_k` is the raw 3D Jacobian's
        # first `ctx.n_obs_dims` rows (all 3, or `XYView`'s first 2 -- see
        # `forward`).
        jac_per_keypoint = jacobian.view(batch_size, n_joints, 3, state_dim)[
            ..., : ctx.n_obs_dims, :
        ]
        grad_positions_internal = weights.unsqueeze(-1) * torch.einsum(
            "bnij,bj->bni", jac_per_keypoint, mu
        )

        # Un-permute from internal joint order back into keypoints_order,
        # matching the order `positions` was actually given in.
        grad_positions = grad_positions_internal.index_select(1, ctx.keypoint_to_joint_idx)

        return None, None, None, None, grad_positions, None


class QuickIKSolve(torch.nn.Module):
    """Thin `nn.Module` wrapper around `SolveIK`, for dropping into a model.

    Holds `kinematic_tree`/`config`/`keypoints_order`/`mapper` (fixed for
    this module's lifetime); `forward(positions, weights)` returns
    `(joint_angles, base_pos, base_quat)`. See `SolveIK` for the actual
    differentiable op and its constraints (`mapper` must be `None` or an
    `XYView`; no gradient w.r.t. weights).
    """

    def __init__(self, kinematic_tree, config, keypoints_order, mapper=None):
        super().__init__()
        self.kinematic_tree = kinematic_tree
        self.config = config
        self.keypoints_order = keypoints_order
        self.mapper = mapper

    def forward(self, positions, weights):
        return SolveIK.apply(
            self.kinematic_tree,
            self.config,
            self.keypoints_order,
            self.mapper,
            positions,
            weights,
        )
