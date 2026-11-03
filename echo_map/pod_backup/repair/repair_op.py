"""Phase 3.1 — the repair operator (frozen models).

Local subgoal repair by gradient descent in latent space:

  E(g) = α · d( ẑ_T(a*), g )²                      # bottom-up: pull g toward what's REACHABLE
       + β · d( F_high(g, u_2..u_m), z_goal )²       # top-down: repaired g must still LEAD to the goal
  g ← g − η ∇_g E(g)      (n_inner Adam steps)

- Term 1 is trivially differentiable in g (ẑ_T(a*) is FROZEN — Danskin: freeze the planner's best action
  seq a*). Term 2 flows through the FROZEN high-level predictor F_high rolled m-1 macro-steps.
- NO model parameters receive gradients (asserted by `assert_only_g_grad`). Only g is optimized.
- `refresh_actions` (default on, once at the end): re-solve a* against the current g via `replan_fn`
  every `k_refresh` steps to refresh ẑ_T; ablated in the sweep.

`d(a,b)` is the planner-cost metric (mean-sq over flattened obs-component); "d²" in the spec == this
squared distance == our mean-sq, so each term uses `_meansq` directly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch

try:
    from pldm.models.utils import flatten_conv_output
except Exception:
    def flatten_conv_output(x):
        return x.reshape(x.shape[0], -1) if x.dim() > 2 else x


def _meansq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Planner-cost distance, KEEPS gradient to inputs (unlike residuals.latent_distance which detaches)."""
    a = flatten_conv_output(a).float()
    b = flatten_conv_output(b).float()
    return ((a - b) ** 2).mean(dim=-1)


@dataclass
class RepairConfig:
    alpha: float = 1.0
    beta: float = 1.0
    eta: float = 1e-2
    n_inner: int = 10
    refresh_actions: bool = True
    k_refresh: int = 10          # default == n_inner  => one refresh at the end


# ------------------------------------------------------------ frozen-parameter guards (unit-tested)
def frozen_snapshot(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def assert_frozen(model, snapshot: dict, atol=0.0):
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), snapshot[n]), f"param changed during repair: {n}"


def assert_only_g_grad(model, g: torch.Tensor):
    """After backward(): g has grad, NO model param has grad."""
    assert g.grad is not None, "g received no gradient"
    for n, p in model.named_parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0, f"model param got gradient: {n}"


# ------------------------------------------------------------ the operator
def repair_subgoal(
    g0: torch.Tensor,                                   # (bs, ...) current subgoal (l2_result.pred_obs[1])
    z_T_pred: torch.Tensor,                             # (bs, ...) ẑ_T(a*) frozen predicted terminal latent
    roll_high: Callable[[torch.Tensor], torch.Tensor],  # g -> F_high(g, u_2..u_m): frozen m-1 macro rollout
    z_goal: torch.Tensor,                              # (bs, ...) encoded goal latent
    cfg: RepairConfig = RepairConfig(),
    replan_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # g -> fresh ẑ_T(a*) (refresh)
    model=None,                                        # optional: assert frozen across the op
) -> torch.Tensor:
    snap = frozen_snapshot(model) if model is not None else None
    g = g0.detach().clone().requires_grad_(True)
    z_T = z_T_pred.detach()
    opt = torch.optim.Adam([g], lr=cfg.eta)
    for step in range(cfg.n_inner):
        if cfg.refresh_actions and replan_fn is not None and step > 0 and step % cfg.k_refresh == 0:
            z_T = replan_fn(g.detach()).detach()        # refresh a* / ẑ_T against current g
        opt.zero_grad(set_to_none=True)
        term1 = cfg.alpha * _meansq(z_T, g)             # bottom-up (g pulled toward reachable ẑ_T)
        term2 = cfg.beta * _meansq(roll_high(g), z_goal)  # top-down (g must still reach the goal)
        E = (term1 + term2).sum()
        E.backward()
        opt.step()
    return g.detach()


def make_roll_high(model, macros_remaining):
    """F_high(g) = roll the FROZEN high-level predictor from g through the remaining macro-actions
    (u_2..u_m), returning the predicted final latent. Params frozen; differentiable in g.
    `macros_remaining`: (bs, m-1, macro_dim) tensor of the coarse plan's remaining macro-actions."""
    l2 = model.level2

    def roll_high(g):
        z = g
        for t in range(macros_remaining.shape[1]):
            out = l2.predictor.forward_multiple(
                z, macros_remaining[:, t:t + 1].transpose(0, 1).float(), 1,
                proprio=None, latents=None, ensemble_input=False,
            )
            z = out.obs_component[-1]
        return z
    return roll_high
