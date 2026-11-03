"""Rung-3a · the differentiable reachability surrogate R_psi(s, g).

The load-bearing signal. R_psi predicts the BOUNDED EXECUTOR's residual — how close a
short-horizon low-level planner (H_lo primitive steps, planning against the FROZEN base)
gets from state s toward goal g. It is SMALL when g is reachable in one hop (free space or
straight through a doorway) and SPIKES when reaching g would require rounding a wall/turn.
Because R_psi is a smooth net, it is differentiable in g — so a boundary net placing subgoals
can get gradients that pull boundaries toward the low-residual "reachable-chunk ends", i.e.
the doorways. This module builds it, generates its training targets by actually running the
executor over the frozen base, trains it, and provides the doorway-spike diagnostic (the 3a gate).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .config import MazeConfig
from .maze import BottleneckMaze
from .base_model import ForwardModel, freeze
from .executor import BoundedMPPI, MPPIConfig


# ---------------------------------------------------------------- frozen base
def load_frozen_base(path, device="cpu"):
    """Return (model, cfg_dict, step_batch) for the frozen base. step_batch is a batched
    (states[N,2], actions[N,2]) -> next[N,2] callable — the executor plans against THIS,
    not the true env, so R_psi measures reachability in the agent's learned world."""
    d = torch.load(path, weights_only=False, map_location=device)
    model = ForwardModel(max_step=d["cfg"]["max_step"]).to(device)
    model.load_state_dict(d["state_dict"]); freeze(model)

    @torch.no_grad()
    def step_batch(states, actions):
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
        return model(s, a).cpu().numpy()

    return model, d["cfg"], step_batch


# ---------------------------------------------------------------- R_psi net
class ReachNet(nn.Module):
    """R_psi(s, g) -> predicted low-level residual (>= 0). Softplus head keeps it non-negative
    and differentiable in g. Inputs are (s, g) and the relative vector g-s (gives the net the
    displacement directly, so it mostly has to learn WHERE walls break reachability)."""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, g):
        x = torch.cat([s, g, g - s], dim=-1)
        return torch.nn.functional.softplus(self.net(x)).squeeze(-1)


# ---------------------------------------------------------------- data (run the executor)
def gen_reach_data(env, step_batch, mppi_cfg, H_lo, n_pairs=20000, max_offset=1.3, seed=0):
    """Sample (s, g) pairs and label each with the bounded executor's residual over the frozen
    base. s ~ free points; g = s + random offset up to max_offset (~one hop-plus, so pairs span
    reachable AND just-unreachable / cross-wall). Returns S[N,2], G[N,2], R[N]."""
    rng = np.random.default_rng(seed)
    S, G, R = [], [], []
    batch = 256
    while len(S) < n_pairs:
        ss = np.stack([env.sample_free_point(rng) for _ in range(batch)])
        ang = rng.uniform(0, 2 * np.pi, batch)
        rad = np.sqrt(rng.uniform(0.05, 1.0, batch)) * max_offset   # area-uniform radius
        gg = ss + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None]
        gg = np.clip(gg, 1e-3, [env.W - 1e-3, env.H - 1e-3])
        for i in range(batch):
            ex = BoundedMPPI(step_batch, mppi_cfg, rng=np.random.default_rng(1000 + len(S)))
            _, final, _, _ = ex.reach(ss[i], gg[i], H_lo, env.cfg.subgoal_eps)
            S.append(ss[i]); G.append(gg[i]); R.append(float(np.linalg.norm(final - gg[i])))
        if len(S) % 2048 < batch:
            print(f"  [reach-data] {len(S)}/{n_pairs}", flush=True)
    return (np.asarray(S, np.float32), np.asarray(G, np.float32), np.asarray(R, np.float32))


def train_reach(S, G, R, steps=8000, bs=512, lr=1e-3, device="cpu", seed=0, hidden=256):
    net = ReachNet(hidden).to(device)
    S, G, R = (torch.tensor(x, device=device) for x in (S, G, R))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    for it in range(steps):
        idx = torch.randint(0, S.shape[0], (bs,), generator=g, device=device)
        pred = net(S[idx], G[idx])
        loss = ((pred - R[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"  [reach-train] it={it} mse={loss.item():.3e}", flush=True)
    return net


# ---------------------------------------------------------------- the 3a gate: does it spike?
@torch.no_grad()
def spike_diagnostic(env, net, step_batch, mppi_cfg, H_lo, device="cpu", n=1500, seed=1):
    """Split (s,g) test pairs by whether the STRAIGHT segment s->g crosses a wall. R_psi should
    be LOW for wall-free pairs (reachable) and HIGH for wall-crossing pairs (must detour) — that
    separation IS the junction signal. Returns a dict incl. means and an AUC-like separation."""
    from .maze import _first_hit_t
    rng = np.random.default_rng(seed)
    S = np.stack([env.sample_free_point(rng) for _ in range(n)])
    ang = rng.uniform(0, 2 * np.pi, n); rad = np.sqrt(rng.uniform(0.05, 1.0, n)) * 1.3
    G = np.clip(S + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None], 1e-3, [env.W - 1e-3, env.H - 1e-3])
    blocked = np.isfinite(_first_hit_t(S.astype(float), G.astype(float), env.wA, env.wB))
    st = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    pred = net(st(S), st(G)).cpu().numpy()
    # ground-truth residual from the executor, for calibration
    true = np.empty(n)
    for i in range(n):
        ex = BoundedMPPI(step_batch, mppi_cfg, rng=np.random.default_rng(9000 + i))
        _, f, _, _ = ex.reach(S[i], G[i], H_lo, env.cfg.subgoal_eps)
        true[i] = np.linalg.norm(f - G[i])
    # separation: P(R_psi[blocked] > R_psi[free]) over random pairs (AUC)
    rb, rf = pred[blocked], pred[~blocked]
    m = min(len(rb), len(rf), 400)
    auc = float(np.mean(rng.choice(rb, m) > rng.choice(rf, m))) if m > 0 else float("nan")
    return {
        "R_free_mean": float(pred[~blocked].mean()), "R_blocked_mean": float(pred[blocked].mean()),
        "true_free_mean": float(true[~blocked].mean()), "true_blocked_mean": float(true[blocked].mean()),
        "spike_ratio": float(pred[blocked].mean() / max(pred[~blocked].mean(), 1e-6)),
        "auc_blocked_gt_free": auc, "frac_blocked": float(blocked.mean()),
        "fit_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
    }
