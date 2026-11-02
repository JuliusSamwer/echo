"""Flavor-B emergent: a LEARNED boundary net B, driven by the reachability residual.

Faithful port of the grid-stack planning-pressure training (zeus/train_emergent.train_planning_pressure)
onto the bottleneck maze + frozen base. No fixed k, no predictive-horizon coarse model — B is
driven ONLY by the planning signal:

  reachability residual : from z_a, head STRAIGHT at each future point z_{a+j} through the FROZEN
                          base; residual_j = ||pos_after − z_{a+j}||. Small if reachable in a
                          straight hop, SPIKES across a wall / around a turn.
  B objective (L_plan)  : Σ_j p_j·(residual_j − β·steps_j), p_j = B's stopping prob at j.
                          -> p concentrates on the LONGEST chunk that stays reachable, so B learns
                          to break where extending would cross a junction. Light rate anchor only.

The test (vs flavor A's hand-rule): does a LEARNED B localize its gates at doorways? junction
lift = P(doorway | gate fires) / P(doorway | any step). >1 = yes -> continue rung 3 on state;
<=1 = state-only is the wall -> graduate to vision.

  python -m zeus_maze.rung3_flavorB [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .config import MazeConfig
from .maze import BottleneckMaze
from .base_model import load_frozen_base
from .hwm import collect_base_episodes

BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
OUT = Path(__file__).resolve().parents[1] / "figures"


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


def gumbel_sigmoid(logits, tau=1.0, hard=False, noisy=True):
    if noisy:
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        logits = logits + (torch.log(u) - torch.log1p(-u))
    y = torch.sigmoid(logits / tau)
    if hard:
        y = (y > 0.5).float() + y - y.detach()
    return y


class BoundaryNet(nn.Module):
    """Reads [z_t, z_t - z_{t-1}] -> boundary logit. Same mechanism as the grid stack."""
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = _mlp([2 * dim, hidden, hidden // 2, 1])

    def logit(self, z_prev, z):
        return self.net(torch.cat([z, z - z_prev], dim=-1)).squeeze(-1)

    def prob(self, z_prev, z):
        return torch.sigmoid(self.logit(z_prev, z))

    def gate(self, z_prev, z, tau=1.0, hard=False, noisy=True):
        return gumbel_sigmoid(self.logit(z_prev, z), tau, hard, noisy)


def reachability(base, z_a, targets, max_step):
    """z_a (B,2), targets (B,J,2) -> residual (B,J). Head straight at target j through the frozen
    base for j+1 steps; residual = distance to target j. Differentiable in targets (base frozen)."""
    Bsz, J, _ = targets.shape
    pos = z_a[:, None, :].expand(Bsz, J, 2).reshape(Bsz * J, 2)
    tgt = targets.reshape(Bsz * J, 2)
    steps_needed = torch.arange(1, J + 1, device=z_a.device).repeat(Bsz)
    resid = torch.zeros(Bsz * J, device=z_a.device)
    for s in range(J):
        act = tgt - pos
        n = act.norm(dim=-1, keepdim=True)
        act = act * torch.clamp(max_step / (n + 1e-9), max=1.0)   # head straight, L2-clipped
        pos = base(pos, act)                                       # frozen ForwardModel
        arrived = steps_needed == (s + 1)
        if arrived.any():
            resid = torch.where(arrived, (pos - tgt).norm(dim=-1), resid)
    return resid.reshape(Bsz, J)


def _windows(episodes, Lmax):
    Z0, ZF = [], []
    for S in episodes:
        T = len(S) - 1
        for t in range(T - Lmax + 1):
            Z0.append(S[t]); ZF.append(S[t + 1:t + 1 + Lmax])
    return np.asarray(Z0, np.float32), np.asarray(ZF, np.float32)


def train_B(base, episodes, max_step, device="cpu", epochs=60, bs=512, lr=2e-3, Lmax=12,
            beta=0.03, w_anchor=5.0, target_rate=0.16, tau_start=2.0, tau_end=0.4,
            seed=0, verbose=False):
    # w_anchor is strong ON PURPOSE: force a fixed boundary BUDGET (gate ~ target_rate) so B
    # can't collapse to no-boundaries; the planning pressure then decides WHERE they land. The
    # localization question is "given we place some boundaries, do they go to junctions?"
    Z0, ZF = _windows(episodes, Lmax)
    Z0 = torch.tensor(Z0, device=device); ZF = torch.tensor(ZF, device=device)
    W = Z0.shape[0]
    B = BoundaryNet().to(device)
    B.net[-1].bias.data.fill_(-1.6)                       # start sparse
    opt = torch.optim.Adam(B.parameters(), lr=lr)
    steps = torch.arange(1, Lmax + 1, device=device).float()
    rng = np.random.default_rng(seed); nb = max(1, W // bs)
    for ep in range(epochs):
        tau = tau_start + (tau_end - tau_start) * (ep / max(epochs - 1, 1))
        perm = rng.permutation(W); acc = 0.0
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            z0 = Z0[idx]; zf = ZF[idx]; Bsz = z0.shape[0]
            zprev = torch.cat([z0.unsqueeze(1), zf[:, :-1]], dim=1)
            g = B.gate(zprev, zf, tau=tau, hard=False, noisy=True)          # (B,Lmax)
            not_g = (1 - g).clamp(1e-4, 1)
            cum = torch.cumprod(not_g, dim=1)
            prefix = torch.cat([torch.ones(Bsz, 1, device=device), cum[:, :-1]], dim=1)
            p = g * prefix
            p = p + (1 - p.sum(1, keepdim=True)).clamp_min(0) * \
                torch.nn.functional.one_hot(torch.full((Bsz,), Lmax - 1, device=device), Lmax).float()
            with torch.no_grad():
                resid = reachability(base, z0, zf, max_step)               # (B,Lmax)
            plan = (p * (resid - beta * steps)).sum(1).mean()
            anchor = (g.mean() - target_rate) ** 2
            loss = plan + w_anchor * anchor
            opt.zero_grad(); loss.backward(); opt.step(); acc += plan.item()
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            print(f"  [B] ep {ep:3d} L_plan={acc/nb:.3f} tau={tau:.2f} gate={float(g.mean()):.2f}", flush=True)
    return B


# ---------- eval: do B's gates localize at doorways? ----------
def ref_path(s, doors, g, ds=0.14):
    pts = [np.asarray(s, float)] + [np.asarray(d, float) for d in doors] + [np.asarray(g, float)]
    path = []
    for a, b in zip(pts[:-1], pts[1:]):
        nseg = max(1, int(np.linalg.norm(b - a) / ds))
        for t in range(nseg):
            path.append(a + (b - a) * t / nseg)
    path.append(pts[-1]); return np.array(path)


@torch.no_grad()
def junction_lift(env, B, L_values, n=20, tol=0.2, seed=2, device="cpu"):
    rng = np.random.default_rng(seed); nb = njb = ns = njs = 0
    for L in L_values:
        got = 0
        while got < n:
            env.reset(L=L, rng=rng); s = env.state.copy(); g = env.goal.copy()
            rooms, doors = env.path_doorways(s, g)
            if not doors:
                continue
            got += 1
            path = ref_path(s, doors, g); dcen = [np.asarray(d, float) for d in doors]
            near = lambda p: min(np.linalg.norm(p - d) for d in dcen) < tol
            z = torch.tensor(path, dtype=torch.float32, device=device)
            gates = B.prob(z[:-1], z[1:]).cpu().numpy()
            for i, gi in enumerate(gates):
                p = path[i + 1]; nd = near(p)
                ns += 1; njs += nd
                if gi > 0.5:
                    nb += 1; njb += nd
    base = njs / max(ns, 1); pj = njb / max(nb, 1) if nb else 0.0
    return dict(fires=nb, gate_rate=nb / max(ns, 1), P_door_given_fire=pj, base_rate=base,
                lift=(pj / base if base > 0 else 0.0))


def gen_ref_episodes(env, n, L_values, seed):
    """Reference-path trajectories (start -> doorways -> goal): they thread doorways and TURN at
    junctions, so the reachability residual actually spikes there (what B needs to learn from).
    B only ever sees [z, z-z_prev] + the residual — never the doorway labels — so this is not
    supervision, it's the faithful grid-stack setup (stepA used BFS reference paths too)."""
    rng = np.random.default_rng(seed); eps = []
    for L in L_values:
        got = 0
        while got < n:
            env.reset(L=L, rng=rng); s = env.state.copy(); g = env.goal.copy()
            rooms, doors = env.path_doorways(s, g)
            if not doors:
                continue
            path = ref_path(s, doors, g)
            if len(path) >= 14:
                eps.append(path); got += 1
    return eps


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    model, cfg_d, base_step = load_frozen_base(BASE)
    env = BottleneckMaze(MazeConfig(**cfg_d)); mx = env.cfg.max_step
    print("[flavorB] building reference-path training trajectories ...")
    S_eps = gen_ref_episodes(env, 120 if smoke else 300, (2, 3, 4), seed=0)
    print(f"[flavorB] training B (planning-pressure) on {len(S_eps)} reference-path episodes ...")
    B = train_B(model, S_eps, mx, epochs=25 if smoke else 70, verbose=True)
    Ls = (2, 3) if smoke else (2, 3, 4)
    jl = junction_lift(env, B, Ls, n=10 if smoke else 25)
    print(f"\n[flavorB] junction lift={jl['lift']:.2f}  "
          f"(P(door|fire)={jl['P_door_given_fire']:.2f} base={jl['base_rate']:.2f} "
          f"gate_rate={jl['gate_rate']:.2f})")
    print("[flavorB] RESULT:", "LOCALIZES (lift>1.15) — learned B beats the hand-rule; continue rung 3 on state"
          if jl['lift'] > 1.15 else "NO localization — state-only is the wall; graduate to vision")
