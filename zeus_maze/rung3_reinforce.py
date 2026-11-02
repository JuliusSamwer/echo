"""Rung 3 · emergent subgoal policy trained by REINFORCE on PLANNING + SUCCESS only.

The thesis PoC: a learned high-level policy B_theta proposes subgoals (one hop at a time,
variable count — no fixed k, no predictive coarse model). The frozen low-level MPC reaches each
subgoal. B is trained end-to-end by REINFORCE to maximize goal-reaching success, using only:
  success  : reached the goal in the TRUE env,
  planning : geodesic cost-to-go as potential-based shaping + whether each subgoal was reachable.
No junction supervision, no doorway labels. Ground-truth ceiling = oracle placement (0.85).

  python -m zeus_maze.rung3_reinforce [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .config import MazeConfig
from .maze import BottleneckMaze, _first_hit_t
from .base_model import load_frozen_base
from .executor import MPPIConfig
from .hwm import ModelPlanner

BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
POLICY = Path(__file__).resolve().parents[1] / "models" / "rung3_policy.pt"

# THE FROZEN LOW LEVEL — exactly the config that reaches oracle doorways at ~0.85. NEVER changed.
# (running-terminal ModelPlanner; plan in base, act in true env.) Only B is trainable.
LL_MPPI = dict(n_samples=384, n_iters=3, plan_horizon=14)
H_LO = 10   # per-hop step budget (the certified single-hop band)


# ------------------------------------------------------------ policy
class SubgoalPolicy(nn.Module):
    def __init__(self, hidden=128, max_hop=1.3):
        super().__init__()
        self.max_hop = max_hop
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def mean_disp(self, s, g):
        x = torch.cat([s, g, g - s], dim=-1)
        return torch.tanh(self.net(x)) * self.max_hop        # bounded displacement

    def sample(self, s, g, std):
        mean = self.mean_disp(s, g)
        eps = torch.randn_like(mean)
        d = mean + std * eps
        logp = (-0.5 * ((d - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)).sum(-1)
        return d, logp


# ------------------------------------------------------------ frozen low level
def make_lowlevel(base_step, true_step, mx, rng):
    """THE frozen low level — the exact ModelPlanner config that reaches oracle doorways ~0.85."""
    cfg = MPPIConfig(max_step=mx, **LL_MPPI)
    return ModelPlanner(base_step, true_step, cfg, rng=rng)


# ------------------------------------------------------------ fine geodesic cost-to-go (planning signal)
import heapq

_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class GeoGrid:
    """Fine geodesic distance-to-goal — the DENSE, STRONG planning signal (true maze cost-to-go
    at grid resolution). The expensive part (which adjacent cells a wall blocks) is GOAL-INDEPENDENT,
    so we precompute the 8-neighbor open-edge adjacency ONCE (vectorized), then each goal's field is
    a cheap graph Dijkstra. In robotics this cost-to-go becomes a learned value function."""
    def __init__(self, env, res=40):
        self.nx = env.cfg.n_rooms_x * res; self.ny = env.cfg.n_rooms_y * res
        self.cx = env.W / self.nx; self.cy = env.H / self.ny
        ix = (np.arange(self.nx)[None, :] + 0.5) * self.cx
        iy = (np.arange(self.ny)[:, None] + 0.5) * self.cy
        cX = np.broadcast_to(ix, (self.ny, self.nx)); cY = np.broadcast_to(iy, (self.ny, self.nx))
        P = np.stack([cX.ravel(), cY.ravel()], 1)                      # [ny*nx, 2] cell centers
        self.open = np.zeros((len(_DIRS), self.ny, self.nx), bool)
        self.cost = np.zeros(len(_DIRS))
        for k, (dx, dy) in enumerate(_DIRS):
            Q = P + np.array([dx * self.cx, dy * self.cy])             # neighbor centers
            blocked = np.isfinite(_first_hit_t(P, Q, env.wA, env.wB)).reshape(self.ny, self.nx)
            inb = np.zeros((self.ny, self.nx), bool)
            xs = slice(max(0, -dx), self.nx - max(0, dx)); ys = slice(max(0, -dy), self.ny - max(0, dy))
            inb[ys, xs] = True
            self.open[k] = inb & ~blocked
            self.cost[k] = np.hypot(dx * self.cx, dy * self.cy)

    def to_cell(self, p):
        return int(np.clip(p[0] / self.cx, 0, self.nx - 1)), int(np.clip(p[1] / self.cy, 0, self.ny - 1))

    def field(self, goal):
        dist = np.full((self.ny, self.nx), np.inf)
        gx, gy = self.to_cell(np.asarray(goal, float)); dist[gy, gx] = 0.0
        pq = [(0.0, gx, gy)]
        while pq:
            d, x, y = heapq.heappop(pq)
            if d > dist[y, x]:
                continue
            for k, (dx, dy) in enumerate(_DIRS):
                if not self.open[k, y, x]:
                    continue
                nx_, ny_ = x + dx, y + dy; nd = d + self.cost[k]
                if nd < dist[ny_, nx_]:
                    dist[ny_, nx_] = nd; heapq.heappush(pq, (nd, nx_, ny_))
        return dist


def make_geo(geogrid, goal):
    """Return geo(p) = fine geodesic distance from p to goal (dense planning cost-to-go)."""
    field = geogrid.field(goal)

    def geo(p):
        ix, iy = geogrid.to_cell(np.asarray(p, float))
        v = field[iy, ix]
        return float(min(v, 12.0)) if np.isfinite(v) else 12.0
    return geo


def sample_pairs(env, L, n, seed):
    rng = np.random.default_rng(seed); P = []
    while len(P) < n:
        env.reset(L=L, rng=rng); P.append((env.state.copy(), env.goal.copy()))
    return P


# ------------------------------------------------------------ rollout
def rollout(policy, lowlevel, s0, g, geo, cfg, std, H_max,
            R_success=3.0, w_shape=2.0, c_hop=0.05, gamma=0.99):
    """Reward = PROGRESS toward the goal (planning cost-to-go, dominant) + success − per-hop cost.
    No reach bonus (it rewarded trivial tiny hops); per-HOP cost (not per-step) so bigger hops win."""
    s = np.asarray(s0, float).copy()
    logps, rewards = [], []
    geo_s = geo(s)
    success = False; total_steps = 0
    for hop in range(H_max):
        st = torch.tensor(s, dtype=torch.float32); gt = torch.tensor(g, dtype=torch.float32)
        d, logp = policy.sample(st, gt, std)
        subgoal = np.clip(s + d.detach().numpy(), 1e-3, [cfg_W(cfg) - 1e-3, cfg_H(cfg) - 1e-3])
        reached, s2, used = lowlevel.reach(s, subgoal, H_LO, cfg.subgoal_eps)   # FROZEN low level
        total_steps += used
        geo_s2 = geo(s2)
        done = np.linalg.norm(s2 - g) < cfg.goal_eps
        r = w_shape * (geo_s - gamma * geo_s2) - c_hop
        if done:
            r += R_success; success = True
        logps.append(logp); rewards.append(r)
        s = s2; geo_s = geo_s2
        if done:
            break
    return logps, rewards, success, total_steps


def cfg_W(cfg):
    return cfg.n_rooms_x * cfg.room_size


def cfg_H(cfg):
    return cfg.n_rooms_y * cfg.room_size


# ------------------------------------------------------------ train
def train(policy, base_step, true_step, env, geos, pool, cfg, mx, device="cpu",
          updates=400, batch=32, lr=3e-3, std0=0.5, std1=0.15, seed=0, verbose=True):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    lowlevel = make_lowlevel(base_step, true_step, mx, rng)   # FROZEN, built once
    hist = []
    for up in range(updates):
        std = std0 + (std1 - std0) * (up / max(updates - 1, 1))
        succ = []; byL = {}
        recs = []
        for b in range(batch):
            idx = int(rng.integers(len(pool)))
            (s0, g), geo, L = pool[idx], geos[idx], poolL[idx]
            H_max = 3 * L + 4
            logps, rewards, s_ok, steps = rollout(policy, lowlevel, s0, g, geo, cfg, std, H_max)
            # returns-to-go
            G = np.zeros(len(rewards)); acc = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                acc = rewards[t] + 0.99 * acc; G[t] = acc
            recs.append((logps, G, L)); succ.append(s_ok)
            byL.setdefault(L, []).append(G[0] if len(G) else 0.0)
        # per-L baseline
        baseL = {L: np.mean(v) for L, v in byL.items()}
        loss = 0.0; nterm = 0
        for logps, G, L in recs:
            for t, lp in enumerate(logps):
                loss = loss - lp * (G[t] - baseL[L]); nterm += 1
        loss = loss / max(nterm, 1)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(np.mean(succ))
        if verbose and (up % max(1, updates // 20) == 0 or up == updates - 1):
            print(f"  [reinforce] up {up:3d} | train_success={np.mean(succ):.2f} "
                  f"| std={std:.2f} | loss={float(loss):.2f}", flush=True)
    return hist


# module-level parallel arrays filled in main (pool[i], geos[i], poolL[i])
pool = []; geos = []; poolL = []


@torch.no_grad()
def evaluate(policy, base_step, true_step, env, cfg, mx, geogrid, L_values, n, seed=999):
    out = {}
    lowlevel = make_lowlevel(base_step, true_step, mx, np.random.default_rng(4242))
    for L in L_values:
        ps = sample_pairs(env, L, n, seed=seed + L)
        ok = 0
        for i, (s0, g) in enumerate(ps):
            geo = make_geo(geogrid, g)
            _, _, s_ok, _ = rollout(policy, lowlevel, s0, g, geo, cfg, 0.0, 3 * L + 4, R_success=0)
            ok += s_ok
        out[L] = ok / len(ps)
    return out


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    model, cfg_d, base_step = load_frozen_base(BASE)
    cfg = MazeConfig(**{**cfg_d, "geodesic_res": 24})       # coarser field = fast shaping
    env = BottleneckMaze(cfg); true_step = env.step_batch; mx = env.cfg.max_step
    Ls = (2, 3) if smoke else (2, 3, 4)
    per_L = 20 if smoke else 40

    print("[reinforce] precomputing fine-geodesic adjacency + training pool ...", flush=True)
    geogrid = GeoGrid(env, res=30 if smoke else 40)
    for L in Ls:
        for (s0, g) in sample_pairs(env, L, per_L, seed=100 + L):
            pool.append((s0, g)); poolL.append(L)
            geos.append(make_geo(geogrid, g))
    print(f"[reinforce] pool={len(pool)} pairs. training B (dense planning + success REINFORCE) ...", flush=True)

    policy = SubgoalPolicy()
    train(policy, base_step, true_step, env, geos, pool, cfg, mx,
          updates=60 if smoke else 300, batch=16 if smoke else 24)
    torch.save({"state_dict": policy.state_dict()}, POLICY)

    res = evaluate(policy, base_step, true_step, env, cfg, mx, geogrid, Ls,
                   n=15 if smoke else 40)
    print("\n[reinforce] EMERGENT B success vs L (vs flat~0.15, HWM~0.25, oracle 0.85):")
    for L, s in res.items():
        print(f"   L={L}: success={s:.2f}")
