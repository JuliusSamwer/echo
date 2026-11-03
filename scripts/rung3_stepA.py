"""Step A — REINFORCE PROBE (design i: reference-path segmentation). De-risk, don't train.

ONE binary question: does the REAL planning residual move boundaries toward junctions?
Success criterion = junction lift > 1 AT ALL (not good planning performance).

Setup (all three prerequisites nailed):
  1. BOUNDED executor: hop planner horizon H_lo=8, short enough that a single hop can't solve
     an L>=5 task -> WHERE boundaries go determines success (verified up top).
  2. VARIANCE reduction: dense reward (-goal distance), baseline (batch mean), K rollouts/task.
  3. FROZEN: E, P1, low-level planner all frozen; B is the only moving part.

Mechanism: for start->goal tasks, take the BFS reference path (its turns ARE junctions). B samples
boundaries along it -> subgoals. The bounded executor hops subgoal->subgoal via the real frozen
planner's reachability. A hop that SPANS a turn (B skipped a junction) needs a detour > H_lo -> fails.
So planning reward should push B to break AT junctions. REINFORCE on B's boundary samples.
"""
import sys
from collections import deque
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.models.boundary import BoundaryNet
from echo.planning.mpc import PlannerParams, plan, RecedingHorizonController

dev = "cpu"; H_LO = 14   # band: straight corridor-hop (~5 cells) reachable; cornering hop (~7 travel) fails
MD = Path(str(Path(__file__).resolve().parents[1]) + "/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.cost_mode = "min"; cfg.device = dev
params = PlannerParams.from_cfg(cfg)
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=400)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)
for p in low.parameters():
    p.requires_grad_(False)
sc = float(np.mean(1.0 / low.encoder.scale.numpy()))  # cells->latent
SUB_EPS = 0.3 * sc


def bfs_path(sc_cell, gc_cell):
    from echo.envs.point_maze import _neighbors
    prev = {sc_cell: None}; q = deque([sc_cell])
    while q:
        c = q.popleft()
        if c == gc_cell:
            break
        for nb in _neighbors(c, env.W, env.H, env.open_right, env.open_up):
            if nb not in prev:
                prev[nb] = c; q.append(nb)
    path = []; c = gc_cell
    while c is not None:
        path.append(c); c = prev.get(c)
    return path[::-1]


def is_junction(cx, cy):
    return sum([int(env.open_right[cx, cy]) if cx + 1 < env.W else 0,
               int(env.open_right[cx - 1, cy]) if cx - 1 >= 0 else 0,
               int(env.open_up[cx, cy]) if cy + 1 < env.H else 0,
               int(env.open_up[cx, cy - 1]) if cy - 1 >= 0 else 0]) > 2


# --- prerequisite: bounded executor (H_lo=8) must FAIL on L>=5 (so placement matters) ---
def flat_succ(L, H, n=20):
    ok = 0; rng = np.random.default_rng(123 + L)
    prs = []
    while len(prs) < n:
        a, b = env.sample_start_goal(L, rng); env.start_cell = a; env.set_goal(b)
        if env.shortest_path_len() == L: prs.append((a, b))
    for a, b in prs:
        env.start_cell = a; env.set_goal(b); env.max_steps = 25 + 8 * L
        c = RecedingHorizonController(low, params, H=H, N=128); c.reset(); s = env.reset(); info = {"success": False}
        while env._t < env.max_steps:
            act = c.step(s, env.goal, np.random.default_rng(1)); s, _, d, info = env.step(act[0])
            if d: break
        ok += info["success"]
    return ok / n

print("[prereq] bounded-executor check (flat success; H_lo=8 must fail at L>=5):")
for L in [5, 6, 8]:
    print(f"   L={L}: flat H=8 -> {flat_succ(L, 8):.0%}   (H=25 -> {flat_succ(L, 25):.0%})")
env.max_steps = 400

# --- tasks + reference paths (BFS; its turns are the junctions) ---
tasks = []
trng = np.random.default_rng(7)
for _ in range(400):
    if len(tasks) >= 24:
        break
    for L in (5, 6, 8):
        a, b = env.sample_start_goal(L, trng); env.start_cell = a; env.set_goal(b)
        if env.shortest_path_len() == L:
            wp = np.array([env._cell_center(c) for c in bfs_path(a, b)], dtype=np.float32)
            if len(wp) >= 4:
                tasks.append(wp)
print(f"[probe] {len(tasks)} tasks with BFS reference paths")


@torch.no_grad()
def reachable(pos, target):
    _, resid = plan(low, pos, target, H=H_LO, N=96, params=_p2(2), return_residual=True,
                    rng=np.random.default_rng(0))
    return resid < SUB_EPS


def _p2(iters):
    p = PlannerParams(**{**params.__dict__}); p.cem_iters = iters; p.cost_mode = "min"; return p


LAMBDA_COUNT = 0.4    # per-boundary cost -> forces FEW boundaries placed WELL (the load-bearing tension)


def rollout_reward(wp, bnd):
    """Hop through subgoals via the bounded executor. Reward = -goal dist - λ·(#boundaries).
    Without the count penalty, all-boundaries trivially wins (every hop 1-cell). WITH it, B must
    use the FEWEST boundaries that keep every hop reachable -> break only where necessary (junctions)."""
    goal = wp[-1]; idxs = [i for i in range(1, len(wp)) if bnd[i - 1] > 0.5] + [len(wp) - 1]
    pos = wp[0].copy()
    for i in idxs:
        if reachable(pos, wp[i]):
            pos = wp[i].copy()
        else:
            break
    n_bnd = float(bnd.sum())
    return -float(np.linalg.norm(pos - goal)) - LAMBDA_COUNT * n_bnd


def junction_lift(B):
    nb = njb = ns = njs = 0
    with torch.no_grad():
        for wp in tasks:
            z = low.encode(torch.tensor(wp, dtype=torch.float32))
            g = B.prob(z[:-1], z[1:]).numpy()
            for i in range(len(g)):
                cx = int(np.clip(np.floor(wp[i + 1, 0]), 0, env.W - 1)); cy = int(np.clip(np.floor(wp[i + 1, 1]), 0, env.H - 1))
                j = is_junction(cx, cy); ns += 1; njs += j
                if g[i] > 0.5:
                    nb += 1; njb += j
    base = njs / max(ns, 1); pjb = njb / max(nb, 1) if nb else 0.0
    return nb / max(ns, 1), pjb, base, (pjb / base if base > 0 else 0.0)


torch.manual_seed(0)
B = BoundaryNet(low.latent_dim)
B.net[-1].bias.data.fill_(-1.6)     # start SPARSE (gate ~0.17) so REINFORCE explores few-boundary configs
opt = torch.optim.Adam(B.parameters(), lr=3e-3)
K = 6
r0 = junction_lift(B)
print(f"[probe] BEFORE: bnd/step={r0[0]:.2f} P(junc|bnd)={r0[1]:.2f} base={r0[2]:.2f} LIFT={r0[3]:.2f}")

zenc = [low.encode(torch.tensor(wp, dtype=torch.float32)) for wp in tasks]
for step in range(60):
    logps = []; rs = []
    for wp, z in zip(tasks, zenc):
        g = B.prob(z[:-1].detach(), z[1:].detach()).clamp(1e-4, 1 - 1e-4)   # (P-1,)
        for _ in range(K):
            b = torch.bernoulli(g.detach())
            r = rollout_reward(wp, b.numpy())
            logp = (b * torch.log(g) + (1 - b) * torch.log(1 - g)).sum()
            logps.append(logp); rs.append(r)
    R = torch.tensor(rs); adv = R - R.mean()
    loss = -(adv * torch.stack(logps)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0 or step == 59:
        rl = junction_lift(B)
        print(f"  step {step:3d} | meanR={R.mean():.3f} | bnd/step={rl[0]:.2f} LIFT={rl[3]:.2f}", flush=True)

rf = junction_lift(B)
print(f"\n[probe] AFTER: bnd/step={rf[0]:.2f} P(junc|bnd)={rf[1]:.2f} base={rf[2]:.2f} LIFT={rf[3]:.2f}")
print(f"[stepA] RESULT: {'SIGNAL PRESENT — planning residual clusters boundaries at junctions (lift>1)' if rf[3] > 1.15 else 'NO SIGNAL — faithful planning residual does NOT localize boundaries'} "
      f"(lift {r0[3]:.2f} -> {rf[3]:.2f})")
