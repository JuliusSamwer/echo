"""Flavor-A emergent test: does the RAW reachability residual localize at doorways?

No learned boundary net, no fixed k. residual(s, g) = how close the bounded planner (over the
FROZEN base) gets from s toward g. Emergent hypothesis: residual is LOW through doorways
(reachable in one hop) and SPIKES across walls / around turns — so a chunk greedily extended
until the residual spikes BREAKS at the junction. If this holds on state obs, flavor A works and
we build the learned B (flavor b); if it doesn't, state-only obs is the wall -> graduate to vision.

Two metrics:
  separation AUC : over random (s,g), P(residual[wall between] > residual[clear]). Does the
                   residual even SEE walls? >~0.8 = yes.
  junction lift  : along reference paths (through the true doorways), P(doorway in the look-ahead
                   segment | residual spiked) / P(doorway in segment | any point). >1 = boundaries
                   from the residual localize at junctions.

  python -m zeus_maze.rung3_flavorA [--smoke]
"""
import sys
from pathlib import Path
import numpy as np

from .config import MazeConfig
from .maze import BottleneckMaze, _first_hit_t
from .base_model import load_frozen_base
from .hwm_faithful import cem_plan

OUT = Path(__file__).resolve().parents[1] / "figures"
BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"


def resid(base_step, s, g, H_lo, mx, N=192, iters=2, seed=0):
    """Bounded-planner reachability residual: min distance to g achieved within H_lo steps."""
    return cem_plan(base_step, s, g, H_lo, N, iters, mx, cost_mode="min",
                    rng=np.random.default_rng(seed))[1]


def _seg_point_dist(a, b, p):
    ab = b - a; t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return np.linalg.norm(a + t * ab - p)


def separation(env, base_step, H_lo, mx, n=700, seed=1):
    rng = np.random.default_rng(seed)
    S = np.stack([env.sample_free_point(rng) for _ in range(n)])
    ang = rng.uniform(0, 2 * np.pi, n); rad = np.sqrt(rng.uniform(0.05, 1, n)) * 1.3
    G = np.clip(S + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None], 1e-3,
               [env.W - 1e-3, env.H - 1e-3])
    blocked = np.isfinite(_first_hit_t(S.astype(float), G.astype(float), env.wA, env.wB))
    R = np.array([resid(base_step, S[i], G[i], H_lo, mx, seed=i) for i in range(n)])
    rb, rf = R[blocked], R[~blocked]; m = min(len(rb), len(rf), 400)
    auc = float(np.mean(rng.choice(rb, m) > rng.choice(rf, m))) if m else float("nan")
    return dict(R_clear=float(rf.mean()), R_wall=float(rb.mean()), auc=auc,
                frac_blocked=float(blocked.mean()))


def ref_path(s, doors, g, ds=0.14):
    pts = [np.asarray(s, float)] + [np.asarray(d, float) for d in doors] + [np.asarray(g, float)]
    path = []
    for a, b in zip(pts[:-1], pts[1:]):
        nseg = max(1, int(np.linalg.norm(b - a) / ds))
        for t in range(nseg):
            path.append(a + (b - a) * t / nseg)
    path.append(pts[-1])
    return np.array(path)


def greedy_boundaries(base_step, path, H_lo, mx, eps=0.15, cap=12):
    """Segment the path into reachable chunks: from a, extend the chunk as far as the residual
    stays < eps, break at the last reachable point. Boundary indices = the chunk breaks."""
    bnd = []; a = 0; T = len(path)
    while a < T - 1:
        j = 1
        while j < cap and a + j < T - 1 and resid(base_step, path[a], path[a + j], H_lo, mx, seed=a * 31 + j) < eps:
            j += 1
        b = min(a + j, T - 1); bnd.append(b); a = b
    return bnd


def junction_lift(env, base_step, H_lo, mx, L_values, n=15, eps=0.15, tol=0.2, seed=2):
    """lift = P(boundary point within tol of a doorway CENTER) / P(any path point within tol).
    Low base rate (doorways are sparse) -> discriminative. lift>1 = residual boundaries localize."""
    rng = np.random.default_rng(seed)
    nb = njb = ns = njs = 0
    for L in L_values:
        got = 0
        while got < n:
            env.reset(L=L, rng=rng); s = env.state.copy(); g = env.goal.copy()
            rooms, doors = env.path_doorways(s, g)
            if not doors:
                continue
            got += 1
            path = ref_path(s, doors, g)
            dcen = [np.asarray(d, float) for d in doors]
            near = lambda p: min(np.linalg.norm(p - d) for d in dcen) < tol
            ns += len(path); njs += sum(near(p) for p in path)
            for b in greedy_boundaries(base_step, path, H_lo, mx, eps=eps):
                nb += 1; njb += near(path[b])
    base = njs / max(ns, 1); p = njb / max(nb, 1) if nb else 0.0
    return dict(boundaries=nb, P_door_given_boundary=p, base_rate=base,
                lift=(p / base if base > 0 else 0.0))


def plot_example(env, base_step, H_lo, mx, L=3, Delta=8, eps=0.2, seed=7):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed); env.reset(L=L, rng=rng)
    s, g = env.state.copy(), env.goal.copy(); rooms, doors = env.path_doorways(s, g)
    path = ref_path(s, doors, g)
    rs = np.array([resid(base_step, path[a], path[min(a + Delta, len(path) - 1)], H_lo, mx, seed=a)
                   for a in range(len(path))])
    fig, ax = plt.subplots(figsize=(6, 6))
    for (x1, y1, x2, y2) in env.walls:
        ax.plot([x1, x2], [y1, y2], color="#333", lw=2)
    for d in doors:
        ax.plot(*d, "s", color="#2ca02c", ms=9, alpha=.7)
    sc = ax.scatter(path[:, 0], path[:, 1], c=rs, cmap="inferno", s=40, zorder=3)
    bpts = path[rs > eps]
    ax.scatter(bpts[:, 0], bpts[:, 1], facecolors="none", edgecolors="cyan", s=120, lw=1.8,
               label="residual spike (boundary)", zorder=4)
    ax.plot(*s, "o", color="w", mec="k", ms=10); ax.plot(*g, "*", color="#ff7f0e", ms=18)
    ax.set_aspect("equal"); ax.legend(fontsize=8); fig.colorbar(sc, label="reachability residual")
    ax.set_title(f"Flavor-A: raw reachability residual along the route (H_lo={H_lo})")
    OUT.mkdir(exist_ok=True); p = OUT / "rung3_flavorA_residual.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    model, cfg_d, base_step = load_frozen_base(BASE)
    env = BottleneckMaze(MazeConfig(**cfg_d)); mx = env.cfg.max_step
    Ls = (2, 3) if smoke else (2, 3, 4)
    for H_lo in ((10,) if smoke else (8, 10, 12)):
        sep = separation(env, base_step, H_lo, mx, n=200 if smoke else 700)
        jl = junction_lift(env, base_step, H_lo, mx, Ls, n=8 if smoke else 15)
        print(f"H_lo={H_lo:2d} | sep AUC={sep['auc']:.2f} (R_clear={sep['R_clear']:.2f} "
              f"R_wall={sep['R_wall']:.2f}) | junction lift={jl['lift']:.2f} "
              f"(P(door|bnd)={jl['P_door_given_boundary']:.2f} base={jl['base_rate']:.2f})", flush=True)
    print("figure:", plot_example(env, base_step, 10, mx))
