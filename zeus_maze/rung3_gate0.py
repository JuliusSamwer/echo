"""Rung 3 · GATE 0 — certify the reachability energy R(z,g) BEFORE any training.

A plot, not a training run. For a fixed source z, render R(z,g) over a grid of targets g, plus the
descent field -grad_g R. Oracle evaluator only: roll action sequences through the FROZEN P1 (base),
take the GIBBS-weighted endpoint aimed at g, R = ||weighted_endpoint - g|| (Danskin: freeze the weights,
one step).  Two checks the field must pass:
  (1) NO LEAK behind walls  — R must stay HIGH in the wall-shadow (points a straight shot would reach
      only by passing through a wall). If R is low there, P1 leaks and the whole energy is invalid.
  (2) FAT BASIN through the doorway — a wide low-R finger must funnel through the gap, and -grad R must
      point through it, so a descending subgoal is guided to the doorway from a broad catchment.
If it fails, the fix is HERE (inner-planner budget / evaluator design), not inside a descent loop.

  python -m zeus_maze.rung3_gate0
"""
from pathlib import Path
import numpy as np

from .config import MazeConfig
from .maze import BottleneckMaze, _first_hit_t, collide_batch
from .base_model import load_frozen_base

BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
OUT = Path(__file__).resolve().parents[1] / "figures"

H = 12                 # inner-planner horizon (trust budget); sweepable if the field fails
LAMBDA = 0.02          # Gibbs temperature on endpoint distances


def make_step(base_step, env, collide):
    """Inner-planner one-step dynamics. collide=False -> raw frozen base (LEAKS through walls).
    collide=True -> learned base motion CLAMPED by the known collision operator (physics the forward
    model is allowed to know; still no doorway coords anywhere). Straight rays then stop at walls and
    thread only where a gap is -> a wall-faithful reachable set."""
    if not collide:
        return base_step
    def step(s, a):
        s2 = base_step(s, a)
        return collide_batch(np.asarray(s, float), np.asarray(s2, float), env.wA, env.wB)
    return step


def reachable_cloud(step_fn, z, mx, n_seq=6000, seed=0):
    """Roll n_seq action sequences from z through `step_fn` and return ALL visited points (the reachable
    set under P1). Directed rays (heading theta) + per-step noise, variable length so points fill every
    radius. If step_fn leaks, the cloud bleeds behind walls and the plot exposes it."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_seq)
    heads = np.stack([np.cos(theta), np.sin(theta)], 1) * mx           # (n_seq,2) straight headings
    lengths = rng.integers(1, H + 1, n_seq)
    s = np.repeat(np.asarray(z, float)[None], n_seq, 0)               # (n_seq,2)
    pts = [s.copy()]
    for t in range(H):
        a = heads + rng.normal(0, 0.3 * mx, (n_seq, 2))               # straight + jitter
        s = step_fn(s, a)
        pts.append(s.copy())
    P = np.stack(pts, 1)                                              # (n_seq, H+1, 2)
    mask = np.arange(H + 1)[None, :] <= lengths[:, None]              # keep up to each seq's length
    return P[mask]                                                    # (M,2) reachable cloud


def energy_field(cloud, grid, lam=LAMBDA):
    """R(z,g) for every g in grid: Gibbs-weighted endpoint aimed at g, then residual to g.
       w_i(g) = softmax(-||c_i-g||^2/lam) ;  ghat(g)=Σ w_i c_i ;  R=||ghat-g||.  (chunked over grid)"""
    G = np.asarray(grid, float); C = np.asarray(cloud, float)
    R = np.empty(len(G))
    for a in range(0, len(G), 512):
        gb = G[a:a + 512]                                            # (b,2)
        d2 = ((gb[:, None, :] - C[None, :, :]) ** 2).sum(-1)         # (b,M)
        w = np.exp(-(d2 - d2.min(1, keepdims=True)) / lam)
        w /= w.sum(1, keepdims=True)
        ghat = w @ C                                                 # (b,2)
        R[a:a + 512] = np.linalg.norm(ghat - gb, axis=1)
    return R


def run(env, step_fn, mx, z, tag, res=70):
    xs = np.linspace(0.01, env.W - 0.01, res * env.cfg.n_rooms_x)
    ys = np.linspace(0.01, env.H - 0.01, res * env.cfg.n_rooms_y)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X.ravel(), Y.ravel()], 1)

    cloud = reachable_cloud(step_fn, z, mx)
    R = energy_field(cloud, grid).reshape(X.shape)

    # --- quantitative certification ---
    doors = [np.asarray(c, float) for c, _ in env.bottlenecks]
    blocked = np.isfinite(_first_hit_t(np.repeat(np.asarray(z, float)[None], len(grid), 0),
                                       grid, env.wA, env.wB))         # straight z->g crosses a wall
    ddoor = np.min([np.linalg.norm(grid - d, axis=1) for d in doors], axis=0)
    near_door = ddoor < 0.28
    shadow = blocked & ~near_door                                    # wall-shadow (leak test region)
    finger = blocked & near_door                                     # through-doorway region
    leak_R = float(R.ravel()[shadow].mean())
    door_R = float(R.ravel()[finger].mean())
    open_R = float(R.ravel()[~blocked].mean())
    print(f"[gate0:{tag}] z={np.round(z,2)}  R  open={open_R:.3f}  through-door={door_R:.3f}  "
          f"wall-shadow(LEAK)={leak_R:.3f}   contrast(shadow/door)={leak_R/max(door_R,1e-6):.2f}", flush=True)
    return X, Y, R, grid, cloud, doors, (open_R, door_R, leak_R)


def plot(env, panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (z, tag, X, Y, R, doors) in zip(axes, panels):
        im = ax.contourf(X, Y, np.minimum(R, 1.5), levels=24, cmap="viridis")
        gy, gx = np.gradient(np.minimum(R, 1.5))                     # -grad R = descent direction
        step = max(1, X.shape[0] // 24)
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  -gx[::step, ::step], -gy[::step, ::step], color="white", alpha=0.7, scale=25)
        for (x1, y1, x2, y2) in env.walls:
            ax.plot([x1, x2], [y1, y2], color="k", lw=2.5)
        ax.scatter([d[0] for d in doors], [d[1] for d in doors], marker="*", s=240,
                   facecolor="gold", edgecolor="k", zorder=5)
        ax.scatter([z[0]], [z[1]], c="red", s=130, marker="o", edgecolor="w", zorder=6)
        ax.set_title(f"R(z,g) + descent field  [{tag}]\nz=red, doorways=gold  (low R = reachable)")
        ax.set_aspect("equal"); ax.set_xlim(0, env.W); ax.set_ylim(0, env.H)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    OUT.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[gate0] wrote {path}", flush=True)


if __name__ == "__main__":
    import sys
    model, cfg_d, base_step = load_frozen_base(BASE)
    env = BottleneckMaze(MazeConfig(**cfg_d)); mx = env.cfg.max_step
    collide = "--leak" not in sys.argv          # default: collision-clamped evaluator (the fix)
    step_fn = make_step(base_step, env, collide)
    mode = "base+collision (wall-faithful)" if collide else "raw base (LEAKS)"
    print(f"[gate0] certified off-center-doorway maze; H={H}, Gibbs lam={LAMBDA}; evaluator={mode}",
          flush=True)
    sources = [(env.room_center(0), "corner room0"), (env.room_center(4), "center room4 (4 doors)")]
    panels = []
    for z, tag in sources:
        X, Y, R, grid, cloud, doors, stats = run(env, step_fn, mx, np.asarray(z, float), tag)
        panels.append((np.asarray(z, float), tag, X, Y, R, doors))
    out = OUT / ("rung3_gate0_leak.png" if not collide else "rung3_gate0_energy.png")
    plot(env, panels, out)
    print("\n[gate0] VERDICT: no-leak <=> wall-shadow R >> through-door R (contrast >> 1). "
          "Inspect the figure for a FAT low-R finger funnelling through each doorway.", flush=True)
