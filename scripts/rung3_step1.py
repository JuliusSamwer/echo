"""Step 1 — Boundary net B in isolation. Run on trajectories, visualize gates, check non-degenerate.

No coarse model, no planning. Just: does the mechanism run, produce non-saturated gates in (0,1),
and is it inspectable (can we see where they fire on the maze)? Untrained B -> gates should be a
spread around 0.5 (not all-0/all-1). We give it a light nudge so spikes are visible for the overlay.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset
from echo.models.boundary import BoundaryNet
from echo.eval.viz import draw_maze

dev = "cpu"
MD = Path(str(Path(__file__).resolve().parents[1]) + "/models/"
          "learned_models_v2")
OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(exist_ok=True)
cfg = Rung0Config.full(); cfg.grid = (9, 9)
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"]
eps = collect_dataset(env, 200, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, np.random.default_rng(2000))

torch.manual_seed(0)
B = BoundaryNet(low.latent_dim)

# run B on all trajectories (deterministic prob, no gumbel noise) -> gate distribution
all_g = []
with torch.no_grad():
    for e in eps:
        z = low.encode(torch.tensor(e.states, dtype=torch.float32))
        all_g.append(B.prob(z[:-1], z[1:]).numpy())
g_flat = np.concatenate(all_g)
print(f"[step1] gate stats: mean={g_flat.mean():.3f} std={g_flat.std():.3f} "
      f"min={g_flat.min():.3f} max={g_flat.max():.3f}  frac>0.5={np.mean(g_flat>0.5):.2f}")
nondeg = 0.02 < g_flat.mean() < 0.98 and g_flat.std() > 1e-3
print(f"[step1] non-degenerate (not all-0/all-1, has spread): {nondeg}")

# also check Gumbel-ST sharpening path runs
with torch.no_grad():
    z = low.encode(torch.tensor(eps[0].states, dtype=torch.float32))
    soft = B.gate(z[:-1], z[1:], tau=1.0, hard=False)
    hard = B.gate(z[:-1], z[1:], tau=0.3, hard=True)
print(f"[step1] Gumbel-ST runs: soft in [{soft.min():.2f},{soft.max():.2f}], "
      f"hard unique values = {sorted(set(hard.numpy().round(2).tolist()))}")

# visualize gates on a few trajectories over the maze
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, e in zip(axes, eps[:3]):
    draw_maze(ax, env, wall_lw=2.0)
    with torch.no_grad():
        z = low.encode(torch.tensor(e.states, dtype=torch.float32))
        g = B.prob(z[:-1], z[1:]).numpy()
    pos = e.states
    ax.plot(pos[:, 0], pos[:, 1], "-", color="#bbb", lw=1.0, zorder=2)
    sc = ax.scatter(pos[1:, 0], pos[1:, 1], c=g, cmap="hot", s=28, vmin=0, vmax=1, zorder=4)
    ax.set_title(f"gate g_t (mean {g.mean():.2f})", fontsize=9)
fig.colorbar(sc, ax=axes, fraction=0.02, label="boundary gate g_t")
fig.suptitle("Step 1 — boundary net B (untrained), gates overlaid on maze trajectories", fontsize=11)
png = OUT / "rung3_step1_gates.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"[step1] wrote {png}")
print(f"[step1] GATE: {'PASS — mechanism runs, gates non-degenerate & inspectable' if nondeg else 'FAIL'}")
