"""Visual: flat MPC vs hierarchical (coarse->fine) MPC on the SAME hard goal.

Renders what each planner actually searches over:
  - LEFT  (flat):  the executed path (gets stuck) + the CEM sample-rollout CLOUD at the
                   stuck point — all imagined rollouts pressing the wall = myopia.
  - RIGHT (hier):  the executed fine path (reaches goal) + the COARSE subgoal chain it
                   commits to + the high-level macro-action search cloud.

Same frozen world model for both; only the planner differs. Picks a high-L (start,goal)
where flat fails and hierarchical succeeds, for maximum contrast.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeus.config import Rung0Config
from zeus.envs.point_maze import PointMaze
from zeus.models.world_model import build_world_model
from zeus.planning.mpc import PlannerParams, RecedingHorizonController
from zeus.planning.hierarchical import HierarchicalController
from zeus.eval.viz import draw_maze, plot_rollout_cloud, plot_subgoal_chain

CACHE = Path("/Users/juliius/Nobel30/Zeus/models/"
             "rung1_model.pt")
OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(exist_ok=True)

GRID, L = (9, 9), 10
cfg = Rung0Config.full(); cfg.grid = GRID; cfg.cost_mode = "min"
cfg.horizon, cfg.n_samples, cfg.cem_iters = 25, 256, 5
torch.manual_seed(0)
# reproduce the EXACT maze the cached model was trained on (same gen call/seed)
rng = np.random.default_rng(0)
env = PointMaze.generate(GRID, path_len=12, rng=rng, a_max=cfg.a_max,
                         goal_eps=cfg.goal_eps, max_steps=200)
model = torch.load(CACHE, weights_only=False)["model"]
params = PlannerParams.from_cfg(cfg)


def run_flat(env, sc, gc, rng):
    env.start_cell = sc; env.set_goal(gc)
    ctrl = RecedingHorizonController(model, params, H=25, N=256)
    ctrl.reset(); s = env.reset(); traj = [s.copy()]
    done = False
    while env._t < env.max_steps:
        plan = ctrl.step(s, env.goal, rng)
        s, _, done, info = env.step(plan[0]); traj.append(s.copy())
        if done: break
    return np.asarray(traj), (done and info["success"])


def run_hier(env, sc, gc, rng):
    env.start_cell = sc; env.set_goal(gc)
    ctrl = HierarchicalController(model, params, k=6, M=10, N_hi=256, hi_iters=6, H_lo=12, N_lo=256)
    ctrl.reset(); s = env.reset(); traj = [s.copy()]; subgoals = [s.copy()]
    done = False
    while env._t < env.max_steps:
        plan = ctrl.step(s, env.goal, rng)
        # record subgoal (decoded to world coords) whenever it changes
        sg = model.decode(torch.as_tensor(ctrl._subgoal, dtype=torch.float32)).detach().cpu().numpy()
        if np.linalg.norm(sg - subgoals[-1]) > 1e-6:
            subgoals.append(sg)
        s, _, done, info = env.step(plan[0]); traj.append(s.copy())
        if done: break
    return np.asarray(traj), np.asarray(subgoals), (done and info["success"])


# find a contrastive pair (flat fails, hier succeeds); cache it so styling re-renders are instant
DATA = Path("/Users/juliius/Nobel30/Zeus/models/"
            "contrast_pair.npz")
env.max_steps = 25 + 8 * L
if DATA.exists():
    d = np.load(DATA, allow_pickle=True)
    sc, gc, tf, th, sg = tuple(d["sc"]), tuple(d["gc"]), d["tf"], d["th"], d["sg"]
    print(f"loaded cached contrast pair start={sc} goal={gc}")
else:
    pick = None
    psearch = np.random.default_rng(123)
    for _ in range(40):
        sc, gc = env.sample_start_goal(L, psearch)
        tf, sf = run_flat(env, sc, gc, np.random.default_rng(7))
        th, sg, sh = run_hier(env, sc, gc, np.random.default_rng(7))
        if (not sf) and sh:
            pick = (sc, gc, tf, th, sg); break
    if pick is None:
        pick = (sc, gc, tf, th, sg)
    sc, gc, tf, th, sg = pick
    np.savez(DATA, sc=np.array(sc), gc=np.array(gc), tf=tf, th=th, sg=sg)
    print(f"contrast pair start={sc} goal={gc} (cached)")
goal = env._cell_center(gc); start = env._cell_center(sc)

def base_panel(ax, title, title_color):
    draw_maze(ax, env, wall_lw=2.4)
    ax.add_patch(plt.Circle(goal, env.goal_eps, color="#2ca02c", alpha=0.20))
    ax.scatter(*goal, marker="*", s=420, color="#2ca02c", edgecolor="k", lw=.6, zorder=8)
    ax.scatter(*start, marker="o", s=150, color="#222", edgecolor="w", lw=1.0, zorder=8)
    ax.set_title(title, fontsize=12, color=title_color, fontweight="bold", pad=8)

# --- build the static figure ---
fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))

# LEFT: flat — sample cloud at the closest-approach (stall) point
ax = axes[0]
base_panel(ax, f"FLAT MPC  —  fails at L={L}", "#d62728")
stuck = tf[np.argmin(np.linalg.norm(tf - goal, axis=1))]
samples = np.clip(np.random.default_rng(1).standard_normal((256, 25, 2)) * 0.5 * cfg.a_max,
                  -cfg.a_max, cfg.a_max)
plot_rollout_cloud(ax, model, stuck, samples, color="#d62728", alpha=0.05, max_show=160)
ax.plot(tf[:, 0], tf[:, 1], "-", color="#d62728", lw=3.0, zorder=5)
ax.scatter(*stuck, marker="X", s=160, color="#d62728", edgecolor="w", lw=1.0, zorder=9)
ax.text(0.5, -0.04, "imagined rollouts (faint) all press the wall — myopic, no route around",
        transform=ax.transAxes, ha="center", fontsize=9.5, color="#444")

# RIGHT: hierarchical — macro search cloud + coarse subgoal chain + fine path
ax = axes[1]
base_panel(ax, f"HIERARCHICAL MPC (HWM-style)  —  succeeds at L={L}", "#1f77b4")
macros = np.clip(np.random.default_rng(1).standard_normal((120, 10, 2)) * 0.5 * cfg.a_max,
                 -cfg.a_max, cfg.a_max)
plot_rollout_cloud(ax, model, start, np.repeat(macros, 6, axis=1), color="#1f77b4", alpha=0.035, max_show=120)
plot_subgoal_chain(ax, sg, color="#ff7f0e")
ax.plot(th[:, 0], th[:, 1], "-", color="#1f77b4", lw=3.0, zorder=5)
ax.text(0.5, -0.04, "coarse subgoals (★) route around walls; fine MPC reaches each one",
        transform=ax.transAxes, ha="center", fontsize=9.5, color="#444")

fig.suptitle("Same frozen world model — only the planner differs   (start ●  →  goal ★)",
             fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = OUT / "planning_flat_vs_hier.png"
fig.savefig(png, dpi=150); plt.close(fig)
print(f"wrote {png}")

# --- animated GIF: render frames explicitly via Pillow (robust, looping) ---
from PIL import Image
T = max(len(tf), len(th))
def padded(t): return np.vstack([t, np.repeat(t[-1:], T - len(t), axis=0)]) if len(t) < T else t
tf_p, th_p = padded(tf), padded(th)
# reveal subgoals progressively as the hier path passes near them
sg_reach_t = [int(np.argmin(np.linalg.norm(th_p - s, axis=1))) for s in sg]

figA, axA = plt.subplots(1, 2, figsize=(13, 6.6))
base_panel(axA[0], "FLAT — fails", "#d62728")
base_panel(axA[1], "HIERARCHICAL — succeeds", "#1f77b4")
figA.suptitle("Same frozen world model — only the planner differs", fontsize=13, y=0.985)
figA.tight_layout(rect=[0, 0, 1, 0.95])
(lf,) = axA[0].plot([], [], "-", color="#d62728", lw=2.8, zorder=5)
(df,) = axA[0].plot([], [], "o", color="#d62728", ms=12, mec="w", zorder=9)
(lh,) = axA[1].plot([], [], "-", color="#1f77b4", lw=2.8, zorder=5)
(dh,) = axA[1].plot([], [], "o", color="#1f77b4", ms=12, mec="w", zorder=9)
(sgp,) = axA[1].plot([], [], "*", color="#ff7f0e", ms=15, mec="k", mew=.5, ls="", zorder=7)

frames = []
for i in range(T):
    lf.set_data(tf_p[:i+1, 0], tf_p[:i+1, 1]); df.set_data([tf_p[i, 0]], [tf_p[i, 1]])
    lh.set_data(th_p[:i+1, 0], th_p[:i+1, 1]); dh.set_data([th_p[i, 0]], [th_p[i, 1]])
    shown = sg[[j for j, tr in enumerate(sg_reach_t) if tr <= i]]
    if len(shown): sgp.set_data(shown[:, 0], shown[:, 1])
    figA.canvas.draw()
    frames.append(Image.fromarray(np.asarray(figA.canvas.buffer_rgba())).convert("P", palette=Image.ADAPTIVE))
plt.close(figA)
# hold the final frame a moment, then loop forever
frames += [frames[-1]] * 12
gif = OUT / "planning_flat_vs_hier.gif"
frames[0].save(gif, save_all=True, append_images=frames[1:], duration=90, loop=0, disposal=2)
print(f"wrote {gif} ({len(frames)} frames, looping)")
