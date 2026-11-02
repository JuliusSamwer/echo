"""RUNG 1 — the failure curve.

Fix model capacity and planner budget; vary ONLY the optimal path length L; plot
flat success vs L. The goal is to find the regime where the flat world-model + MPC
collapses — the proof that this env now needs hierarchy/abstraction (HWM-style) to
recover accuracy on long-horizon goals.

Design (isolate L as the sole independent variable):
  - ONE larger maze (so a range of L exists), ONE frozen flat model trained on it.
    => model + data are held fixed; only the query (start,goal) BFS distance varies.
  - FIXED planner budget across all L: H, N, cem_iters constant. A fixed-budget flat
    receding-horizon planner can only look H ahead; as L grows, deceptive junctions
    (where the path must first move AWAY from the goal for longer than H) defeat it.
  - Episode step-limit scales with L (long paths legitimately need more steps), so we
    measure planner capability, not a wall-clock cutoff.

Output: figures/rung1_failure_curve.png + printed table. Model cached for re-plots.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeus.config import Rung0Config
from zeus.envs.point_maze import PointMaze
from zeus.data.collect import collect_dataset, dataset_bounds, coverage_report
from zeus.models.encoder import IdentityEncoder
from zeus.models.world_model import build_world_model
from zeus.train import train_world_model
from zeus.planning.mpc import PlannerParams, RecedingHorizonController
from zeus.eval.harness import evaluate, probe_model

CACHE = Path("/Users/juliius/Nobel30/Zeus/models/"
             "rung1_model.pt")
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

# ---- fixed setup -----------------------------------------------------------
GRID = (9, 9)
L_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]
PAIRS_PER_L = 20
# fixed planner budget (the whole point of rung 1). N=256 keeps the sweep tractable
# on CPU; the collapse SHAPE is budget-invariant (we sweep the same fixed N across L).
H, N, CEM_ITERS = 25, 256, 5

cfg = Rung0Config.full()
cfg.grid = GRID
cfg.horizon, cfg.n_samples, cfg.cem_iters = H, N, CEM_ITERS
cfg.cost_mode = "min"
torch.manual_seed(cfg.seed)
rng = np.random.default_rng(cfg.seed)

# one fixed maze big enough to admit the whole L range
env = PointMaze.generate(GRID, path_len=max(L_VALUES), rng=rng,
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)

# ---- frozen flat model: train once on this maze, with strong coverage ------
if CACHE.exists():
    model = torch.load(CACHE, weights_only=False)["model"]
    print("loaded cached rung1 model")
else:
    print("collecting coverage data on the 9x9 maze ...")
    eps = collect_dataset(env, n_episodes=1500, episode_len=80,
                          momentum=cfg.explore_momentum, noise=cfg.explore_noise,
                          goal_directed_frac=0.3, rng=rng)
    cov = coverage_report(eps, env)
    print(f"coverage={cov:.0%}, transitions={sum(len(e.actions) for e in eps)}")
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
    cfg.epochs = 200
    train_world_model(model, eps, cfg, verbose=False)
    torch.save({"model": model}, CACHE)
    print("trained + cached rung1 model")

pr = probe_model(env, model, cfg, rng, horizon=H)
print(f"[frozen model] RMSE @1={pr['per_step_rmse'][0]:.3f} @{H}={pr['per_step_rmse'][-1]:.3f} "
      f"tunnel={pr['tunnel_rate']:.0%}")
print(f"[fixed budget] H={H} N={N} cem_iters={CEM_ITERS}  (planner cost/plan = {N*H})\n")

# ---- sweep L ---------------------------------------------------------------
params = PlannerParams.from_cfg(cfg)
rows = []
print(f"{'L':>3} | {'success':>8} | {'mean steps':>10} | rollout_mse")
print("-" * 44)
for L in L_VALUES:
    cfg.path_len = L
    cfg.n_eval_episodes = PAIRS_PER_L
    cfg.max_steps = 25 + 8 * L          # step-limit scales with L
    env.max_steps = cfg.max_steps
    ctrl = RecedingHorizonController(model, params, H=H, N=N)
    rep = evaluate(env, model, ctrl, cfg, np.random.default_rng(cfg.seed + 100 + L), verbose=False)
    mean_steps = float(np.mean([r.steps for r in rep.results]))
    rows.append((L, rep.success_rate, mean_steps, rep.mean_rollout_mse))
    print(f"{L:>3} | {rep.success_rate:>7.1%} | {mean_steps:>10.1f} | {rep.mean_rollout_mse:.2e}")

# ---- plot ------------------------------------------------------------------
import matplotlib.pyplot as plt
Ls = [r[0] for r in rows]; succ = [r[1] * 100 for r in rows]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(Ls, succ, "o-", color="#1f77b4", lw=2.2, ms=7, label="flat WM + MPC (fixed budget)")
ax.axhline(95, ls="--", color="#2ca02c", lw=1.3, label="rung-0 gate (95%)")
ax.axhline(50, ls=":", color="#999", lw=1)
# shade the collapse regime (first L below 50%)
collapse = next((L for L, s, *_ in rows if s < 0.5), None)
if collapse is not None:
    ax.axvspan(collapse - 0.5, max(Ls) + 0.5, color="#d62728", alpha=0.07)
    ax.text(collapse, 8, "flat collapses\n(hierarchy needed)", color="#d62728", fontsize=9, ha="left")
ax.set_xlabel("optimal path length  L  (BFS cells)")
ax.set_ylabel("success rate (%)")
ax.set_title(f"Rung 1 failure curve — flat model collapses as L grows\n"
             f"9x9 maze, fixed budget H={H} N={N}", fontsize=10)
ax.set_ylim(-2, 102); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
png = OUT / "rung1_failure_curve.png"
fig.savefig(png, dpi=140); plt.close(fig)
print(f"\nwrote {png}")
