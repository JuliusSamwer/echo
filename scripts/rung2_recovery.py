"""RUNG 2 POC — abstraction recovers the failure regime.

Same FROZEN flat model, same 9x9 maze, same L-sweep as the rung-1 failure curve.
Run BOTH planners and overlay:
  - flat:          RecedingHorizonController (fixed budget)         [the collapsing baseline]
  - hierarchical:  hand-fixed coarse jump Δ + frozen flat low level [the recovery]
Plot success vs L for both -> figures/rung2_recovery.png.

Pass story: hierarchical stays high through the L>=8 regime where flat collapses,
confirming abstraction is the right lever. This hierarchy is the baseline the later
emergent (not-told-the-level) version must match.

  python scripts/rung2_recovery.py --smoke   # quick validity (few L, few pairs)
  python scripts/rung2_recovery.py            # full overlay
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
from zeus.planning.hierarchical import HierarchicalController
from zeus.eval.harness import evaluate, probe_model

SMOKE = "--smoke" in sys.argv
CACHE = Path("/Users/juliius/Nobel30/Zeus/models/"
             "rung1_model.pt")
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

GRID = (9, 9)
L_VALUES = [3, 8, 12] if SMOKE else [2, 3, 4, 5, 6, 8, 10, 12]
PAIRS_PER_L = 8 if SMOKE else 20
# FIXED flat budget (matches rung-1 failure curve)
H_FLAT, N_FLAT, CEM = 25, 256, 5
# hierarchical LOW-level budget: short horizon (that's the point — low compounding error)
H_LO, N_LO = 12, 256

cfg = Rung0Config.full()
cfg.grid = GRID
cfg.cost_mode = "min"
cfg.horizon, cfg.n_samples, cfg.cem_iters = H_FLAT, N_FLAT, CEM
torch.manual_seed(cfg.seed)
rng = np.random.default_rng(cfg.seed)
env = PointMaze.generate(GRID, path_len=max(L_VALUES), rng=rng,
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)

if CACHE.exists():
    model = torch.load(CACHE, weights_only=False)["model"]; print("loaded cached frozen model")
else:
    print("training frozen model on 9x9 ...")
    eps = collect_dataset(env, 1500, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, rng)
    print(f"coverage={coverage_report(eps, env):.0%}")
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
    cfg.epochs = 200
    train_world_model(model, eps, cfg, verbose=False)
    torch.save({"model": model}, CACHE)

pr = probe_model(env, model, cfg, rng, horizon=H_FLAT)
print(f"[frozen model] RMSE @1={pr['per_step_rmse'][0]:.3f}  (model is FIXED for both planners)")
print(f"flat budget: H={H_FLAT} N={N_FLAT}   |   hier low-level: H={H_LO} N={N_LO}, jump=2.5 cells\n")

params = PlannerParams.from_cfg(cfg)
flat_succ, hier_succ = [], []
print(f"{'L':>3} | {'flat':>7} | {'hier':>7} | {'hier cost/flat cost':>20}")
print("-" * 48)
for L in L_VALUES:
    cfg.path_len = L
    cfg.n_eval_episodes = PAIRS_PER_L
    cfg.max_steps = 25 + 8 * L
    env.max_steps = cfg.max_steps
    seed = cfg.seed + 200 + L

    flat = RecedingHorizonController(model, params, H=H_FLAT, N=N_FLAT)
    rf = evaluate(env, model, flat, cfg, np.random.default_rng(seed), verbose=False)

    hier = HierarchicalController(model, params, k=6, M=10, N_hi=256, hi_iters=6,
                                  H_lo=H_LO, N_lo=N_LO)
    rh = evaluate(env, model, hier, cfg, np.random.default_rng(seed), verbose=False)

    flat_succ.append(rf.success_rate); hier_succ.append(rh.success_rate)
    fc = flat.stats.total_cost or 1
    print(f"{L:>3} | {rf.success_rate:>6.1%} | {rh.success_rate:>6.1%} | {hier.total_cost/fc:>18.2f}x")

# ---- plot ----
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.7))
ax.plot(L_VALUES, [s*100 for s in flat_succ], "o-", color="#d62728", lw=2.2, ms=7,
        label="flat WM + MPC (fixed budget)")
ax.plot(L_VALUES, [s*100 for s in hier_succ], "s-", color="#1f77b4", lw=2.2, ms=7,
        label="hierarchical (hand-fixed coarse jump Δ)")
ax.axhline(95, ls="--", color="#2ca02c", lw=1.2, label="rung-0 gate (95%)")
ax.fill_between(L_VALUES, [f*100 for f in flat_succ], [h*100 for h in hier_succ],
                where=[h >= f for h, f in zip(hier_succ, flat_succ)], color="#1f77b4", alpha=0.08,
                label="recovery from abstraction")
ax.set_xlabel("optimal path length  L  (BFS cells)")
ax.set_ylabel("success rate (%)")
ax.set_title("Rung 2 POC — abstraction recovers the flat-collapse regime\n9x9 maze, identical frozen world model", fontsize=10)
ax.set_ylim(-2, 102); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
png = OUT / ("rung2_recovery_smoke.png" if SMOKE else "rung2_recovery.png")
fig.savefig(png, dpi=140); plt.close(fig)
print(f"\nwrote {png}")
