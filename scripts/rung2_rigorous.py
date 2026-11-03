"""Rigorous rung-2: flat vs hierarchical with error bars, many mazes, PERFECT model.

Addresses the three reproducibility problems:
  1. error bars         -> many mazes x pairs, report mean +/- 95% CI
  2. L is a noisy proxy -> average over many independent mazes per L
  3. model-error confound-> use the EXACT dynamics (TrueDynamicsModel): rollout error
                            is 0, so any flat collapse is PURELY bounded-horizon/search.
PAIRED design: flat and hier run on the SAME (start,goal) pairs -> tight hier-flat diff.

Sharded for parallelism: run several shards over disjoint maze seeds; each writes a CSV
of per-pair outcomes. aggregate_rigorous.py combines them into the curve + CIs.

  python scripts/rung2_rigorous.py --shard 0 --n-shards 5 --mazes 10 --pairs 10 --out-dir DIR
"""
import sys, argparse, csv
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.models.true_model import TrueDynamicsModel
from echo.planning.mpc import PlannerParams, RecedingHorizonController
from echo.planning.hierarchical import HierarchicalController

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--mazes", type=int, default=10)
ap.add_argument("--pairs", type=int, default=10)
ap.add_argument("--out-dir", type=str, required=True)
args = ap.parse_args()

GRID = (9, 9)
L_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]
cfg = Rung0Config.full(); cfg.grid = GRID; cfg.cost_mode = "min"
params = PlannerParams.from_cfg(cfg)
# fixed budgets (identical to the earlier runs, for comparability)
H_FLAT, N_FLAT, CEM = 25, 256, 5
HI = dict(k=6, M=10, N_hi=256, hi_iters=6, H_lo=12, N_lo=256)

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
fpath = out / f"shard_{args.shard}.csv"


def run_episode(env, model, ctrl, max_steps, rng):
    ctrl.reset(); s = env.reset(); done = False; info = {"success": False}
    while env._t < max_steps:
        plan = ctrl.step(s, env.goal, rng)
        s, _, done, info = env.step(plan[0])
        if done: break
    return bool(info["success"])


def sample_exact_pairs(env, L, P, rng):
    pairs, tries = [], 0
    while len(pairs) < P and tries < 400:
        tries += 1
        sc, gc = env.sample_start_goal(L, rng)
        env.start_cell = sc; env.set_goal(gc)
        if env.shortest_path_len() == L:
            pairs.append((sc, gc))
    return pairs


rows = []
maze_ids = [m for m in range(args.mazes) if m % args.n_shards == args.shard]
for m in maze_ids:
    gen_rng = np.random.default_rng(1000 + m)        # this maze's identity (== its seed)
    env = PointMaze.generate(GRID, path_len=max(L_VALUES), rng=gen_rng,
                             a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
    model = TrueDynamicsModel(env)                   # EXACT dynamics -> zero model error
    for L in L_VALUES:
        max_steps = 25 + 8 * L; env.max_steps = max_steps
        pair_rng = np.random.default_rng(7000 + 100 * m + L)
        pairs = sample_exact_pairs(env, L, args.pairs, pair_rng)
        for pi, (sc, gc) in enumerate(pairs):
            env.start_cell = sc; env.set_goal(gc)
            er = np.random.default_rng(50000 + 1000 * m + 10 * L + pi)   # SAME seed -> paired
            flat = RecedingHorizonController(model, params, H=H_FLAT, N=N_FLAT)
            sf = run_episode(env, model, flat, max_steps, np.random.default_rng(er.integers(1 << 30)))
            env.start_cell = sc; env.set_goal(gc)
            hier = HierarchicalController(model, params, **HI)
            sh = run_episode(env, model, hier, max_steps, np.random.default_rng(er.integers(1 << 30)))
            rows.append((m, L, pi, int(sf), int(sh)))
        flat_r = np.mean([r[3] for r in rows if r[0] == m and r[1] == L]) if pairs else float("nan")
        hier_r = np.mean([r[4] for r in rows if r[0] == m and r[1] == L]) if pairs else float("nan")
        print(f"[shard {args.shard}] maze {m} L={L:2d} n={len(pairs):2d} | flat {flat_r:.2f} hier {hier_r:.2f}",
              flush=True)

with open(fpath, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["maze", "L", "pair", "flat", "hier"]); w.writerows(rows)
print(f"[shard {args.shard}] wrote {fpath} ({len(rows)} rows)")
