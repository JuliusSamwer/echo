"""Rigorous rung-2 on the LEARNED model — the apples-to-apples baseline for rung 3.

Why this exists (not a 'realistic side panel'): rung-3's emergent hierarchy runs on a
LEARNED world model, so the imposed-hierarchy baseline it must match has to sit on the
SAME learned substrate — else 'emergent vs imposed' is confounded with 'learned vs
perfect model'. So: same CI rigor as the perfect-model run, but flat & imposed-Δ-
hierarchy on a learned per-maze model. The perfect-model panel stays as the clean
mechanism-isolation result; THIS produces the number rung 3 fights.

Scoped to the baseline (per the caution): failure-regime L's only, paired flat-vs-hier,
mean ± 95% CI. Models are CACHED per maze so rung 3 reuses the identical substrate.

Identical maze gen seeds + pair seeds to scripts/rung2_rigorous.py => the perfect and
learned panels differ ONLY in the model, and overlay directly.

  python scripts/rung2_learned_rigorous.py --shard 0 --n-shards 5 --mazes 10 --pairs 10 \
      --out-dir DIR --model-dir MDIR
"""
import sys, argparse, csv
from pathlib import Path
import numpy as np
import os
import torch
torch.set_num_threads(int(os.environ.get("ECHO_THREADS", "3")))   # avoid multi-proc thread thrash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset, dataset_bounds, coverage_report
from echo.models.encoder import IdentityEncoder
from echo.models.world_model import build_world_model
from echo.train import train_world_model
from echo.planning.mpc import PlannerParams, RecedingHorizonController
from echo.planning.hierarchical import HierarchicalController

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--mazes", type=int, default=10)
ap.add_argument("--pairs", type=int, default=10)
ap.add_argument("--out-dir", type=str, required=True)
ap.add_argument("--model-dir", type=str, required=True)
args = ap.parse_args()

GRID = (9, 9)
L_VALUES = [4, 5, 6, 8, 10, 12]          # failure regime (scoped to the baseline)
cfg = Rung0Config.full(); cfg.grid = GRID; cfg.cost_mode = "min"
# training recipe that produced the validated frozen model (skip + act-norm)
cfg.n_train_episodes = 1500; cfg.episode_len = 80; cfg.goal_directed_frac = 0.3; cfg.epochs = 200
params = PlannerParams.from_cfg(cfg)
H_FLAT, N_FLAT = 25, 256
HI = dict(k=6, M=10, N_hi=256, hi_iters=6, H_lo=12, N_lo=256)

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
mdir = Path(args.model_dir); mdir.mkdir(parents=True, exist_ok=True)
fpath = out / f"shard_{args.shard}.csv"


def get_model(env, m):
    mp = mdir / f"maze_{m}.pt"
    if mp.exists():
        blob = torch.load(mp, weights_only=False)
        return blob["model"], blob["rmse1"]
    torch.manual_seed(m)
    data_rng = np.random.default_rng(2000 + m)
    eps = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len, cfg.explore_momentum,
                          cfg.explore_noise, cfg.goal_directed_frac, data_rng)
    cov = coverage_report(eps, env)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
    train_world_model(model, eps, cfg, verbose=False)
    # quick 1-step accuracy probe (free diagnostic stored with the model)
    s = np.concatenate([e.states[:-1] for e in eps]); a = np.concatenate([e.actions for e in eps])
    s2 = np.concatenate([e.states[1:] for e in eps])
    with torch.no_grad():
        z = model.encode(torch.tensor(s, dtype=torch.float32, device=cfg.device))
        av = torch.tensor(a, dtype=torch.float32, device=cfg.device)
        pred = model.decode(model.step(z, av)).cpu().numpy()
    rmse1 = float(np.sqrt(np.mean((pred - s2) ** 2)))
    torch.save({"model": model, "rmse1": rmse1, "coverage": cov}, mp)
    print(f"[shard {args.shard}] trained maze {m}: cov={cov:.0%} rmse@1={rmse1:.3f}", flush=True)
    return model, rmse1


def run_episode(env, ctrl, max_steps, rng):
    ctrl.reset(); s = env.reset(); info = {"success": False}
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
    gen_rng = np.random.default_rng(1000 + m)        # SAME maze identity as the perfect-model run
    env = PointMaze.generate(GRID, path_len=12, rng=gen_rng,
                             a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
    model, rmse1 = get_model(env, m)
    for L in L_VALUES:
        max_steps = 25 + 8 * L; env.max_steps = max_steps
        pair_rng = np.random.default_rng(7000 + 100 * m + L)   # SAME pairs as the perfect-model run
        pairs = sample_exact_pairs(env, L, args.pairs, pair_rng)
        for pi, (sc, gc) in enumerate(pairs):
            env.start_cell = sc; env.set_goal(gc)
            er = np.random.default_rng(50000 + 1000 * m + 10 * L + pi)
            flat = RecedingHorizonController(model, params, H=H_FLAT, N=N_FLAT)
            sf = run_episode(env, flat, max_steps, np.random.default_rng(er.integers(1 << 30)))
            env.start_cell = sc; env.set_goal(gc)
            hier = HierarchicalController(model, params, **HI)
            sh = run_episode(env, hier, max_steps, np.random.default_rng(er.integers(1 << 30)))
            rows.append((m, L, pi, int(sf), int(sh)))
        fr = np.mean([r[3] for r in rows if r[0] == m and r[1] == L]) if pairs else float("nan")
        hr = np.mean([r[4] for r in rows if r[0] == m and r[1] == L]) if pairs else float("nan")
        print(f"[shard {args.shard}] maze {m} L={L:2d} n={len(pairs):2d} | flat {fr:.2f} hier {hr:.2f}", flush=True)

with open(fpath, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["maze", "L", "pair", "flat", "hier"]); w.writerows(rows)
print(f"[shard {args.shard}] wrote {fpath} ({len(rows)} rows)")
