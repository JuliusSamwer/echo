"""Full rung-0 gate with model caching (fast iteration on the planner/escape).
Trains the skip model once, caches it, runs the 100-episode gate with the
stuck-escape controller. Re-runs reuse the cache so only the eval re-runs.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset, dataset_bounds
from echo.models.encoder import IdentityEncoder
from echo.models.world_model import build_world_model
from echo.train import train_world_model
from echo.planning.mpc import PlannerParams, RecedingHorizonController
from echo.eval.harness import evaluate, probe_model

CACHE = Path(str(Path(__file__).resolve().parents[1]) + "/models/"
             "skip_model.pt")
cfg = Rung0Config.full()
torch.manual_seed(cfg.seed)
rng = np.random.default_rng(cfg.seed)
env = PointMaze.generate(cfg.grid, cfg.path_len, rng, a_max=cfg.a_max,
                         goal_eps=cfg.goal_eps, max_steps=cfg.max_steps)

if CACHE.exists():
    model = torch.load(CACHE, weights_only=False)["model"]; print("loaded cached skip model")
else:
    eps = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len, cfg.explore_momentum,
                          cfg.explore_noise, cfg.goal_directed_frac, rng)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
    train_world_model(model, eps, cfg, verbose=False)
    torch.save({"model": model}, CACHE); print("trained + cached skip model")

pr = probe_model(env, model, cfg, rng, horizon=cfg.horizon)
print(f"[probe] RMSE @1={pr['per_step_rmse'][0]:.3f} @20={pr['per_step_rmse'][-1]:.3f} "
      f"tunnel={pr['tunnel_rate']:.0%}")

params = PlannerParams.from_cfg(cfg)
ctrl = RecedingHorizonController(model, params, H=cfg.horizon, N=cfg.n_samples)
rep = evaluate(env, model, ctrl, cfg, np.random.default_rng(cfg.seed + 7), verbose=True)
print("-" * 60)
print(f"RESULT  {rep.summary()}")
passed = rep.success_rate >= cfg.pass_threshold
print(f"GATE: {'PASS' if passed else 'FAIL'} ({rep.success_rate:.1%} "
      f"{'>=' if passed else '<'} {cfg.pass_threshold:.0%})")
