"""Planner sweep against ONE cached model. Model error is as low as it'll go;
now find the planner config that clears the gate. Reuses a single trained P so
each row is purely a planner-budget / energy comparison (cheap, apples-to-apples).
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeus.config import Rung0Config
from zeus.envs.point_maze import PointMaze
from zeus.data.collect import collect_dataset, dataset_bounds
from zeus.models.encoder import IdentityEncoder
from zeus.models.world_model import build_world_model
from zeus.train import train_world_model
from zeus.planning.mpc import PlannerParams, RecedingHorizonController
from zeus.eval.harness import evaluate, probe_model

cfg = Rung0Config.full()
cfg.n_eval_episodes = 60          # enough to rank configs; final gate uses 100
torch.manual_seed(cfg.seed)
rng = np.random.default_rng(cfg.seed)

env = PointMaze.generate(cfg.grid, cfg.path_len, rng, a_max=cfg.a_max,
                         goal_eps=cfg.goal_eps, max_steps=cfg.max_steps)
eps = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len, cfg.explore_momentum,
                      cfg.explore_noise, cfg.goal_directed_frac, rng)
low, high = dataset_bounds(eps)
enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
train_world_model(model, eps, cfg, verbose=False)
pr = probe_model(env, model, cfg, rng, horizon=cfg.horizon)
print(f"model: RMSE @1={pr['per_step_rmse'][0]:.3f} @20={pr['per_step_rmse'][-1]:.3f} "
      f"tunnel={pr['tunnel_rate']:.0%}\n")

# (label, H, N, cost_mode, cem_iters)
configs = [
    ("terminal H20 N512",      20, 512,  "terminal",      5),
    ("min H20 N512",           20, 512,  "min",           5),
    ("min H15 N512",           15, 512,  "min",           5),
    ("min H12 N512",           12, 512,  "min",           5),
    ("min H15 N1024 it8",      15, 1024, "min",           8),
    ("term+mean H20 N512",     20, 512,  "terminal+mean", 5),
]
for label, H, N, mode, iters in configs:
    p = PlannerParams.from_cfg(cfg)
    p.cost_mode = mode
    p.cem_iters = iters
    ctrl = RecedingHorizonController(model, p, H=H, N=N)
    rep = evaluate(env, model, ctrl, cfg, rng, verbose=False)
    print(f"[{label:22s}] success={rep.success_rate:5.1%} | rollout_mse={rep.mean_rollout_mse:.2e}")
