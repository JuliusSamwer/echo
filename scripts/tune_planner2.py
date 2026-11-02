"""Focused planner sweep: close 92% -> >=95%. Caches the trained model to disk so
each config is just an eval. Tests the CEM std-floor + sample-count (NOT more iters,
which over-exploits). 80 eval episodes to rank; the winner gets the 100-ep gate.
"""
import sys, pickle
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
from zeus.eval.harness import evaluate

CACHE = Path("/Users/juliius/Nobel30/Zeus/models/"
             "model_cache.pt")

cfg = Rung0Config.full()
cfg.n_eval_episodes = 80
torch.manual_seed(cfg.seed)
rng = np.random.default_rng(cfg.seed)

env = PointMaze.generate(cfg.grid, cfg.path_len, rng, a_max=cfg.a_max,
                         goal_eps=cfg.goal_eps, max_steps=cfg.max_steps)

if CACHE.exists():
    blob = torch.load(CACHE, weights_only=False)
    model = blob["model"]; print("loaded cached model")
else:
    eps = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len, cfg.explore_momentum,
                          cfg.explore_noise, cfg.goal_directed_frac, rng)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
    train_world_model(model, eps, cfg, verbose=False)
    torch.save({"model": model}, CACHE); print("trained + cached model")

# (label, H, N, iters, min_std_frac, max_steps)
configs = [
    ("min H20 N512 floor0.0",  20, 512, 5, 0.0,  60),
    ("min H20 N512 floor0.1",  20, 512, 5, 0.1,  60),
    ("min H20 N768 floor0.1",  20, 768, 5, 0.1,  60),
    ("min H18 N768 floor0.15", 18, 768, 5, 0.15, 80),
    ("min H20 N1024 floor0.1", 20, 1024, 5, 0.1, 80),
]
for label, H, N, iters, floor, msteps in configs:
    cfg.max_steps = msteps; env.max_steps = msteps
    p = PlannerParams.from_cfg(cfg)
    p.cost_mode = "min"; p.cem_iters = iters; p.min_std_frac = floor
    ctrl = RecedingHorizonController(model, p, H=H, N=N)
    rep = evaluate(env, model, ctrl, cfg, np.random.default_rng(cfg.seed + 1), verbose=False)
    print(f"[{label:24s}] success={rep.success_rate:5.1%} | rollout_mse={rep.mean_rollout_mse:.2e}")
