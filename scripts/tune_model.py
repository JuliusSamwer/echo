"""Fast model-accuracy iteration — no planner eval. Reports open-loop per-step RMSE.

The planner can only discriminate goal-reaching action sequences if the model's
H-step prediction error stays below goal_eps. So before paying for the planner
eval, drive per-step RMSE down here. Target: RMSE @H well under goal_eps (0.2 cell).
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
from zeus.eval.harness import probe_model


def run(cfg, tag):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = PointMaze.generate(cfg.grid, cfg.path_len, rng, a_max=cfg.a_max,
                             goal_eps=cfg.goal_eps, max_steps=cfg.max_steps)
    eps = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len, cfg.explore_momentum,
                          cfg.explore_noise, cfg.goal_directed_frac, rng)
    cov = coverage_report(eps, env)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
    hist = train_world_model(model, eps, cfg, verbose=False)
    pr = probe_model(env, model, cfg, rng, horizon=cfg.horizon)
    rmse = pr["per_step_rmse"]
    print(f"[{tag:22s}] cov={cov:.0%} 1step_loss={hist['onestep'][-1]:.1e} "
          f"| RMSE @1={rmse[0]:.3f} @{cfg.horizon//2}={rmse[cfg.horizon//2-1]:.3f} "
          f"@{cfg.horizon}={rmse[-1]:.3f} | tunnel={pr['tunnel_rate']:.0%}")
    return rmse


def variant(**overrides):
    cfg = Rung0Config.full()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


if __name__ == "__main__":
    # Cheap grid to find what drives per-step RMSE down (now with act-norm + cosine LR).
    print("target: RMSE @H well under goal_eps=0.2 cell\n")
    run(variant(), "full-default")
    run(variant(epochs=200), "epochs=200")
    run(variant(epochs=200, n_train_episodes=600), "ep200+data600")
    run(variant(epochs=200, rollout_len=12), "ep200+K12")
    run(variant(epochs=200, explore_momentum=0.5, explore_noise=0.8), "ep200+diverse-actions")
    run(variant(epochs=300, n_train_episodes=800, rollout_len=12), "ep300+data800+K12")
