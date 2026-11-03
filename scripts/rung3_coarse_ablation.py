"""Ablation: is the coarse model's accuracy limited by the MACRO COMPRESSION (action
abstraction) or by the one-shot k-step prediction itself?

Vary macro_dim (8,16,32 — note 32 > k*act_dim=12, so no information bottleneck) and
also sweep k. Report one-shot k-step RMSE vs flat-rolled-k. The bar to beat for a
long-horizon planning advantage is ~ flat_1step_rmse * sqrt(k).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset
from echo.models.coarse_model import build_coarse_model
from echo.train_coarse import train_coarse_model

dev = "cpu"
MD = Path(str(Path(__file__).resolve().parents[1]) + "/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.device = dev
gen_rng = np.random.default_rng(1000)
env = PointMaze.generate((9, 9), path_len=12, rng=gen_rng, a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)
eps = collect_dataset(env, 1000, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, np.random.default_rng(2000))


def one_shot_rmse(coarse, k):
    s0 = np.concatenate([e.states[:-k] for e in eps], 0); sk = np.concatenate([e.states[k:] for e in eps], 0)
    ac = np.concatenate([e.actions[i:i+k][None] for e in eps for i in range(len(e.actions)-k+1)], 0)
    with torch.no_grad():
        z0 = low.encode(torch.tensor(s0, dtype=torch.float32, device=dev))
        m = coarse.encode_macro(torch.tensor(ac, dtype=torch.float32, device=dev))
        pred = low.decode(coarse.coarse_step(z0, m)).cpu().numpy()
        rolled = low.decode(low.rollout(z0, torch.tensor(ac, dtype=torch.float32, device=dev))[:, -1]).cpu().numpy()
    return float(np.sqrt(np.mean((pred-sk)**2))), float(np.sqrt(np.mean((rolled-sk)**2)))


print(f"{'k':>3} {'macro_dim':>9} | {'one-shot':>9} {'flat-roll':>9} {'bar(0.08√k)':>11}  verdict")
print("-" * 60)
for k in [6, 10]:
    bar = 0.08 * (k ** 0.5)
    for md in [8, 16, 32]:
        torch.manual_seed(0)
        coarse = build_coarse_model(low.encoder, 2, k, macro_dim=md, device=dev, act_scale=cfg.a_max)
        train_coarse_model(coarse, eps, device=dev, epochs=120, verbose=False)
        os_rmse, fr = one_shot_rmse(coarse, k)
        v = "BEATS bar" if os_rmse < bar else ("< flat-roll" if os_rmse < fr else "worse")
        print(f"{k:>3} {md:>9} | {os_rmse:>9.3f} {fr:>9.3f} {bar:>11.3f}  {v}", flush=True)
