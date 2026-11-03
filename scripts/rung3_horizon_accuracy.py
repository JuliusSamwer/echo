"""THE decisive measurement: long-horizon rollout accuracy, coarse vs flat.

Planning uses the full multi-step rollout, not a single step. Over a horizon of H
primitive steps:
   flat  rolls P  H times          (errors compound over H steps)
   coarse rolls G  H/k times        (errors compound over only H/k steps)
At long H, flat compounds catastrophically (rollout_mse explodes); the coarse model
takes far fewer steps. This script finds the crossover horizon where coarse < flat.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset
from echo.models.coarse_model import build_coarse_model
from echo.train_coarse import train_coarse_model

dev = "cpu"; K = 6
MD = Path(str(Path(__file__).resolve().parents[1]) + "/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.device = dev
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)
eps = collect_dataset(env, 1000, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, np.random.default_rng(2000))
print("training coarse model (k=6, macro_dim=16) ...")
coarse = build_coarse_model(low.encoder, 2, K, macro_dim=16, device=dev, act_scale=cfg.a_max)
train_coarse_model(coarse, eps, device=dev, epochs=120)

HORIZONS = [6, 12, 18, 24, 36, 48, 60]
print(f"\n{'H (steps)':>9} {'flat rmse':>10} {'coarse rmse':>12} {'winner':>8}")
print("-" * 46)
for H in HORIZONS:
    s0, ak, true = [], [], []
    for e in eps:
        T = e.actions.shape[0]
        for t in range(0, T - H, 5):          # stride 5 to subsample
            s0.append(e.states[t]); ak.append(e.actions[t:t + H]); true.append(e.states[t + H])
    if not s0:
        continue
    s0 = torch.tensor(np.asarray(s0), dtype=torch.float32, device=dev)
    ak = torch.tensor(np.asarray(ak), dtype=torch.float32, device=dev)
    true = np.asarray(true)
    with torch.no_grad():
        z0 = low.encode(s0)
        # flat: roll P over all H steps
        flat_pred = low.decode(low.rollout(z0, ak)[:, -1]).cpu().numpy()
        # coarse: split H into H/k macros, encode each chunk, roll G H/k times
        nM = H // K
        macros = torch.stack([coarse.encode_macro(ak[:, i*K:(i+1)*K, :]) for i in range(nM)], dim=1)  # (B,nM,md)
        coarse_pred = low.decode(coarse.coarse_rollout(z0, macros)[:, -1]).cpu().numpy()
    fr = np.sqrt(np.mean((flat_pred - true) ** 2))
    cr = np.sqrt(np.mean((coarse_pred - true) ** 2))
    print(f"{H:>9} {fr:>10.3f} {cr:>12.3f} {'coarse' if cr < fr else 'flat':>8}", flush=True)
