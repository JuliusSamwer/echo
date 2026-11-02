"""Step 2 — Coarse model P2 + action encoder, sufficiency only. Gate: beats copy-last.

Fixed-scale (k=6) boundaries isolate P2 (B has no job until Step 5). Train P2 on coarse
prediction accuracy only, then compare next-boundary prediction error to the trivial
copy-last baseline (predict z_{t+k} = z_t). Gate: P2 error < copy-last error.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeus.config import Rung0Config
from zeus.envs.point_maze import PointMaze
from zeus.data.collect import collect_dataset
from zeus.models.coarse_model import build_coarse_model
from zeus.train_coarse import train_coarse_model

dev = "cpu"; K = 6
MD = Path("/Users/juliius/Nobel30/Zeus/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.device = dev
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)
eps = collect_dataset(env, 1000, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, np.random.default_rng(2000))

print("[step2] training coarse P2 (sufficiency: dyn l1) ...")
coarse = build_coarse_model(low.encoder, 2, K, macro_dim=16, device=dev, act_scale=cfg.a_max)
train_coarse_model(coarse, eps, device=dev, epochs=100)

# next-boundary prediction error vs copy-last, at scale k
s0 = np.concatenate([e.states[:-K] for e in eps], 0); sk = np.concatenate([e.states[K:] for e in eps], 0)
ac = np.concatenate([e.actions[i:i+K][None] for e in eps for i in range(len(e.actions)-K+1)], 0)
with torch.no_grad():
    z0 = low.encode(torch.tensor(s0, dtype=torch.float32, device=dev))
    m = coarse.encode_macro(torch.tensor(ac, dtype=torch.float32, device=dev))
    p2 = low.decode(coarse.coarse_step(z0, m)).cpu().numpy()
p2_err = np.sqrt(np.mean((p2 - sk) ** 2))
copylast_err = np.sqrt(np.mean((s0 - sk) ** 2))    # predict z_{t+k} = z_t
print(f"[step2] P2 next-boundary RMSE = {p2_err:.3f} cells | copy-last RMSE = {copylast_err:.3f} cells")
ok = p2_err < copylast_err
print(f"[step2] GATE: {'PASS — P2 predicts its own scale, beats copy-last' if ok else 'FAIL'} "
      f"({p2_err:.3f} {'<' if ok else '>='} {copylast_err:.3f})")
