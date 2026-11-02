"""Step 0 — Freeze and load the substrate; confirm the sampling executor still clears easy mazes.

Loads E, P1 from the baseline model, sets them non-trainable, and runs the flat CEM/MPPI
executor on easy mazes on top of the FROZEN pair. Gate: reproduces rung-0 easy-maze success.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeus.config import Rung0Config
from zeus.device import resolve_device
from zeus.envs.point_maze import PointMaze
from zeus.planning.mpc import PlannerParams, RecedingHorizonController

dev = resolve_device("cpu")
MD = Path("/Users/juliius/Nobel30/Zeus/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.cost_mode = "min"; cfg.device = dev
params = PlannerParams.from_cfg(cfg)
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)

# FREEZE the substrate
for p in low.parameters():
    p.requires_grad_(False)
low.eval()
n_train = sum(p.requires_grad for p in low.parameters())
print(f"[step0] substrate frozen: {n_train} trainable params in (E,P1)  (must be 0)")


def run(sc, gc, ms, rng):
    env.start_cell = sc; env.set_goal(gc)
    ctrl = RecedingHorizonController(low, params, H=25, N=256); ctrl.reset()
    s = env.reset(); info = {"success": False}
    while env._t < ms:
        a = ctrl.step(s, env.goal, rng); s, _, done, info = env.step(a[0])
        if done: break
    return bool(info["success"])


print(f"[step0] flat sampling executor on easy mazes (frozen substrate):")
ok = True
for L in [2, 3]:
    ms = 25 + 8 * L
    prs, rng = [], np.random.default_rng(7000 + L)
    while len(prs) < 25:
        sc, gc = env.sample_start_goal(L, rng); env.start_cell = sc; env.set_goal(gc)
        if env.shortest_path_len() == L: prs.append((sc, gc))
    succ = np.mean([run(sc, gc, ms, np.random.default_rng(50000 + 10 * L + i)) for i, (sc, gc) in enumerate(prs)])
    ok = ok and succ >= 0.9
    print(f"   L={L}: success={succ:.0%}  (n={len(prs)})  {'PASS' if succ >= 0.9 else 'FAIL'}")
print(f"\n[step0] GATE: {'PASS — substrate frozen + executor clears easy mazes' if ok and n_train == 0 else 'FAIL'}")
