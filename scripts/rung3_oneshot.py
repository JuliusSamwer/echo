"""Rung-3 first test (ONE maze): does the learned HWM coarse model recover on learned dynamics?

Loads a cached flat (low-level) model, trains the HWM-faithful coarse world model on the
SAME offline data, then compares three controllers on the SAME (start,goal) pairs:
    flat        — flat WM + MPC (the collapsing baseline)
    naive-hier  — old hierarchy that re-rolls the flat model k steps (compounds -> failed)
    HWM-hier    — high level over the LEARNED coarse model (one-shot k-step, no compounding)

If HWM-hier recovers where naive-hier and flat don't, the learned abstract dynamics is the
missing piece — and we scale to many mazes for paper-level CIs next.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeus.config import Rung0Config
from zeus.device import resolve_device
from zeus.envs.point_maze import PointMaze
from zeus.data.collect import collect_dataset
from zeus.models.coarse_model import build_coarse_model
from zeus.train_coarse import train_coarse_model
from zeus.planning.mpc import PlannerParams, RecedingHorizonController
from zeus.planning.hierarchical import HierarchicalController
from zeus.planning.hwm_hierarchical import HWMController

dev = resolve_device("cpu")
MD = Path("/Users/juliius/Nobel30/Zeus/models/"
          "learned_models_v2")
MAZE = 0
K = 6
L_VALUES = [5, 6, 8, 10, 12]
PAIRS = 30

cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.cost_mode = "min"; cfg.device = dev
params = PlannerParams.from_cfg(cfg)

# reproduce maze 0 + its training data, load its cached flat model
gen_rng = np.random.default_rng(1000 + MAZE)
env = PointMaze.generate((9, 9), path_len=12, rng=gen_rng, a_max=cfg.a_max,
                         goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / f"maze_{MAZE}.pt", weights_only=False)["model"].to(dev)
eps = collect_dataset(env, 1000, 80, cfg.explore_momentum, cfg.explore_noise, 0.3,
                      np.random.default_rng(2000 + MAZE))

# train the HWM coarse model on the shared latent
print(f"[rung3] training coarse model (k={K}, macro_dim=8) on maze {MAZE} ...")
coarse = build_coarse_model(low.encoder, act_dim=2, k=K, macro_dim=16, device=dev, act_scale=cfg.a_max)
t0 = time.time()
hist = train_coarse_model(coarse, eps, device=dev, epochs=120, verbose=True)
# coarse one-shot accuracy vs flat-rolled-k (the whole point)
s0 = np.concatenate([e.states[:-K] for e in eps]); sk = np.concatenate([e.states[K:] for e in eps])
ac = np.concatenate([e.actions[i:i+K][None] for e in eps for i in range(len(e.actions)-K+1)], 0)
with torch.no_grad():
    z0 = low.encode(torch.tensor(np.concatenate([e.states[:-K] for e in eps], 0), dtype=torch.float32, device=dev))
    zk = low.encode(torch.tensor(sk, dtype=torch.float32, device=dev))
    m = coarse.encode_macro(torch.tensor(ac, dtype=torch.float32, device=dev))
    one_shot = low.decode(coarse.coarse_step(z0, m)).cpu().numpy()
    # flat rolled k steps for comparison
    a_t = torch.tensor(ac, dtype=torch.float32, device=dev)
    rolled = low.decode(low.rollout(z0, a_t)[:, -1]).cpu().numpy()
sk_true = sk
print(f"[rung3] k-step prediction RMSE (cells):  HWM one-shot={np.sqrt(np.mean((one_shot-sk_true)**2)):.3f}"
      f"   flat-rolled-{K}={np.sqrt(np.mean((rolled-sk_true)**2)):.3f}   ({time.time()-t0:.0f}s)")


def run(env, ctrl, max_steps, rng):
    ctrl.reset(); s = env.reset(); info = {"success": False}
    while env._t < max_steps:
        a = ctrl.step(s, env.goal, rng); s, _, done, info = env.step(a[0])
        if done: break
    return bool(info["success"])


def pairs_at(L, P, rng):
    out, tries = [], 0
    while len(out) < P and tries < 400:
        tries += 1; sc, gc = env.sample_start_goal(L, rng)
        env.start_cell = sc; env.set_goal(gc)
        if env.shortest_path_len() == L: out.append((sc, gc))
    return out


print(f"\n{'L':>3} | {'flat':>6} | {'naive-hier':>10} | {'HWM-hier':>9}")
print("-" * 40)
for L in L_VALUES:
    ms = 25 + 8 * L; env.max_steps = ms
    prs = pairs_at(L, PAIRS, np.random.default_rng(7000 + L))
    f = h = w = 0
    for pi, (sc, gc) in enumerate(prs):
        seed = 50000 + 10 * L + pi
        env.start_cell = sc; env.set_goal(gc)
        f += run(env, RecedingHorizonController(low, params, H=25, N=256), ms, np.random.default_rng(seed))
        env.start_cell = sc; env.set_goal(gc)
        h += run(env, HierarchicalController(low, params, k=K, M=10, N_hi=256, hi_iters=6, H_lo=12, N_lo=256), ms, np.random.default_rng(seed))
        env.start_cell = sc; env.set_goal(gc)
        w += run(env, HWMController(low, coarse, params, M=10, N_hi=256, hi_iters=6, N_lo=256), ms, np.random.default_rng(seed))
    n = len(prs)
    print(f"{L:>3} | {f/n:>6.0%} | {h/n:>10.0%} | {w/n:>9.0%}   (n={n})")
