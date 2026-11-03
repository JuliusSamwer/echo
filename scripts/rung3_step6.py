"""Step 6 — Planning-pressure boundaries. Train B with the reachability residual; measure emergence.

B is driven by the continuous reachability residual (reward longest chunk whose low-level
residual to the next boundary stays small). Hypothesis: this places boundaries at junctions
(where extending a chunk straight would hit a wall -> residual spikes), which SSL couldn't.
Measure junction lift + chunk length + a quick success number vs flat.

  python scripts/rung3_step6.py [--smoke]
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset
from echo.models.emergent_coarse import build_emergent_coarse
from echo.train_emergent import train_planning_pressure
from echo.planning.mpc import PlannerParams, RecedingHorizonController
from echo.planning.hwm_hierarchical import HWMController

SMOKE = "--smoke" in sys.argv
dev = "cpu"
MD = Path(str(Path(__file__).resolve().parents[1]) + "/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.cost_mode = "min"; cfg.device = dev
params = PlannerParams.from_cfg(cfg)
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)
for p in low.parameters():
    p.requires_grad_(False)
eps = collect_dataset(env, 800, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, np.random.default_rng(2000))

def _arg(f, d, c): return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
EPOCHS = _arg("--epochs", 4 if SMOKE else 70, int)
print(f"[step6] training planning-pressure boundaries (epochs={EPOCHS}) ...")
t0 = time.time()
emg = build_emergent_coarse(low.encoder, 2, macro_dim=16, device=dev, act_scale=cfg.a_max)
train_planning_pressure(emg, low, eps, device=dev, epochs=EPOCHS, verbose=True)
print(f"[step6] trained in {time.time()-t0:.0f}s")


def is_junction(cx, cy):
    o = sum([int(env.open_right[cx, cy]) if cx + 1 < env.W else 0,
             int(env.open_right[cx - 1, cy]) if cx - 1 >= 0 else 0,
             int(env.open_up[cx, cy]) if cy + 1 < env.H else 0,
             int(env.open_up[cx, cy - 1]) if cy - 1 >= 0 else 0])
    return o > 2


nb = njb = ns = njs = 0; lens = []
with torch.no_grad():
    for e in eps[:300]:
        z = low.encode(torch.tensor(e.states, dtype=torch.float32, device=dev))
        g = emg.gate(z[:-1], z[1:]).cpu().numpy(); last = 0
        for t in range(len(g)):
            cx = int(np.clip(np.floor(e.states[t+1, 0]), 0, env.W-1)); cy = int(np.clip(np.floor(e.states[t+1, 1]), 0, env.H-1))
            j = is_junction(cx, cy); ns += 1; njs += j
            if g[t] > 0.5:
                nb += 1; njb += j; lens.append(t-last+1); last = t+1
base = njs/max(ns,1); pjb = njb/max(nb,1)
print(f"\n[step6] boundary rate={nb/max(ns,1):.2f}/step  chunk_len={np.mean(lens) if lens else 0:.1f}  "
      f"P(junc|bnd)={pjb:.2f} base={base:.2f}  LIFT={pjb/max(base,1e-6):.2f}x "
      f"({'JUNCTION-CLUSTERED' if pjb>1.3*base else 'not clustered'})")

# quick success: Emergent-planning-hier vs flat
def run(ctrl, sc, gc, ms, rng):
    env.start_cell = sc; env.set_goal(gc); ctrl.reset(); s = env.reset(); info = {"success": False}
    while env._t < ms:
        a = ctrl.step(s, env.goal, rng); s, _, done, info = env.step(a[0])
        if done: break
    return bool(info["success"])

Ls = [8] if SMOKE else [6, 8, 10]
print(f"\n{'L':>3} | {'flat':>6} | {'Emergent-plan-hier':>18}")
for L in Ls:
    ms = 25 + 8*L; prs, rng = [], np.random.default_rng(7000+L)
    while len(prs) < (5 if SMOKE else 20):
        sc, gc = env.sample_start_goal(L, rng); env.start_cell = sc; env.set_goal(gc)
        if env.shortest_path_len() == L: prs.append((sc, gc))
    f = e = 0
    for pi,(sc,gc) in enumerate(prs):
        sd = 50000+10*L+pi
        f += run(RecedingHorizonController(low, params, H=25, N=256), sc, gc, ms, np.random.default_rng(sd))
        e += run(HWMController(low, emg, params, M=10, N_hi=256, hi_iters=6, N_lo=256), sc, gc, ms, np.random.default_rng(sd))
    print(f"{L:>3} | {f/len(prs):>6.0%} | {e/len(prs):>18.0%}", flush=True)
