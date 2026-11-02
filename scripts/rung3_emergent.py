"""Zeus v0: does an emergent (learned-boundary) coarse level match the imposed fixed-k one?

Apples-to-apples with rung3_oneshot: SAME maze, SAME frozen flat substrate, SAME planner
(HWMController) and eval pairs. The ONLY change is the coarse model:
    HWM-hier      : fixed-stride k=6 coarse model (the imposed baseline that won)
    Emergent-hier : SSL-learned boundaries + coarse model (no fixed Δ)   <-- the test
Plus emergence diagnostics: where do boundaries land (junction lift), chunk length, count vs L.

SSL-only (Approach 1): boundaries trained on closure+sufficiency+rate on the frozen substrate,
NO planning signal. (Planning-pressure = Approach 2, a later swap.)

  python scripts/rung3_emergent.py            # full
  python scripts/rung3_emergent.py --smoke    # fast bug-check
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
from zeus.models.emergent_coarse import build_emergent_coarse
from zeus.train_coarse import train_coarse_model
from zeus.train_emergent import train_emergent
from zeus.planning.mpc import PlannerParams, RecedingHorizonController
from zeus.planning.hwm_hierarchical import HWMController

SMOKE = "--smoke" in sys.argv
def _arg(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
W_RATE = _arg("--w-rate", 0.02, float)     # boundary-rate penalty; SWEEP this to find the band
W_CLO = _arg("--w-clo", 1.0, float)        # closure (Chapman-Kolmogorov) weight
dev = resolve_device("cpu")
MD = Path("/Users/juliius/Nobel30/Zeus/models/"
          "learned_models_v2")
MAZE, K = 0, 6
L_VALUES = [6, 12] if SMOKE else [5, 6, 8, 10, 12]
PAIRS = 6 if SMOKE else 30
EPOCHS = 8 if SMOKE else 100

cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.cost_mode = "min"; cfg.device = dev
params = PlannerParams.from_cfg(cfg)
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000 + MAZE),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / f"maze_{MAZE}.pt", weights_only=False)["model"].to(dev)
eps = collect_dataset(env, 1000, 80, cfg.explore_momentum, cfg.explore_noise, 0.3,
                      np.random.default_rng(2000 + MAZE))

# --- imposed fixed-k coarse model (HWM-hier baseline) ---
print("[emergent] training imposed fixed-k coarse model ...")
fixed = build_coarse_model(low.encoder, 2, K, macro_dim=16, device=dev, act_scale=cfg.a_max)
train_coarse_model(fixed, eps, device=dev, epochs=EPOCHS)

# --- emergent (learned-boundary) coarse model ---
print("[emergent] training emergent coarse model (SSL: sufficiency+closure+rate) ...")
t0 = time.time()
emg = build_emergent_coarse(low.encoder, 2, macro_dim=16, device=dev, act_scale=cfg.a_max, k_nominal=K)
print(f"[emergent] w_rate={W_RATE} w_clo={W_CLO}")
train_emergent(emg, low, eps, device=dev, epochs=EPOCHS, w_rate=W_RATE, w_clo=W_CLO, verbose=True)
print(f"[emergent] trained in {time.time()-t0:.0f}s")

# --- emergence diagnostics: where do boundaries land? ---
def is_junction(cx, cy):
    o = 0
    o += env.open_right[cx, cy] if cx + 1 < env.W else 0
    o += env.open_right[cx - 1, cy] if cx - 1 >= 0 else 0
    o += env.open_up[cx, cy] if cy + 1 < env.H else 0
    o += env.open_up[cx, cy - 1] if cy - 1 >= 0 else 0
    return o > 2

with torch.no_grad():
    nb_tot = nb_junc = steps_tot = steps_junc = 0
    chunk_lens = []
    for e in eps[:200]:
        z = low.encode(torch.tensor(e.states, dtype=torch.float32, device=dev))
        g = emg.gate(z[:-1], z[1:]).cpu().numpy()
        bnd = np.where(g > 0.5)[0]
        last = 0
        for t in range(len(g)):
            cx = int(np.clip(e.states[t + 1, 0], 0, env.W - 1)); cy = int(np.clip(e.states[t + 1, 1], 0, env.H - 1))
            j = is_junction(cx, cy); steps_tot += 1; steps_junc += j
            if g[t] > 0.5:
                nb_tot += 1; nb_junc += j; chunk_lens.append(t - last + 1); last = t + 1
base = steps_junc / max(steps_tot, 1)
bnd_rate = nb_junc / max(nb_tot, 1)
print(f"\n[emergence] boundary rate={nb_tot/max(steps_tot,1):.2f}/step  mean chunk len={np.mean(chunk_lens) if chunk_lens else 0:.1f}")
print(f"[emergence] P(junction | boundary)={bnd_rate:.2f}  vs  base P(junction)={base:.2f}  "
      f"-> lift {bnd_rate/max(base,1e-6):.2f}x  ({'clustered at junctions' if bnd_rate>1.3*base else 'NOT junction-clustered'})")


def run(ctrl, sc, gc, ms, rng):
    env.start_cell = sc; env.set_goal(gc); ctrl.reset(); s = env.reset(); info = {"success": False}
    while env._t < ms:
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


print(f"\n{'L':>3} | {'flat':>6} | {'HWM-hier(k=6)':>13} | {'Emergent-hier':>13}")
print("-" * 48)
for L in L_VALUES:
    ms = 25 + 8 * L
    prs = pairs_at(L, PAIRS, np.random.default_rng(7000 + L))
    f = h = e = 0
    for pi, (sc, gc) in enumerate(prs):
        seed = 50000 + 10 * L + pi
        f += run(RecedingHorizonController(low, params, H=25, N=256), sc, gc, ms, np.random.default_rng(seed))
        h += run(HWMController(low, fixed, params, M=10, N_hi=256, hi_iters=6, N_lo=256), sc, gc, ms, np.random.default_rng(seed))
        e += run(HWMController(low, emg, params, M=10, N_hi=256, hi_iters=6, N_lo=256), sc, gc, ms, np.random.default_rng(seed))
    n = len(prs)
    print(f"{L:>3} | {f/n:>6.0%} | {h/n:>13.0%} | {e/n:>13.0%}   (n={n})", flush=True)
