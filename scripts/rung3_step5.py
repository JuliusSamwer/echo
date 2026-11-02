"""Step 5 — SSL-only control. Train B+P2 on sufficiency+closure+rate ONLY (no planning signal).

Two rate settings:
  natural       : one-sided mean(g) push  -> expect collapse (every-step or no boundaries)
  forced-to-chunk: target-rate (mean g -> ~0.16, chunk ~6) -> chunks > 1, so junction lift is measurable
Measure junction lift = P(junction | boundary) / P(junction | any step) in BOTH.
Prediction (the control): even forced-to-chunk, boundaries land ~randomly (lift ~= 1.0),
proving SSL/compression alone cannot place boundaries at decision points.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeus.config import Rung0Config
from zeus.envs.point_maze import PointMaze
from zeus.data.collect import collect_dataset
from zeus.models.emergent_coarse import build_emergent_coarse
from zeus.train_emergent import train_emergent

dev = "cpu"
MD = Path("/Users/juliius/Nobel30/Zeus/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.device = dev
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)
for p in low.parameters():
    p.requires_grad_(False)
eps = collect_dataset(env, 800, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, np.random.default_rng(2000))


def is_junction(cx, cy):
    o = 0
    o += int(env.open_right[cx, cy]) if cx + 1 < env.W else 0
    o += int(env.open_right[cx - 1, cy]) if cx - 1 >= 0 else 0
    o += int(env.open_up[cx, cy]) if cy + 1 < env.H else 0
    o += int(env.open_up[cx, cy - 1]) if cy - 1 >= 0 else 0
    return o > 2


def junction_lift(coarse):
    nb = njb = ns = njs = 0
    lens = []
    with torch.no_grad():
        for e in eps[:300]:
            z = coarse.encoder.encode(torch.tensor(e.states, dtype=torch.float32, device=dev))
            g = coarse.gate(z[:-1], z[1:]).cpu().numpy()
            last = 0
            for t in range(len(g)):
                cx = int(np.clip(np.floor(e.states[t + 1, 0]), 0, env.W - 1))
                cy = int(np.clip(np.floor(e.states[t + 1, 1]), 0, env.H - 1))
                j = is_junction(cx, cy); ns += 1; njs += j
                if g[t] > 0.5:
                    nb += 1; njb += j; lens.append(t - last + 1); last = t + 1
    base = njs / max(ns, 1); pjb = njb / max(nb, 1)
    return nb / max(ns, 1), (np.mean(lens) if lens else 0.0), pjb, base, (pjb / base if base > 0 else 0.0)


print(f"{'condition':>16} | {'bnd/step':>8} {'chunk_len':>9} | {'P(junc|bnd)':>11} {'base':>6} {'LIFT':>6}")
print("-" * 70)
for name, kw in [("natural", dict(w_rate=0.02, target_rate=None)),
                 ("forced-to-chunk", dict(w_rate=2.0, target_rate=0.16))]:
    torch.manual_seed(0)
    emg = build_emergent_coarse(low.encoder, 2, macro_dim=16, device=dev, act_scale=cfg.a_max)
    train_emergent(emg, low, eps, device=dev, epochs=60, w_clo=1.0, **kw)
    rate, clen, pjb, base, lift = junction_lift(emg)
    print(f"{name:>16} | {rate:>8.2f} {clen:>9.1f} | {pjb:>11.2f} {base:>6.2f} {lift:>6.2f}", flush=True)

print("\n[step5] GATE/prediction: if forced-to-chunk lift ~= 1.0 -> SSL can't place boundaries "
      "meaningfully (the control). If lift > 1.3 -> SSL alone already finds structure (upside).")
