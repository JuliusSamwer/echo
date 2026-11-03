"""Step 3 — Add the closure (Chapman-Kolmogorov) loss; watch the residual fall.

Closure: roll P2 forward M fixed-k boundaries vs roll the frozen P1 forward the real fine
steps and read latents at those boundary times; penalize the gap. This is the load-bearing
property. Gate: closure residual measurably DECREASES during training.
(Entry-context slot: tested separately; if closure holds without it, we drop it — it does here.)
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset
from echo.models.coarse_model import build_coarse_model

dev = "cpu"; K = 6; M = 4                       # 4 coarse boundaries = 24 fine steps
MD = Path(str(Path(__file__).resolve().parents[1]) + "/models/"
          "learned_models_v2")
cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.device = dev
env = PointMaze.generate((9, 9), path_len=12, rng=np.random.default_rng(1000),
                         a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
low = torch.load(MD / "maze_0.pt", weights_only=False)["model"].to(dev)
for p in low.parameters():
    p.requires_grad_(False)
eps = collect_dataset(env, 1000, 80, cfg.explore_momentum, cfg.explore_noise, 0.3, np.random.default_rng(2000))

# windows of M*k fine steps
H = M * K
s0, af = [], []
for e in eps:
    for t in range(0, len(e.actions) - H, 3):
        s0.append(e.states[t]); af.append(e.actions[t:t + H])
S0 = torch.tensor(np.stack(s0), dtype=torch.float32, device=dev)
AF = torch.tensor(np.stack(af), dtype=torch.float32, device=dev)
W = S0.shape[0]

coarse = build_coarse_model(low.encoder, 2, K, macro_dim=16, device=dev, act_scale=cfg.a_max)
opt = torch.optim.Adam(list(coarse.E_a.parameters()) + list(coarse.G.parameters()) +
                       list(coarse.D_a.parameters()), lr=1e-3)
rng = np.random.default_rng(0)


def closure_residual(idx):
    z0 = low.encode(S0[idx]); af = AF[idx]; B = z0.shape[0]
    with torch.no_grad():
        fine = low.rollout(z0, af)                       # (B,H,z)
    fine_wp = fine[:, K - 1::K, :]                        # (B,M,z) fine latent at boundary times
    zc = z0; preds = []
    for j in range(M):
        chunk = af[:, j * K:(j + 1) * K, :]
        m = coarse.encode_macro(chunk)
        zc = coarse.coarse_step(zc, m)
        preds.append(zc)
    coarse_wp = torch.stack(preds, dim=1)                # (B,M,z)
    return torch.linalg.norm(coarse_wp - fine_wp, dim=-1).mean()


print("[step3] training coarse with dyn + closure; logging closure residual ...")
res_hist = []
for ep in range(80):
    perm = rng.permutation(W)
    acc = 0.0; nb = max(1, W // 512)
    for b in range(nb):
        idx = perm[b * 512:(b + 1) * 512]
        clo = closure_residual(idx)
        # also a 1-step-scale sufficiency (dyn) so P2 stays grounded
        z0 = low.encode(S0[idx]); m0 = coarse.encode_macro(AF[idx][:, :K])
        with torch.no_grad():
            zk = low.rollout(z0, AF[idx][:, :K])[:, -1]
        dyn = F.l1_loss(coarse.coarse_step(z0, m0), zk)
        loss = clo + dyn
        opt.zero_grad(); loss.backward(); opt.step()
        acc += clo.item()
    res_hist.append(acc / nb)
    if ep % 10 == 0 or ep == 79:
        print(f"   ep {ep:3d} | closure residual = {res_hist[-1]:.4f}")

start, end = np.mean(res_hist[:3]), np.mean(res_hist[-3:])
ok = end < 0.9 * start
print(f"[step3] closure residual: start={start:.4f} -> end={end:.4f}  ({100*(1-end/start):.0f}% drop)")
print(f"[step3] GATE: {'PASS — closure residual measurably falls (coarse dynamics close)' if ok else 'FAIL'}")
