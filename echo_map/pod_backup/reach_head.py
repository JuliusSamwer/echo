"""Reachability / temporal-distance head R(z, g) on FROZEN JEPA latents — residual (c).

Gate experiment core (model-agnostic; the repo glue that produces latents is separate).
Supervision is free & hindsight: for states k steps apart on a trajectory, (z_t, z_{t+k}) is a
labeled example with answer k. No rewards, no demos, no env interaction.

R becomes the universal interface signal: trigger (replaces r_plan), repair gradient (differentiable
in g by construction), subgoal-quality score, optional low-level cost.

THE GATE: after training on hindsight pairs, does R recover the control-relevant metric that raw
latent distance (r_plan AUC=0.64) misses and the oracle (0.92) has?
  - R_metric_quality: on held-out pairs, does R(z_t,z_{t+k}) predict k?  (is the info IN the latent?)
  - R_failure_auc: does an R-based signal separate the diagnostic failing episodes?  (vs 0.64 / 0.92)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------- the head
class ReachHead(nn.Module):
    """R([z,g]) -> predicted temporal distance (regressed in log-steps for scale stability).
    Latents are flattened frozen obs-components (large); a linear proj keeps the MLP small."""
    def __init__(self, in_dim: int, proj: int = 256, hidden: int = 256):
        super().__init__()
        self.pz = nn.Linear(in_dim, proj)
        self.pg = nn.Linear(in_dim, proj)
        self.mlp = nn.Sequential(
            nn.Linear(proj * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, g):
        a, b = self.pz(z), self.pg(g)
        h = torch.cat([a, b, a - b], dim=-1)
        return self.mlp(h).squeeze(-1)          # predicted log(1+steps)


# ----------------------------------------------------------------- hindsight pairs
def make_pairs(trajs, k_max=32, neg_per_pos=1, rng=None):
    """trajs: list of (T, D) float32 latent arrays (one per trajectory).
    Returns (Z, G, Y): z_t, z_{t+k}, and target y = log(1+k). Negatives: random cross-traj pairs
    labeled at a 'far' distance (k_max*2) so R learns 'unreachable' is large."""
    rng = rng or np.random.default_rng(0)
    Z, G, Y = [], [], []
    far = float(np.log1p(k_max * 2))
    for tr in trajs:
        T = len(tr)
        if T < 2:
            continue
        for t in range(T - 1):
            k = rng.integers(1, min(k_max, T - 1 - t) + 1)
            Z.append(tr[t]); G.append(tr[t + k]); Y.append(float(np.log1p(k)))
            for _ in range(neg_per_pos):
                otr = trajs[rng.integers(len(trajs))]
                Z.append(tr[t]); G.append(otr[rng.integers(len(otr))]); Y.append(far)
    return (np.stack(Z).astype(np.float32), np.stack(G).astype(np.float32),
            np.array(Y, np.float32))


# ----------------------------------------------------------------- train / eval
def train_head(Z, G, Y, in_dim, epochs=30, bs=512, lr=1e-3, device="cpu", val_frac=0.15, seed=0):
    torch.manual_seed(seed)
    n = len(Z); idx = np.random.default_rng(seed).permutation(n)
    nv = int(n * val_frac); vi, ti = idx[:nv], idx[nv:]
    head = ReachHead(in_dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Zt = torch.tensor(Z, device=device); Gt = torch.tensor(G, device=device); Yt = torch.tensor(Y, device=device)
    for ep in range(epochs):
        head.train(); perm = np.random.default_rng(seed + ep).permutation(ti)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            pred = head(Zt[b], Gt[b])
            loss = ((pred - Yt[b]) ** 2).mean()
            loss.backward(); opt.step()
    # validation metric quality: rank corr + MAE (in steps) between R and true k
    head.eval()
    with torch.no_grad():
        pv = head(Zt[vi], Gt[vi]).cpu().numpy()
    yv = Y[vi]
    mae_steps = float(np.mean(np.abs(np.expm1(pv) - np.expm1(yv))))
    rho = _spearman(pv, yv)
    return head, dict(val_spearman=rho, val_mae_steps=mae_steps, n_train=len(ti), n_val=len(vi))


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


# ----------------------------------------------------------------- gate AUC
def auc(scores_fail, scores_succ):
    """P(fail signal > succ signal). >0.5 => higher signal on failures."""
    if len(scores_fail) == 0 or len(scores_succ) == 0:
        return float("nan")
    c = sum(1 for a in scores_fail for b in scores_succ if a > b)
    t = sum(1 for a in scores_fail for b in scores_succ if a == b)
    return (c + 0.5 * t) / (len(scores_fail) * len(scores_succ))


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Synthetic sanity: if latent == true 2D position (+noise), temporal distance IS recoverable
    # -> R should predict k well (high spearman).  If latent is RANDOM, it should NOT (rho~0).
    # This validates the harness detects the fork before touching real latents.
    rng = np.random.default_rng(1)
    def make_trajs(informative):
        trajs = []
        for _ in range(60):
            T = rng.integers(20, 40); pos = np.cumsum(rng.normal(0, 1, (T, 2)), axis=0)
            feat = np.concatenate([pos, rng.normal(0, 0.1, (T, 6))], axis=1) if informative \
                   else rng.normal(0, 1, (T, 8))
            trajs.append(feat.astype(np.float32))
        return trajs
    for name, inf in [("INFORMATIVE latent", True), ("RANDOM latent", False)]:
        Z, G, Y = make_pairs(make_trajs(inf), k_max=16)
        _, m = train_head(Z, G, Y, in_dim=8, epochs=25)
        print(f"{name:20} val_spearman(R vs true k)={m['val_spearman']:.3f}  mae_steps={m['val_mae_steps']:.2f}")
    print("expect: informative rho high (>0.6), random rho ~0 -> harness distinguishes recoverable vs not")
