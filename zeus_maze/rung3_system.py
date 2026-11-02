"""Rung 3 · the FULL emergent system, end-to-end (test the whole architecture, then debug step by step).

Frozen: E (identity), P1 (base fine dynamics), Π (bounded MPPI in P1, acts in env). Learned: G (coarse
dynamics = achieved-endpoint model), F (proposer). Energy: R via G. Everything goal-conditioned, reward-
free, latent (= state here).

Pipeline:
  1. COLLECT Π triples (z, g, achieved): run the frozen low level trying to hop z->g in the TRUE env,
     record where it LANDS. This is reality — across a wall it lands stuck, not at g.
  2. TRAIN G:(z,g)->achieved on those triples. G is the differentiable, WALL-AWARE surrogate of one hop.
     R_ψ(z,g) := ||G(z,g) - g|| is the reachability energy (0 iff the hop lands on g).
  3. CLOSED-LOOP LOSS (the design's L): chain g_1..g_M; ẑ_i = G(ẑ_{i-1}, g_i) (achieved, hop starts where
     the last LANDED); L = -Σ log σ((ε-||ẑ_i - g_i||)/temp) - log σ(goal term) + β·M_eff. G makes it
     differentiable in every g_i, so the goal term pulls subgoals to where hops actually succeed (doorways).
  4a. LOOP-1 (descent): per (start,goal), descend L wrt the g's directly. Tests loss+G with no F.
  4b. LOOP-2 (amortize): train F:(z_start,z_goal)->chain to MINIMIZE E[L(F(...))] through G (amortized
      optimisation — not regression to targets, so no multi-modal-target collapse).
  5. INFERENCE: F proposes a chain; execute hop-by-hop with Π in the TRUE env; success vs L.

  python -m zeus_maze.rung3_system [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .config import MazeConfig
from .maze import BottleneckMaze
from .base_model import load_frozen_base
from .rung3_reinforce import make_lowlevel

BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
MDL = Path(__file__).resolve().parents[1] / "models"
TRIPLES = MDL / "rung3_G_triples.npz"
G_CKPT = MDL / "rung3_G.pt"
F_CKPT = MDL / "rung3_F.pt"

H_LO = 10
LL_MPPI = dict(n_samples=384, n_iters=3, plan_horizon=14)


# ============================================================ 1. collect Π triples
def collect_triples(env, base_step, true_step, mx, n, seed, hop_lo=0.3, hop_hi=1.6):
    """(z, g, achieved): z a free point, g = z + random displacement in [hop_lo,hop_hi], achieved = where
    the frozen low level LANDS trying to reach g in the true env. Covers within-room (reachable) AND
    across-wall (lands stuck) — the wall nonlinearity G must learn."""
    rng = np.random.default_rng(seed)
    ll = make_lowlevel(base_step, true_step, mx, rng)
    Z, Gg, A = [], [], []
    while len(Z) < n:
        z = env.sample_free_point(rng)
        ang = rng.uniform(0, 2 * np.pi); r = rng.uniform(hop_lo, hop_hi)
        g = z + r * np.array([np.cos(ang), np.sin(ang)])
        g = np.clip(g, 1e-3, [env.W - 1e-3, env.H - 1e-3])
        _, ach, _ = ll.reach(z, g, H_LO, env.cfg.subgoal_eps)
        Z.append(z); Gg.append(g); A.append(ach)
        if len(Z) % 250 == 0:
            print(f"  [triples] {len(Z)}/{n}", flush=True)
    return (np.asarray(Z, np.float32), np.asarray(Gg, np.float32), np.asarray(A, np.float32))


# ============================================================ 2. G : coarse dynamics (achieved endpoint)
class CoarseG(nn.Module):
    """achieved = z + f(z, g, g-z). Residual form: starts as 'go straight to g' only if f learns it; the
    net must learn the wall clamp (achieved != g across a wall). Wall-aware BY DATA (trained on Π reality)."""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z, g):
        x = torch.cat([z, g, g - z], dim=-1)
        return z + self.net(x)


def train_G(Z, Gg, A, epochs=300, bs=512, lr=1e-3, device="cpu", seed=0, verbose=True):
    Zt, Gt, At = (torch.tensor(x, device=device) for x in (Z, Gg, A))
    G = CoarseG().to(device)
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    rng = np.random.default_rng(seed); W = Zt.shape[0]; nb = max(1, W // bs)
    for ep in range(epochs):
        perm = rng.permutation(W); acc = 0.0
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            pred = G(Zt[idx], Gt[idx])
            loss = ((pred - At[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); acc += loss.item()
        sch.step()
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [G] ep {ep:3d} mse={acc/nb:.4e}", flush=True)
    return G


@torch.no_grad()
def check_G(G, env, base_step, true_step, mx, n=400, seed=77):
    """G accuracy split by whether the straight z->g crosses a wall (the hard, load-bearing case)."""
    from .maze import _first_hit_t
    Z, Gg, A = collect_triples(env, base_step, true_step, mx, n, seed)
    pred = G(torch.tensor(Z), torch.tensor(Gg)).numpy()
    err = np.linalg.norm(pred - A, axis=1)
    crossed = np.isfinite(_first_hit_t(Z, Gg, env.wA, env.wB))
    print(f"[check G] achieved-MAE  open={err[~crossed].mean():.3f}  across-wall={err[crossed].mean():.3f} "
          f"(n_open={(~crossed).sum()} n_wall={crossed.sum()})", flush=True)
    return err[crossed].mean()


# ============================================================ 2b. EXACT differentiable wall-aware hop
# Isolates "is the architecture sound?" from "can G be learned?". A torch collision rollout: head straight
# from z to g, stop just before any wall, capped by the reach budget. Differentiable in z,g. This is the
# agent's dynamics knowing physics (collision) -- NOT doorway labels in the loss. It is STRICTER than the
# real maneuvering low level (can't thread a door to a point past it), so its minima are subgoals AT the
# doorways -- the certified-good placement, which the real executor then reaches easily.
def torch_first_hit_frac(P, P2, wA, wB):
    r = P2 - P; s = wB - wA                                   # (N,2),(M,2)
    rxs = r[:, 0:1] * s[None, :, 1] - r[:, 1:2] * s[None, :, 0]        # (N,M)
    qp = wA[None, :, :] - P[:, None, :]
    t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / (rxs + 1e-12)
    u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / (rxs + 1e-12)
    valid = (rxs.abs() > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return torch.where(valid, t, torch.ones_like(t)).min(dim=1).values      # (N,) first-hit fraction


class CollisionHop:
    """achieved(z,g): straight hop, stop just before the first wall, capped by the reach budget."""
    def __init__(self, env, max_travel, margin=0.03):
        self.wA = torch.tensor(env.wA, dtype=torch.float32)
        self.wB = torch.tensor(env.wB, dtype=torch.float32)
        self.max_travel = float(max_travel); self.margin = margin

    def __call__(self, z, g):
        single = (z.dim() == 1)
        if single:
            z = z[None]; g = g[None]
        d = g - z; dist = d.norm(dim=-1, keepdim=True) + 1e-9
        reach = torch.clamp(torch.tensor(self.max_travel) / dist, max=1.0)     # budget cap
        wall_t = torch_first_hit_frac(z, g, self.wA, self.wB)[:, None]         # first wall
        wall_frac = torch.clamp(wall_t - self.margin / dist, min=0.0)
        alpha = torch.minimum(reach, wall_frac)
        out = z + alpha * d
        return out[0] if single else out


# ============================================================ 3. closed-loop log-success loss (via G)
def closed_loop_loss(G, start, goal, interior, eps, temp, beta, delta):
    """interior: (M,2) subgoals (free). targets = interior then goal. ẑ chains through G (achieved).
    L = -Σ log σ((ε-||ẑ_i-target_i||)/temp) + β·M_eff.  Differentiable in interior."""
    z = start
    targets = torch.cat([interior, goal[None]], 0)          # (M+1,2)
    logsucc = torch.zeros(())
    for i in range(targets.shape[0]):
        g = targets[i]
        z = G(z.unsqueeze(0), g.unsqueeze(0)).squeeze(0)     # achieved endpoint
        d = (z - g).norm()
        logsucc = logsucc + torch.log(torch.sigmoid((eps - d) / temp) + 1e-6)
    # M_eff: number of DISTINCT hops among the interior (collapsed slots don't count)
    steps = torch.cat([(interior[0] - start)[None], interior[1:] - interior[:-1]], 0)
    m_eff = torch.sigmoid((steps.norm(dim=-1) - delta) / (0.5 * delta)).sum()
    return -logsucc + beta * m_eff


# ============================================================ 4a. LOOP-1 descent (test loss+G, no F)
def descend_chain(G, start, goal, M, iters, lr, eps, temp, beta, delta, seed, arena,
                  n_restarts=5, temp_hi=0.4, noise=0.15):
    """Descend the closed-loop loss wrt interior subgoals. MULTI-INIT (restarts w/ noise) + TEMP ANNEAL
    (smooth high-temp gradients early -> sharp success late): the sharp reachability landscape has narrow
    gap basins and bad wall-face local minima, so a single straight-line descent stalls. Return the
    lowest-final-loss chain across restarts."""
    tline = torch.linspace(0, 1, M + 2)[1:-1]
    base = start[None] * (1 - tline[:, None]) + goal[None] * tline[:, None]
    best, best_L = None, float("inf")
    for r in range(n_restarts):
        g = torch.Generator().manual_seed(seed * 100 + r)
        interior = (base + noise * torch.randn(M, 2, generator=g)).clamp(1e-3, arena - 1e-3)
        interior = interior.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([interior], lr=lr)
        for it in range(iters):
            tp = temp_hi + (temp - temp_hi) * (it / max(iters - 1, 1))     # anneal high -> low
            opt.zero_grad()
            L = closed_loop_loss(G, start, goal, interior, eps, tp, beta, delta)
            L.backward(); opt.step()
            with torch.no_grad():
                interior.clamp_(1e-3, arena - 1e-3)
        with torch.no_grad():
            Lf = float(closed_loop_loss(G, start, goal, interior, eps, temp, beta, delta))
        if Lf < best_L:
            best_L, best = Lf, interior.detach().clone()
    return best


# ============================================================ 4b. F : proposer, amortize the descent
class ProposerF(nn.Module):
    """(z_start, z_goal) -> M interior subgoals (positions in arena coords)."""
    def __init__(self, M, arena, hidden=256):
        super().__init__()
        self.M = M; self.arena = arena
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * M),
        )

    def forward(self, start, goal):
        x = torch.cat([start, goal], dim=-1)
        return torch.sigmoid(self.net(x)).reshape(*start.shape[:-1], self.M, 2) * self.arena


def train_F(F, G, pairs, arena, epochs=300, bs=64, lr=1e-3, eps=0.12, temp=0.05,
            beta=0.3, delta=0.3, seed=0, verbose=True):
    """Amortized: F proposes chains, minimize E[closed_loop_loss(G, F(start,goal))]. G frozen."""
    if isinstance(G, nn.Module):
        for p in G.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(F.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    S = torch.tensor(np.array([p[0] for p in pairs]), dtype=torch.float32)
    Gl = torch.tensor(np.array([p[1] for p in pairs]), dtype=torch.float32)
    rng = np.random.default_rng(seed); W = len(pairs); nb = max(1, W // bs)
    for ep in range(epochs):
        perm = rng.permutation(W); acc = 0.0
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            interior = F(S[idx], Gl[idx])                    # (B,M,2)
            L = torch.stack([closed_loop_loss(G, S[j], Gl[j], F(S[j:j+1], Gl[j:j+1])[0],
                                              eps, temp, beta, delta) for j in idx]).mean()
            opt.zero_grad(); L.backward(); opt.step(); acc += float(L)
        sch.step()
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [F] ep {ep:3d} closed_loop_L={acc/nb:.3f}", flush=True)
    return F


# ============================================================ 5. execution / eval in the TRUE env
def dedup_chain(interior, delta=0.25):
    """Drop collapsed slots (a subgoal within delta of the previous kept one)."""
    out = []
    for p in interior:
        if not out or np.linalg.norm(p - out[-1]) > delta:
            out.append(p)
    return out


def eval_success(propose_fn, env, base_step, true_step, mx, L_values, n, seed=999):
    """propose_fn(start,goal)->interior np array. Execute [interior]+[goal] with Π in the true env."""
    eps, geps = env.cfg.subgoal_eps, env.cfg.goal_eps
    ll = make_lowlevel(base_step, true_step, mx, np.random.default_rng(4242))
    out = {}
    for L in L_values:
        rng = np.random.default_rng(seed + L); ok = 0
        for _ in range(n):
            env.reset(L=L, rng=rng); s, g = env.state.copy(), env.goal.copy()
            interior = dedup_chain(propose_fn(s, g))
            done, _ = ll.execute_chain(s, interior, g, H_LO, eps, geps)
            ok += bool(done)
        out[L] = ok / n
    return out


def sample_pairs(env, L_values, n_per_L, seed):
    rng = np.random.default_rng(seed); pairs = []
    for L in L_values:
        got = 0
        while got < n_per_L:
            env.reset(L=L, rng=rng)
            pairs.append((env.state.copy(), env.goal.copy(), L)); got += 1
    return pairs


# ============================================================ main
if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    model, cfg_d, base_step = load_frozen_base(BASE)
    cfg = MazeConfig(**cfg_d)
    env = BottleneckMaze(cfg); true_step = env.step_batch; mx = env.cfg.max_step
    arena = float(min(env.W, env.H))
    Ls = (2, 3) if smoke else (2, 3, 4)
    M = 5
    eps, temp, beta, delta = 0.12, 0.05, 0.3, 0.3

    exact_g = "--exact-g" in sys.argv
    if exact_g:
        print("[system] EXACT-G mode: differentiable collision hop (architecture test, no learned G)",
              flush=True)
        G = CollisionHop(env, max_travel=H_LO * mx)
        def _skip(*a, **k): pass
        # jump straight to descent + F using the exact hop
        Z = Gg = A = None
    # 1+2. triples + G (cache triples in the repo)
    n_tri = 400 if smoke else 3000
    if exact_g:
        pass
    elif TRIPLES.exists() and not smoke:
        d = np.load(TRIPLES); Z, Gg, A = d["Z"], d["Gg"], d["A"]
        print(f"[system] loaded {len(Z)} cached triples", flush=True)
    else:
        print(f"[system] collecting {n_tri} Π triples (true-env reaches) ...", flush=True)
        Z, Gg, A = collect_triples(env, base_step, true_step, mx, n_tri, seed=1)
        if not smoke:
            np.savez(TRIPLES, Z=Z, Gg=Gg, A=A)
    if not exact_g:
        print("[system] training G (coarse achieved-endpoint model) ...", flush=True)
        G = train_G(Z, Gg, A, epochs=150 if smoke else 300)
        torch.save(G.state_dict(), G_CKPT)
        check_G(G, env, base_step, true_step, mx, n=200 if smoke else 400)

    # 4a. LOOP-1 descent eval (tests loss+G, independent of F)
    def descent_propose(s, g):
        return descend_chain(G, torch.tensor(s, dtype=torch.float32), torch.tensor(g, dtype=torch.float32),
                             M, iters=120, lr=0.05, eps=eps, temp=temp, beta=beta, delta=delta,
                             seed=0, arena=arena).numpy()
    print("\n[system] LOOP-1 DESCENT eval (loss+G, no F) ...", flush=True)
    res_desc = eval_success(descent_propose, env, base_step, true_step, mx, Ls, n=10 if smoke else 30)
    for L in Ls:
        print(f"   descent L={L}: success={res_desc[L]:.2f}", flush=True)

    # 4b. train F to amortize, then eval
    pairs = sample_pairs(env, Ls, 30 if smoke else 120, seed=100)
    print(f"\n[system] training F on {len(pairs)} pairs (amortize closed-loop through G) ...", flush=True)
    F = ProposerF(M, arena)
    train_F(F, G, pairs, arena, epochs=100 if smoke else 250, eps=eps, temp=temp, beta=beta, delta=delta)
    torch.save(F.state_dict(), F_CKPT)

    @torch.no_grad()
    def F_propose(s, g):
        return F(torch.tensor(s, dtype=torch.float32), torch.tensor(g, dtype=torch.float32)).numpy()
    print("\n[system] F (amortized) eval ...", flush=True)
    res_F = eval_success(F_propose, env, base_step, true_step, mx, Ls, n=10 if smoke else 30)

    print("\n[system] RESULT (success vs L | refs: flat~0.15 HWM~0.25 REINFORCE~0 BC~0.05 oracle~0.85):")
    print("   L | DESCENT(loop1) | F(amortized)")
    for L in Ls:
        print(f"   {L} |     {res_desc[L]:.2f}      |    {res_F[L]:.2f}", flush=True)
