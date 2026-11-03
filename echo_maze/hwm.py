"""Rung-2 baseline on the bottleneck maze: HWM-style LEARNED hierarchy over the frozen base.

Everything here plans in the agent's learned world (the frozen base) and ACTS in the true env
(model-based control). Two levels:

  low level  — ModelPlanner: receding-horizon MPPI whose imagined rollouts use the frozen base;
               the chosen action is executed in the TRUE env. Used both by the flat baseline
               (one shot to the goal) and as the HWM low level (hop to a subgoal).
  high level — CoarseModel: a macro-action VAE (E_a) + one-shot k-step coarse dynamics (G),
               fixed stride k. This is the BIASED baseline on purpose — it carries a fixed
               stride AND a predictive-horizon training objective; rung 3 later aims to match
               it with LESS bias. HWMController plans over macro-actions to pick a subgoal.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .executor import BoundedMPPI, MPPIConfig


def _mlp(sizes, act=nn.SiLU):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])


# ============================================================ closed-loop low level
class ModelPlanner:
    """Plan against `base_step` (imagined rollouts in the frozen base), execute with
    `true_step` (the real env). This is the honest world-model-planning loop."""
    def __init__(self, base_step, true_step, cfg: MPPIConfig, rng=None):
        self.ex = BoundedMPPI(base_step, cfg, rng=rng)     # provides _plan_action over the base
        self.true_step = true_step
        self.cfg = cfg

    def reach(self, s, subgoal, H, eps):
        s = np.asarray(s, float).copy()
        nominal = np.zeros((self.cfg.plan_horizon, 2))
        for step in range(H):
            a, nominal = self.ex._plan_action(s, np.asarray(subgoal, float), nominal)
            s = self.true_step(s[None], a[None])[0]          # ACT in the true env
            nominal = np.roll(nominal, -1, axis=0); nominal[-1] = 0.0
            if np.linalg.norm(s - subgoal) < eps:
                return True, s, step + 1
        return False, s, H

    def execute_chain(self, start, subgoals, goal, H, subgoal_eps, goal_eps):
        s = np.asarray(start, float).copy(); used = 0
        targets = list(subgoals) + [goal]
        for gi, g in enumerate(targets):
            eps = goal_eps if gi == len(targets) - 1 else subgoal_eps
            reached, s, st = self.reach(s, g, H, eps)
            used += st
            if not reached:
                return False, used
        return True, used


# ============================================================ coarse world-model
class CoarseModel(nn.Module):
    """m = E_a(a_{t:t+k});  s_{t+k} = s_t + G(s_t, m).  VAE macro space (KL to N(0,I)) so the
    high level can sample macros on-manifold. Trained on base rollouts (predictive-horizon)."""
    def __init__(self, k=8, macro_dim=8, hidden=256, act_scale=0.14):
        super().__init__()
        self.k = k; self.macro_dim = macro_dim
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))
        self.register_buffer("macro_std", torch.ones(macro_dim))
        self.E_a = _mlp([k * 2, hidden, hidden, 2 * macro_dim])
        self.D_a = _mlp([macro_dim, hidden, hidden, k * 2])
        self.G = _mlp([2 + macro_dim, hidden, hidden, 2])

    def encode_macro_dist(self, a_chunk):
        flat = (a_chunk / self.act_scale).reshape(*a_chunk.shape[:-2], self.k * 2)
        mu, logvar = self.E_a(flat).chunk(2, dim=-1)
        return mu, logvar.clamp(-8, 8)

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_macro(self, m):
        return self.D_a(m).reshape(*m.shape[:-1], self.k, 2) * self.act_scale

    def coarse_step(self, s, m):
        return s + self.G(torch.cat([s, m], dim=-1))

    def coarse_rollout(self, s0, macros):
        B = macros.shape[0]
        s = s0.expand(B, -1) if s0.dim() == 2 and s0.shape[0] == 1 else \
            (s0.unsqueeze(0).expand(B, -1) if s0.dim() == 1 else s0)
        outs = []
        for i in range(macros.shape[1]):
            s = self.coarse_step(s, macros[:, i, :]); outs.append(s)
        return torch.stack(outs, dim=1)


def collect_base_episodes(env, base_step, n_ep=500, T=72, seed=0, directed_frac=0.5):
    """Roll the FROZEN BASE from random room starts. A `directed_frac` of episodes drive in a
    (piecewise) STRAIGHT heading — this puts DIRECTED room-crossing / doorway-threading motion
    in the data, which pure OU random walks lack. Without it the coarse model only sees small
    undirected net displacements and cannot represent the directed hops the high level needs."""
    rng = np.random.default_rng(seed)
    S_eps, A_eps = [], []
    for e in range(n_ep):
        s = env.sample_free_point(rng); S = [s.copy()]; A = []
        directed = rng.random() < directed_frac
        ang = rng.uniform(0, 2 * np.pi); a = np.zeros(2)
        for t in range(T):
            if directed:
                if t % 10 == 0:                      # occasionally re-pick a heading
                    ang = rng.uniform(0, 2 * np.pi)
                base = np.array([np.cos(ang), np.sin(ang)]) * env.cfg.max_step
                a = base + rng.normal(0, 0.25 * env.cfg.max_step, 2)   # straight + jitter
            else:
                a = 0.7 * a + 0.3 * rng.normal(0, env.cfg.max_step, 2)  # OU wander
            s = base_step(s[None], a[None])[0]
            S.append(s.copy()); A.append(a.copy())
        S_eps.append(np.asarray(S, np.float32)); A_eps.append(np.asarray(A, np.float32))
    return S_eps, A_eps


def train_coarse(coarse, S_eps, A_eps, k, epochs=120, bs=512, lr=1e-3,
                 w_kl=0.02, w_rec=1.0, device="cpu", seed=0, verbose=False):
    """Predictive-horizon training: sufficiency (predict s_{t+k}) + VAE (KL + recon)."""
    S0, AF, SK = [], [], []
    for S, A in zip(S_eps, A_eps):
        T = A.shape[0]
        for t in range(T - k + 1):
            S0.append(S[t]); AF.append(A[t:t + k]); SK.append(S[t + k])
    S0 = torch.tensor(np.asarray(S0), device=device)
    AF = torch.tensor(np.asarray(AF), device=device)
    SK = torch.tensor(np.asarray(SK), device=device)
    opt = torch.optim.Adam(coarse.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    rng = np.random.default_rng(seed); W = S0.shape[0]; nb = max(1, W // bs)
    for ep in range(epochs):
        perm = rng.permutation(W); acc = 0.0
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            s0, af, sk = S0[idx], AF[idx], SK[idx]
            mu, logvar = coarse.encode_macro_dist(af); m = coarse.reparam(mu, logvar)
            pred = coarse.coarse_step(s0, m)
            suf = ((pred - sk) ** 2).mean()
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(-1).mean()
            rec = F.l1_loss(coarse.decode_macro(m), af)
            loss = suf + w_kl * kl + w_rec * rec
            opt.zero_grad(); loss.backward(); opt.step(); acc += suf.item()
        sched.step()
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [coarse] ep {ep:3d} suf_mse={acc/nb:.3e}", flush=True)
    coarse.macro_std.fill_(1.0)
    return coarse


# ============================================================ HWM two-level controller
class HWMController:
    def __init__(self, coarse: CoarseModel, planner: ModelPlanner, env, device="cpu",
                 M=6, N_hi=256, hi_iters=6, H_lo=12, subgoal_eps=0.12, goal_eps=0.08,
                 min_hop=0.3):
        self.coarse = coarse; self.planner = planner; self.env = env; self.dev = device
        self.M = M; self.N_hi = N_hi; self.hi_iters = hi_iters
        self.H_lo = H_lo; self.k = coarse.k
        self.subgoal_eps = subgoal_eps; self.goal_eps = goal_eps
        self.min_hop = min_hop; self.md = coarse.macro_dim

    @torch.no_grad()
    def _coarse_plan(self, s, goal, rng):
        """CEM over macro-actions using the coarse model, then a REACHABILITY FILTER: the coarse
        model PROPOSES first-step subgoals (from the goal-directed elite macros), but we keep only
        those reachable from s in a straight shot (no wall between) and pick the reachable one
        closest to the goal. This rejects the coarse model's optimistic through-wall subgoals and
        naturally biases placement toward the doorways. Falls back to the closest if none pass."""
        from .maze import _first_hit_t
        s = np.asarray(s, np.float32)
        s0 = torch.tensor(s, device=self.dev).reshape(1, -1)
        gt = torch.tensor(np.asarray(goal, np.float32), device=self.dev)
        mean = np.zeros((self.M, self.md), np.float32); std = np.ones((self.M, self.md), np.float32)
        n_elite = max(4, int(0.1 * self.N_hi))
        elite = None
        for _ in range(self.hi_iters):
            eps = rng.standard_normal((self.N_hi, self.M, self.md)).astype(np.float32)
            macros = mean[None] + std[None] * eps
            way = self.coarse.coarse_rollout(s0, torch.tensor(macros, device=self.dev))  # (N,M,2)
            cost = torch.linalg.norm(way - gt.view(1, 1, -1), dim=-1).min(dim=1).values.cpu().numpy()
            order = np.argsort(cost); elite = macros[order[:n_elite]]
            mean = elite.mean(0); std = np.maximum(elite.std(0), 0.3) + 1e-4
        # candidate subgoals = first coarse step of each elite macro
        m0 = torch.tensor(elite[:, 0, :], device=self.dev)
        subs = self.coarse.coarse_step(s0.expand(m0.shape[0], -1), m0).cpu().numpy()  # (E,2)
        sarr = np.repeat(s[None], len(subs), 0).astype(float)
        blocked = np.isfinite(_first_hit_t(sarr, subs.astype(float), self.env.wA, self.env.wB))
        hop = np.linalg.norm(subs - s, axis=1)
        dgoal = np.linalg.norm(subs - goal, axis=1)
        ok = (~blocked) & (hop > self.min_hop)
        if ok.any():
            cand = np.where(ok)[0]; return subs[cand[np.argmin(dgoal[cand])]]
        return subs[np.argmin(dgoal)]   # fallback: nothing reachable, take closest-to-goal

    def run(self, start, goal, total_budget, rng):
        """Alternate: coarse-plan a subgoal, low-level reach it in the true env, repeat until the
        goal is reached or the shared step budget is spent. Returns (success, steps_used)."""
        s = np.asarray(start, float).copy(); used = 0
        while used < total_budget:
            if np.linalg.norm(s - goal) < self.goal_eps:
                return True, used
            subgoal = self._coarse_plan(s, goal, rng)
            H = min(self.H_lo, total_budget - used)
            # aim at goal directly on the last legs (subgoal ~ goal region)
            tgt = goal if np.linalg.norm(subgoal - goal) < self.subgoal_eps else subgoal
            eps = self.goal_eps if tgt is goal else self.subgoal_eps
            reached, s, st = self.planner.reach(s, tgt, H, eps)
            used += st
            if st == 0:
                break
        return bool(np.linalg.norm(s - goal) < self.goal_eps), used
