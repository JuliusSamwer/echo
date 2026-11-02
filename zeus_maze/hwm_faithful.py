"""Faithful port of the grid-stack HWM strategy onto the bottleneck maze + frozen base.

Mirrors zeus/planning/mpc.py (CEM `plan` + RecedingHorizonController stuck-escape) and
zeus/planning/hwm_hierarchical.py (HWMController) — the recipe that got HWM 63%->20% on the
old maze. Key elements we had dropped and now restore:
  * low-level CEM with cost_mode="min" (reach-ever; robust to wall detours),
  * receding horizon with warm-start AND stuck-escape (cold-start + exploration burst on stall)
    — "what converts deadlock failures into successes",
  * RAW first coarse subgoal (no reachability filter),
  * generous subgoal_eps, replan the subgoal on reach OR timer>=k.
Everything plans against the frozen base (imagined) and ACTS in the true env (closed loop).
"""
from __future__ import annotations
import numpy as np
import torch

from .hwm import CoarseModel  # reuse the predictive coarse model + its training


def _clip_l2(a, max_step):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a * np.minimum(1.0, max_step / (n + 1e-9))


def cem_plan(base_step, s, target, H, N, cem_iters, max_step,
             init_mean=None, explore_scale=1.0, elite_frac=0.1, cost_mode="min", rng=None):
    """CEM sampling MPC against the base. Returns (actions[H,2], best_resid).
    cost = min-over-horizon distance to target ("did the rollout EVER reach it?")."""
    rng = rng or np.random.default_rng()
    s = np.asarray(s, float)
    mean = np.zeros((H, 2), np.float32) if init_mean is None else init_mean.astype(np.float32).copy()
    std = np.full((H, 2), 0.5 * max_step * explore_scale, np.float32)
    n_elite = max(2, int(elite_frac * N))
    best = np.inf
    for _ in range(cem_iters):
        eps = rng.standard_normal((N, H, 2)).astype(np.float32)
        samples = _clip_l2(mean[None] + std[None] * eps, max_step)
        cur = np.repeat(s[None], N, 0).astype(np.float32)
        mind = np.full(N, np.inf)
        for t in range(H):
            cur = base_step(cur, samples[:, t, :])
            mind = np.minimum(mind, np.linalg.norm(cur - target, axis=1))
        term = np.linalg.norm(cur - target, axis=1)
        cost = mind if cost_mode == "min" else term
        best = min(best, float(cost.min()))
        elite = samples[np.argsort(cost)[:n_elite]]
        mean = elite.mean(0)
        std = np.maximum(elite.std(0), 0.1 * max_step) + 1e-4
    return _clip_l2(mean, max_step), best


def _warm(actions):
    return np.vstack([actions[1:], np.zeros((1, 2), np.float32)])


class FlatController:
    """Receding-horizon CEM to the goal, with warm-start + stuck-escape. The rung-1 flat baseline."""
    def __init__(self, base_step, H=18, N=256, cem_iters=3, max_step=0.14,
                 patience=5, escape_explore=2.5, escape_window=3, progress_tol=0.03):
        self.base_step = base_step; self.H = H; self.N = N; self.it = cem_iters
        self.max_step = max_step; self.patience = patience
        self.escape_explore = escape_explore; self.escape_window = escape_window
        self.tol = progress_tol; self.reset()

    def reset(self):
        self._mean = None; self._best = np.inf; self._stall = 0; self._escape = 0

    def act(self, s, goal, rng):
        d = float(np.linalg.norm(np.asarray(s) - np.asarray(goal)))
        if d < self._best - self.tol:
            self._best = d; self._stall = 0
        else:
            self._stall += 1
        if self._stall >= self.patience and self._escape == 0:
            self._escape = self.escape_window; self._mean = None; self._stall = 0
        explore = self.escape_explore if self._escape > 0 else 1.0
        if self._escape > 0:
            self._escape -= 1
        actions, _ = cem_plan(self.base_step, s, np.asarray(goal, float), self.H, self.N, self.it,
                              self.max_step, init_mean=self._mean, explore_scale=explore, rng=rng)
        self._mean = _warm(actions)
        return actions[0]


class HWMController:
    """High level plans over macro-actions (raw first coarse subgoal); low level = min-cost CEM to
    the subgoal, warm-started; replan the subgoal on reach (generous eps) or timer>=k."""
    def __init__(self, coarse: CoarseModel, base_step, device="cpu",
                 M=6, N_hi=256, hi_iters=6, H_lo=None, N_lo=256, lo_iters=3,
                 max_step=0.14, subgoal_eps=0.2, goal_eps=0.08):
        self.coarse = coarse; self.base_step = base_step; self.dev = device
        self.M = M; self.N_hi = N_hi; self.hi_iters = hi_iters
        self.H_lo = H_lo if H_lo is not None else coarse.k + 6
        self.N_lo = N_lo; self.lo_iters = lo_iters
        self.max_step = max_step; self.k = coarse.k
        self.subgoal_eps = subgoal_eps; self.goal_eps = goal_eps
        self.md = coarse.macro_dim; self.reset()

    def reset(self):
        self._subgoal = None; self._timer = 0; self._low_mean = None; self._hi_mean = None

    @torch.no_grad()
    def _coarse_plan(self, s, goal, rng):
        s0 = torch.tensor(np.asarray(s, np.float32), device=self.dev).reshape(1, -1)
        gt = torch.tensor(np.asarray(goal, np.float32), device=self.dev)
        mean = self._hi_mean if self._hi_mean is not None else np.zeros((self.M, self.md), np.float32)
        std = np.ones((self.M, self.md), np.float32)
        n_elite = max(2, int(0.1 * self.N_hi))
        for _ in range(self.hi_iters):
            eps = rng.standard_normal((self.N_hi, self.M, self.md)).astype(np.float32)
            macros = mean[None] + std[None] * eps
            way = self.coarse.coarse_rollout(s0, torch.tensor(macros, device=self.dev))
            cost = torch.linalg.norm(way - gt.view(1, 1, -1), dim=-1).min(dim=1).values.cpu().numpy()
            elite = macros[np.argsort(cost)[:n_elite]]
            mean = elite.mean(0); std = np.maximum(elite.std(0), 0.3) + 1e-4
        self._hi_mean = np.vstack([mean[1:], np.zeros((1, self.md), np.float32)])
        m0 = torch.tensor(mean[0], device=self.dev).reshape(1, -1)
        return self.coarse.coarse_step(s0, m0)[0].cpu().numpy()   # RAW first coarse subgoal

    def act(self, s, goal, rng):
        reached = self._subgoal is not None and np.linalg.norm(np.asarray(s) - self._subgoal) < self.subgoal_eps
        if self._subgoal is None or reached or self._timer >= self.k:
            self._subgoal = self._coarse_plan(s, goal, rng)
            self._low_mean = None; self._timer = 0
        else:
            self._timer += 1
        actions, _ = cem_plan(self.base_step, s, self._subgoal, self.H_lo, self.N_lo, self.lo_iters,
                              self.max_step, init_mean=self._low_mean, rng=rng)
        self._low_mean = _warm(actions)
        return actions[0]


def run_episode(true_step, ctrl, start, goal, goal_eps, max_steps, rng):
    """Closed loop: controller plans against the base (inside `ctrl`), act in the TRUE env."""
    s = np.asarray(start, float).copy(); ctrl.reset()
    for _ in range(max_steps):
        a = ctrl.act(s, goal, rng)
        s = true_step(s[None], a[None])[0]
        if np.linalg.norm(s - goal) < goal_eps:
            return True
    return False
