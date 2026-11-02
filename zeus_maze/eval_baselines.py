"""Rung-1 / rung-2 baselines ON THE BOTTLENECK MAZE — the comparison rung 3 must beat.

Model-based: both planners imagine rollouts in the FROZEN BASE and execute in the TRUE env.
Because the base is one-maze-specific (state-only obs), we hold the maze FIXED (the base's
training maze) and vary only (start, goal) — held-out pairs, exactly like rungs 0-2.

  FLAT (rung-1)  : single-level MPPI straight to the goal. Budget = (L+1)*H_hop total steps.
                   Claim: degrades as L grows (deceptive junctions defeat a fixed-horizon planner).
  HWM  (rung-2)  : learned coarse world-model (macro-action VAE + one-shot k-step dynamics) plans
                   subgoals; the low level reaches them. SAME total budget.
                   Claim: HWM >> FLAT at larger L — decomposition recovers long-horizon success.

Paired 95% CIs over the shared (start,goal) pairs (rung-2 rigor).

  python -m zeus_maze.eval_baselines            # L=2,3,4 x 80 pairs
  python -m zeus_maze.eval_baselines --smoke
"""
import sys
from pathlib import Path
import numpy as np
import torch

from .config import MazeConfig
from .maze import BottleneckMaze
from .executor import MPPIConfig
from .base_model import load_frozen_base
from .hwm import CoarseModel, collect_base_episodes, train_coarse
from .hwm_faithful import FlatController, HWMController, run_episode
from .probe import _mean_ci, _paired_bootstrap_diff

H_HOP = 12
K = 5   # shorter macros: a ~0.7-unit hop stays mostly intra-room, so the one-shot coarse
        # prediction is far more accurate -> subgoals land in reachable spots (k=8 was too hard)
BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
COARSE = Path(__file__).resolve().parents[1] / "models" / "hwm_coarse.pt"


def get_coarse(env, base_step, dev, smoke=False):
    if COARSE.exists():
        d = torch.load(COARSE, weights_only=False, map_location=dev)
        c = CoarseModel(k=d["k"], macro_dim=d["macro_dim"], act_scale=env.cfg.max_step).to(dev)
        c.load_state_dict(d["state_dict"]); c.eval()
        print(f"[coarse] loaded {COARSE}")
        return c
    print("[coarse] training coarse world-model on frozen-base rollouts ...")
    S_eps, A_eps = collect_base_episodes(env, base_step, n_ep=120 if smoke else 600, T=72, seed=0)
    c = CoarseModel(k=K, macro_dim=8, act_scale=env.cfg.max_step).to(dev)
    train_coarse(c, S_eps, A_eps, k=K, epochs=40 if smoke else 140, device=dev, verbose=True)
    torch.save({"state_dict": c.state_dict(), "k": K, "macro_dim": 8}, COARSE)
    print(f"[coarse] saved {COARSE}")
    return c


def sample_pairs(env, L, n, seed):
    rng = np.random.default_rng(seed); pairs = []
    while len(pairs) < n:
        env.reset(L=L, rng=rng)
        pairs.append((env.state.copy(), env.goal.copy()))
    return pairs


def run(smoke=False, dev="cpu"):
    model, cfg_d, base_step = load_frozen_base(BASE, device=dev)
    env = BottleneckMaze(MazeConfig(**cfg_d))          # the base's fixed training maze
    true_step = env.step_batch
    mppi = MPPIConfig(max_step=env.cfg.max_step, n_samples=384, n_iters=3, plan_horizon=14)
    # running_terminal cost (probe default): strong terminal pull -> the low level STOPS at a
    # target. "min" breaks precise subgoal-reaching (drifts past). Flat's myopia is the point.
    coarse = get_coarse(env, base_step, dev, smoke)

    L_values = (2, 3) if smoke else (2, 3, 4)
    n = 20 if smoke else 60
    mx = env.cfg.max_step
    print(f"\n[baselines] fixed maze · FAITHFUL HWM port · plan-in-base/act-in-world · {n} pairs/L")
    print(f"{'L':>3} | {'FLAT':>16} | {'HWM (learned)':>16} | HWM−FLAT (95% CI)")
    print("-" * 74)
    for L in L_values:
        pairs = sample_pairs(env, L, n, seed=100 + L)
        max_steps = 15 * (L + 2)                       # generous capability budget (rung-1 style)
        flat, hwm = [], []
        for i, (s, g) in enumerate(pairs):
            fc = FlatController(base_step, max_step=mx)
            flat.append(int(run_episode(true_step, fc, s, g, env.cfg.goal_eps, max_steps,
                                        np.random.default_rng(500 + i))))
            hc = HWMController(coarse, base_step, device=dev, max_step=mx, goal_eps=env.cfg.goal_eps)
            hwm.append(int(run_episode(true_step, hc, s, g, env.cfg.goal_eps, max_steps,
                                       np.random.default_rng(700 + i))))
        f_m, f_ci = _mean_ci(flat); h_m, h_ci = _mean_ci(hwm)
        d, ci = _paired_bootstrap_diff(hwm, flat)
        verdict = "HWM>FLAT" if ci[0] > 0 else ("FLAT>HWM" if ci[1] < 0 else "n.s.")
        print(f"{L:>3} | {f_m:>6.2f}[{f_ci[0]:.2f},{f_ci[1]:.2f}] | "
              f"{h_m:>6.2f}[{h_ci[0]:.2f},{h_ci[1]:.2f}] | {d:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  {verdict}")


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
