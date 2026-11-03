"""Closed-loop bidirectional hierarchy — the headline mechanism (ladder step 2).

Two-level HWM scaffold, everything FROZEN (base P1, low-level MPPI). The one change vs a standard
top-down hierarchy: the low-level planner's REACHABILITY RESIDUAL (how far execution fell short of the
proposed subgoal) flows back UP and repairs the plan mid-episode.

  TOP-DOWN (open-loop, the HWM baseline): the high level plans the whole subgoal chain ONCE from the
      start; the low level aims each subgoal in sequence with a fixed per-hop budget H. A shortfall on
      one hop is IGNORED — the next hop is aimed from wherever we actually landed at the NEXT planned
      subgoal, so error compounds. Degrades as the horizon (L) grows.

  CLOSED-LOOP (ours): after each hop, read the residual r = ||landed - subgoal||. It reports whether the
      hop reached its subgoal; the high level re-derives the immediate next subgoal from the ACTUAL
      landed state (residual-triggered repair). A missed doorway is simply retried from where we are,
      instead of skipping ahead. Recovers the horizon curve.

High level = ORACLE doorway subgoals (env room path) so the curve isolates the CLOSED-LOOP mechanism,
not subgoal-placement quality. Low level = frozen ModelPlanner (plan in base, act in true env).

Headline figure: success vs L, top-down degrades / closed-loop stays flat.  This is HWM's own named
failure mode (stale/invalid subgoals under a growing horizon).

  python -m echo_maze.closed_loop [--smoke]
"""
import sys
from pathlib import Path
import numpy as np

from .config import MazeConfig
from .maze import BottleneckMaze
from .executor import MPPIConfig
from .base_model import load_frozen_base
from .hwm import ModelPlanner
from .probe import _mean_ci, _paired_bootstrap_diff

BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
OUT = Path(__file__).resolve().parents[1] / "figures"
LOG = Path(__file__).resolve().parents[1] / "models" / "results"
LL_MPPI = dict(n_samples=384, n_iters=3, plan_horizon=14)


# ------------------------------------------------------------ high level (oracle)
def high_level_plan(env, state, goal):
    """Oracle subgoal chain: doorway centers on the shortest ROOM path from state's room to goal's.
    Re-derivable from ANY state (that's what makes repair possible)."""
    rooms, doors = env.path_doorways(state, goal)
    return [np.asarray(d, float) for d in doors] if doors else []


# ------------------------------------------------------------ low level (frozen)
def make_lowlevel(base_step, true_step, mx, rng):
    cfg = MPPIConfig(max_step=mx, **LL_MPPI)
    return ModelPlanner(base_step, true_step, cfg, rng=rng)


# ------------------------------------------------------------ TOP-DOWN (open-loop) — the HWM baseline
def run_open_loop(planner, env, start, goal, H, eps, geps):
    """Plan the subgoal chain ONCE from start; aim each in sequence; NEVER repair. Returns (success,
    steps, residuals)."""
    subgoals = high_level_plan(env, start, goal)          # committed once
    targets = subgoals + [goal]
    s = np.asarray(start, float).copy(); steps = 0; resid = []
    for i, tgt in enumerate(targets):
        e = geps if i == len(targets) - 1 else eps
        _, s, used = planner.reach(s, tgt, H, e)
        steps += used
        resid.append(float(np.linalg.norm(s - tgt)))       # the upward signal (ignored here)
    return bool(np.linalg.norm(s - goal) < geps), steps, resid


# ------------------------------------------------------------ CLOSED-LOOP — residual-triggered repair
def run_closed_loop(planner, env, start, goal, H, eps, geps, tau, budget_hops):
    """After each hop, re-derive the next subgoal from the ACTUAL landed state. The residual r triggers
    the repair: if the hop fell short (r>tau) the next subgoal is re-planned from here (retry the missed
    doorway) instead of skipping ahead. Returns (success, steps, n_repairs)."""
    s = np.asarray(start, float).copy(); steps = 0; repairs = 0
    for _ in range(budget_hops):
        if np.linalg.norm(s - goal) < geps:
            return True, steps, repairs
        subgoals = high_level_plan(env, s, goal)           # REPLAN from current state (repair)
        tgt = subgoals[0] if subgoals else goal
        e = geps if not subgoals else eps
        prev_room = env.which_room(s)
        _, s, used = planner.reach(s, tgt, H, e)
        steps += used
        r = float(np.linalg.norm(s - tgt))                 # reachability residual = upward signal
        if r > tau and env.which_room(s) == prev_room:     # fell short AND didn't cross -> a repair
            repairs += 1
    return bool(np.linalg.norm(s - goal) < geps), steps, repairs


# ------------------------------------------------------------ sweep + figure
def sample_pairs(env, L, n, seed):
    rng = np.random.default_rng(seed); pairs = []
    while len(pairs) < n:
        env.reset(L=L, rng=rng); pairs.append((env.state.copy(), env.goal.copy()))
    return pairs


def run(smoke=False, H=8):
    model, cfg_d, base_step = load_frozen_base(BASE)
    env = BottleneckMaze(MazeConfig(**cfg_d)); true_step = env.step_batch; mx = env.cfg.max_step
    eps, geps = env.cfg.subgoal_eps, env.cfg.goal_eps
    tau = 1.5 * eps                                        # "fell short" threshold on the residual
    Ls = (2, 3, 4) if not smoke else (2, 3)
    n = 15 if smoke else 60
    print(f"[closed_loop] fixed maze · frozen 2-level · oracle high level · H_hop={H} · {n} pairs/L", flush=True)
    print(f"{'L':>3} | {'TOP-DOWN (open)':>16} | {'CLOSED-LOOP':>16} | CL−OL (95% CI)     | repairs/ep")
    print("-" * 82)
    rows = {}
    for L in Ls:
        pairs = sample_pairs(env, L, n, seed=100 + L)
        budget_hops = 3 * L + 4
        ol, cl, reps = [], [], []
        for i, (s, g) in enumerate(pairs):
            pol = make_lowlevel(base_step, true_step, mx, np.random.default_rng(500 + i))
            ok_o, _, _ = run_open_loop(pol, env, s, g, H, eps, geps)
            pcl = make_lowlevel(base_step, true_step, mx, np.random.default_rng(500 + i))
            ok_c, _, rp = run_closed_loop(pcl, env, s, g, H, eps, geps, tau, budget_hops)
            ol.append(int(ok_o)); cl.append(int(ok_c)); reps.append(rp)
        o_m, o_ci = _mean_ci(ol); c_m, c_ci = _mean_ci(cl)
        d, ci = _paired_bootstrap_diff(cl, ol)
        verdict = "CL>OL" if ci[0] > 0 else ("OL>CL" if ci[1] < 0 else "n.s.")
        rows[L] = dict(o=o_m, o_ci=o_ci, c=c_m, c_ci=c_ci, d=d, ci=ci, rep=float(np.mean(reps)))
        print(f"{L:>3} | {o_m:>6.2f}[{o_ci[0]:.2f},{o_ci[1]:.2f}] | {c_m:>6.2f}[{c_ci[0]:.2f},{c_ci[1]:.2f}] "
              f"| {d:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  {verdict} | {np.mean(reps):.1f}", flush=True)
    try:
        _plot(rows, Ls, H)
    except Exception as e:
        print(f"[closed_loop] (plot skipped: {e})", flush=True)
    return rows


def _plot(rows, Ls, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = list(Ls)
    o = [rows[L]["o"] for L in Ls]; oe = [[rows[L]["o"] - rows[L]["o_ci"][0] for L in Ls],
                                          [rows[L]["o_ci"][1] - rows[L]["o"] for L in Ls]]
    c = [rows[L]["c"] for L in Ls]; ce = [[rows[L]["c"] - rows[L]["c_ci"][0] for L in Ls],
                                          [rows[L]["c_ci"][1] - rows[L]["c"] for L in Ls]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(Ls, o, yerr=oe, marker="o", lw=2, capsize=4, color="crimson", label="top-down (open-loop) — HWM baseline")
    ax.errorbar(Ls, c, yerr=ce, marker="s", lw=2, capsize=4, color="royalblue", label="closed-loop (residual-triggered repair)")
    ax.set_xlabel("horizon  L  (doorways to cross)"); ax.set_ylabel("success rate")
    ax.set_xticks(Ls); ax.set_ylim(0, 1.02)
    ax.set_title(f"Bidirectional hierarchy: closed-loop recovers the horizon curve\n(frozen models, oracle high level, H_hop={H})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    OUT.mkdir(exist_ok=True)
    p = OUT / "closed_loop_horizon.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[closed_loop] wrote {p}", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
