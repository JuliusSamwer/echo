"""Draw what the policy is ACTUALLY doing: agent path, high-level subgoals, and repair corrections,
in maze coordinates, on top of the maze walls.

Inputs (all produced by the env-gated logging in repair/instrument.py + repair/seam.py):
  ZEUS_XY_CSV        runs/*_xy.csv         per (env, step): x, y, goal_x, goal_y
  ZEUS_PLAN_CSV      runs/*_plan.csv       per (env, segment, waypoint): x, y  [waypoint 1 = SUBGOAL]
  ZEUS_MAZE_NPZ      runs/*_maze.npz       per env: the binary maze layout
  ZEUS_REPAIR_XY_CSV runs/*_repair_xy.csv  per (env, segment): g_before/g_after/agent/goal  (optional)

Usage:
  python plot_rollout.py --xy runs/diag_xy.csv --plan runs/diag_plan.csv \
      --maze runs/diag_maze.npz --out figures/rollout            # all episodes
  python plot_rollout.py ... --env 16                            # one episode, large
  python plot_rollout.py --demo                                  # synthetic self-test of the plotting

ALIGNMENT: the agent's (x,y) frame and the maze grid indices are reconciled by `--align`, and the
result is CHECKED, not assumed — the script reports what fraction of logged agent positions land in
cells the layout calls walls. If that number isn't near zero, the mapping is wrong; try --swap-xy
and/or a different --cell/--origin. Never read a trajectory figure whose alignment check failed.
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

SUCC, FAIL, SUBGOAL, REPAIR = "#2a78d6", "#d03b3b", "#eda100", "#1baf7a"


# ------------------------------------------------------------------ loading
def load_xy(path):
    """env -> (T,2) agent path, (2,) goal."""
    paths, goals = defaultdict(list), {}
    for r in csv.DictReader(open(path)):
        e = int(r["env_idx"])
        paths[e].append((int(r["step"]), float(r["x"]), float(r["y"])))
        goals[e] = (float(r["goal_x"]), float(r["goal_y"]))
    out = {}
    for e, rows in paths.items():
        rows.sort(key=lambda t: t[0])
        # a run can contain more than one stage per env (main rollout + final transition);
        # keep the longest contiguous run of increasing steps = the main rollout.
        segs, cur = [], [rows[0]]
        for prev, nxt in zip(rows, rows[1:]):
            (cur.append(nxt) if nxt[0] > prev[0] else (segs.append(cur), cur := [nxt]))
        segs.append(cur)
        best = max(segs, key=len)
        out[e] = (np.array([[x, y] for _, x, y in best], float), np.array(goals[e], float))
    return out


def load_plan(path):
    """env -> {segment: {waypoint: (x,y)}}"""
    d = defaultdict(lambda: defaultdict(dict))
    if not path or not os.path.exists(path):
        return {}
    for r in csv.DictReader(open(path)):
        d[int(r["env_idx"])][int(r["segment"])][int(r["waypoint"])] = (float(r["x"]), float(r["y"]))
    return d


def load_repair(path):
    d = defaultdict(list)
    if not path or not os.path.exists(path):
        return {}
    for r in csv.DictReader(open(path)):
        if int(r.get("fired", 0)) != 1:
            continue
        d[int(r["env"])].append((int(r["seg"]),
                                 (float(r["g_before_x"]), float(r["g_before_y"])),
                                 (float(r["g_after_x"]), float(r["g_after_y"]))))
    return d


def load_mazes(path):
    if not path or not os.path.exists(path):
        return {}
    with np.load(path) as z:
        return {int(k.replace("env", "")): z[k] for k in z.files}


# ------------------------------------------------------------------ alignment
def grid_of(xy, maze, cell, origin, swap):
    """map maze coordinates -> (row, col) integer cells."""
    p = xy[:, ::-1] if swap else xy
    c = np.floor((p[:, 0] - origin[0]) / cell).astype(int)
    r = np.floor((p[:, 1] - origin[1]) / cell).astype(int)
    r = np.clip(r, 0, maze.shape[0] - 1); c = np.clip(c, 0, maze.shape[1] - 1)
    return r, c


def alignment_error(paths, mazes, cell, origin, swap):
    """fraction of logged agent positions that fall inside cells the layout calls WALL."""
    bad = tot = 0
    for e, (xy, _) in paths.items():
        m = mazes.get(e)
        if m is None:
            continue
        r, c = grid_of(xy, m, cell, origin, swap)
        bad += int((np.asarray(m)[r, c] > 0).sum()); tot += len(r)
    return (bad / tot) if tot else float("nan")


# ------------------------------------------------------------------ drawing
def draw_episode(ax, xy, goal, maze, plan, repairs, cell, origin, swap, title, failed):
    if maze is not None:
        m = np.asarray(maze)
        h, w = m.shape
        ext = ([origin[0], origin[0] + w * cell, origin[1] + h * cell, origin[1]] if not swap
               else [origin[1], origin[1] + h * cell, origin[0] + w * cell, origin[0]])
        ax.imshow(m, cmap="Greys", alpha=0.28, extent=ext, interpolation="nearest", zorder=0)

    # agent path, shaded by time so direction is readable without an arrow per step
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)),
                        linewidths=2.0, zorder=3)
    ax.add_collection(lc)
    ax.plot(*xy[0], "o", ms=7, color="#52514e", zorder=5)
    ax.plot(*xy[-1], "o", ms=8, mfc="none", mew=2, color=FAIL if failed else SUCC, zorder=5)
    ax.plot(*goal, "*", ms=17, color=SUCC, zorder=6)

    # where the high level POINTED at each replan: subgoal (waypoint 1), plus a hair to the agent
    if plan:
        for seg, wps in sorted(plan.items()):
            if 1 not in wps:
                continue
            sx, sy = wps[1]
            here = xy[min(seg * 4, len(xy) - 1)]
            ax.plot([here[0], sx], [here[1], sy], "-", lw=0.6, color=SUBGOAL, alpha=0.5, zorder=2)
            ax.plot(sx, sy, ".", ms=5, color=SUBGOAL, zorder=4)

    # what repair did to the subgoal, when it fired
    for seg, (bx, by), (ax_, ay) in repairs or []:
        ax.annotate("", xy=(ax_, ay), xytext=(bx, by), zorder=7,
                    arrowprops=dict(arrowstyle="->", color=REPAIR, lw=1.4, alpha=0.9))

    ax.set_title(title, fontsize=10, color=FAIL if failed else SUCC)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c3c2b7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy", default="runs/diag_xy.csv")
    ap.add_argument("--plan", default="runs/diag_plan.csv")
    ap.add_argument("--maze", default="runs/diag_maze.npz")
    ap.add_argument("--repair-xy", default="runs/diag_repair_xy.csv")
    ap.add_argument("--out", default="figures/rollout")
    ap.add_argument("--env", type=int, default=None, help="draw a single episode, large")
    ap.add_argument("--fail-thresh", type=float, default=0.3)
    ap.add_argument("--cell", type=float, default=1.0)
    ap.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--swap-xy", action="store_true")
    ap.add_argument("--demo", action="store_true", help="synthetic data — exercises the plotting only")
    a = ap.parse_args()

    if a.demo:
        paths, plans, mazes, repairs = _demo()
        print("DEMO MODE — synthetic trajectories. Not data. Do not report these figures.")
    else:
        paths = load_xy(a.xy); plans = load_plan(a.plan)
        mazes = load_mazes(a.maze); repairs = load_repair(a.repair_xy)

    if mazes:
        err = alignment_error(paths, mazes, a.cell, tuple(a.origin), a.swap_xy)
        verdict = "OK" if err < 0.05 else "MISALIGNED — fix --cell/--origin/--swap-xy before reading these"
        print(f"alignment check: {err:.1%} of agent positions fall in wall cells  [{verdict}]")
    else:
        print("no maze layouts supplied — drawing trajectories without walls")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    envs = [a.env] if a.env is not None else sorted(paths)
    fails = {e: float(np.linalg.norm(paths[e][0] - paths[e][1], axis=1).min()) >= a.fail_thresh
             for e in envs}

    if a.env is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        xy, goal = paths[a.env]
        draw_episode(ax, xy, goal, mazes.get(a.env), plans.get(a.env), repairs.get(a.env),
                     a.cell, tuple(a.origin), a.swap_xy,
                     f"episode {a.env} — {'FAILED' if fails[a.env] else 'reached goal'}", fails[a.env])
        out = f"{a.out}_env{a.env}.png"
    else:
        cols = 6; rows = int(np.ceil(len(envs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows))
        for ax, e in zip(np.ravel(axes), envs):
            xy, goal = paths[e]
            draw_episode(ax, xy, goal, mazes.get(e), plans.get(e), repairs.get(e),
                         a.cell, tuple(a.origin), a.swap_xy,
                         f"ep {e} — {'FAIL' if fails[e] else 'ok'}", fails[e])
        for ax in np.ravel(axes)[len(envs):]:
            ax.axis("off")
        out = f"{a.out}_all.png"

    fig.suptitle("agent path (dark→bright = time) · orange = subgoal the high level proposed"
                 + (" · green = repair moved it" if repairs else ""), fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def _demo():
    rng = np.random.default_rng(0)
    paths, plans, mazes, repairs = {}, {}, {}, {}
    for e in range(6):
        m = np.zeros((10, 10), int); m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 1
        m[4, 1:5] = 1; m[4, 6:9] = 1
        goal = np.array([7.5, 7.5])
        stuck = e % 2 == 0
        T = 136
        pos = np.array([2.5, 2.5]); xs = []
        for t in range(T):
            tgt = np.array([2.5, 6.0]) if stuck else goal
            pos = pos + 0.06 * (tgt - pos) + rng.normal(0, 0.02, 2)
            xs.append(pos.copy())
        xy = np.array(xs)
        paths[e] = (xy, goal)
        mazes[e] = m
        plans[e] = {s: {1: tuple(xy[min(s * 4 + 6, T - 1)] + rng.normal(0, 0.25, 2))}
                    for s in range(T // 4)}
        if e == 1:
            repairs[e] = [(s, tuple(xy[s * 4] + [0.4, 0.1]), tuple(xy[s * 4] + [0.9, 0.5]))
                          for s in range(0, T // 4, 6)]
    return paths, plans, mazes, repairs


if __name__ == "__main__":
    main()
