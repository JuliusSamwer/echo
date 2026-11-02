"""Certification probe (the gate). Hand-place boundaries three ways, measure success
under the bounded executor. NO learning. Answers "does placement matter in this env?"
for near-zero cost, and doubles as the rung-3 performance ceiling (GT placement).

Three conditions, SAME subgoal count (= L) in all three:
  GT          : ordered doorway centers on the shortest route (known-optimal placement).
  RANDOM      : L uniformly-random free points.
  ADVERSARIAL : the L route-room centers (right route, wrong precise placement).
                Fix vs spec: spec used rooms[1:-1] = L-1 subgoals, breaking the paired
                test's "same count". rooms[1:] gives exactly L -> a fair paired diff.

Pass: success(GT) >= 0.8 AND GT-RANDOM and GT-ADVERSARIAL both strictly positive with
95% CIs excluding 0 (paired bootstrap over instances, rung-2 rigor).
"""
import numpy as np
from .config import MazeConfig
from .maze import BottleneckMaze
from .executor import BoundedMPPI, MPPIConfig


def _paired_bootstrap_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def _mean_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def _adversarial_subgoals(env, rooms):
    """L route-room centers: rooms has L+1 entries [start_room, ..., goal_room];
    rooms[1:] is exactly L points on the correct route but not at the doorways."""
    return [env.room_center(r) for r in rooms[1:]]


def run_probe(maze_cfg: MazeConfig, mppi_cfg: MPPIConfig,
              L_values=(2, 3, 4), n_instances=100, H=8, base_seed=0):
    results = {}   # L -> {condition: [0/1 per instance]}
    for L in L_values:
        succ = {"GT": [], "RANDOM": [], "ADVERSARIAL": []}
        for k in range(n_instances):
            cfg = MazeConfig(**{**maze_cfg.__dict__, "seed": base_seed + k})
            env = BottleneckMaze(cfg)
            rng = np.random.default_rng(base_seed + 10000 + k)
            env.reset(L=L, rng=rng)
            start, goal = env.state.copy(), env.goal.copy()
            rooms, gt_doors = env.path_doorways(start, goal)
            m = MPPIConfig(**{**mppi_cfg.__dict__, "max_step": cfg.max_step})
            # per-instance seeded executor RNG -> reproducible paired bootstrap
            ex = BoundedMPPI(env.step_batch, m, rng=np.random.default_rng(base_seed + 20000 + k))

            ok, *_ = ex.execute_chain(start, gt_doors, goal, H, cfg.subgoal_eps, cfg.goal_eps)
            succ["GT"].append(int(ok))

            rand_sg = [env.sample_free_point(rng) for _ in range(len(gt_doors))]
            ok, *_ = ex.execute_chain(start, rand_sg, goal, H, cfg.subgoal_eps, cfg.goal_eps)
            succ["RANDOM"].append(int(ok))

            adv_sg = _adversarial_subgoals(env, rooms)
            ok, *_ = ex.execute_chain(start, adv_sg, goal, H, cfg.subgoal_eps, cfg.goal_eps)
            succ["ADVERSARIAL"].append(int(ok))
        results[L] = succ
    return results


def summarize(results):
    all_pass = True
    for L, succ in results.items():
        gt_m, gt_ci = _mean_ci(succ["GT"])
        rd_m, rd_ci = _mean_ci(succ["RANDOM"])
        ad_m, ad_ci = _mean_ci(succ["ADVERSARIAL"])
        d_gr, ci_gr = _paired_bootstrap_diff(succ["GT"], succ["RANDOM"])
        d_ga, ci_ga = _paired_bootstrap_diff(succ["GT"], succ["ADVERSARIAL"])
        passed = (gt_m >= 0.8 and ci_gr[0] > 0 and ci_ga[0] > 0)
        all_pass = all_pass and passed
        print(f"L={L}: GT={gt_m:.2f}[{gt_ci[0]:.2f},{gt_ci[1]:.2f}]  "
              f"RAND={rd_m:.2f}[{rd_ci[0]:.2f},{rd_ci[1]:.2f}]  "
              f"ADV={ad_m:.2f}[{ad_ci[0]:.2f},{ad_ci[1]:.2f}]")
        print(f"       GT-RAND={d_gr:+.2f} [{ci_gr[0]:+.2f},{ci_gr[1]:+.2f}]   "
              f"GT-ADV={d_ga:+.2f} [{ci_ga[0]:+.2f},{ci_ga[1]:+.2f}]   "
              f"{'PASS' if passed else 'FAIL'}")
    return all_pass


# ---------- qualitative overlay ----------

def plot_overlay(env, start, goal, chains, path, title=""):
    """chains: {'GT': (subgoals, trajectory), 'ADVERSARIAL': (subgoals, trajectory)}.
    Draws walls + start/goal + subgoal chains + executed trajectories so you can SEE
    the adversarial chain stall against a wall while GT threads the doorways."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(chains), figsize=(5.2 * len(chains), 5.0), squeeze=False)
    for ax, (name, (subgoals, traj)) in zip(axes[0], chains.items()):
        for (x1, y1, x2, y2) in env.walls:
            ax.plot([x1, x2], [y1, y2], color="#333", lw=2)
        for c, _ in env.bottlenecks:
            ax.plot(*c, "s", color="#2ca02c", ms=5, alpha=0.5)   # doorways
        if len(traj):
            ax.plot(traj[:, 0], traj[:, 1], "-", color="#1f77b4", lw=1.6, alpha=0.9, label="executed")
        sg = np.asarray(subgoals, float)
        if len(sg):
            ax.plot(sg[:, 0], sg[:, 1], "x--", color="#d62728", ms=9, mew=2, lw=1, label="subgoals")
        ax.plot(*start, "o", color="#111", ms=9, label="start")
        ax.plot(*goal, "*", color="#ff7f0e", ms=15, label="goal")
        ax.set_title(name); ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.05, env.W + 0.05); ax.set_ylim(-0.05, env.H + 0.05)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig)
    return path
