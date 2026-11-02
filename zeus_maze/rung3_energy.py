"""Rung 3 · ITEM 0 — the Danskin migration demo (the mechanism, before any learned F/G).

THE claim to make visible: subgoal boundaries are not placed by a rule or a label — they EMERGE by
gradient descent on a reachability energy. Initialise a subgoal chain at ROOM CENTERS (the certified-
FAIL placement: a room-center subgoal is not collinear with an off-center doorway, so the bounded low
level cannot thread it), then descend the chain energy. Watch the subgoals SLIDE to the doorways.

The atom is the reachability energy under the FROZEN fine dynamics P1:

    R(z, g) = min_{a_1:H} || roll_P1(z, a_1:H) - g ||         (Π = MPPI solves the min)

Gradient by DANSKIN / envelope theorem: at the optimum a*, ∂R/∂g (and ∂R/∂z through the rollout
Jacobian) is the partial of the residual with a* held FIXED — one backward pass, no differentiating
through the optimiser. Concretely: Π plans the hop (non-differentiable MPPI in P1) and returns a*; we
FREEZE a* and RE-ROLL it through the frozen ForwardModel in torch, which is differentiable in the hop's
start and end points. So a whole subgoal chain is differentiable and we can descend on it.

Teacher-forced energy (the smooth warm-up in the curriculum): consecutive subgoals must be mutually
reachable, so we minimise Σ_i R(g_i, g_{i+1}) over the interior points (endpoints = start, goal fixed).
Each shared subgoal is pulled to where BOTH its incoming hop lands and its outgoing hop can start — the
doorway between the two rooms. No wall/doorway coordinate appears in the loss.

  python -m zeus_maze.rung3_energy [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
import torch

from .config import MazeConfig
from .maze import BottleneckMaze
from .base_model import load_frozen_base
from .executor import BoundedMPPI, MPPIConfig
from .rung3_reinforce import make_lowlevel

BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
OUT = Path(__file__).resolve().parents[1] / "figures"
LOG = Path(__file__).resolve().parents[1] / "models" / "results"

LL_MPPI = dict(n_samples=384, n_iters=3, plan_horizon=14)
H_LO = 10            # the measured trust horizon (certified placement-critical band)


# ------------------------------------------------------------ Π : plan a hop, return the frozen a*
def plan_hop_actions(base_step, mx, z, g, H, eps, rng):
    """Π: receding-horizon MPPI in the FROZEN base from z toward g, for up to H steps. Returns the
    executed action sequence a*_1:h (the plan we FREEZE for Danskin) and the achieved endpoint. Plans
    AND rolls in the base (imagination under P1) -- this is the energy's own rollout, not the env."""
    cfg = MPPIConfig(max_step=mx, **LL_MPPI)
    ex = BoundedMPPI(base_step, cfg, rng=rng)
    s = np.asarray(z, float).copy()
    nominal = np.zeros((cfg.plan_horizon, 2)); acts = []
    for _ in range(H):
        a, nominal = ex._plan_action(s, np.asarray(g, float), nominal)
        acts.append(a.copy())
        s = ex._step_one(s, a)
        nominal = np.roll(nominal, -1, axis=0); nominal[-1] = 0.0
        if np.linalg.norm(s - g) < eps:
            break
    return np.asarray(acts, np.float32), s


# ------------------------------------------------------------ differentiable re-roll (a* frozen)
def roll_diff(model, z_t, a_star_t):
    """Re-roll the FROZEN plan a* through the frozen ForwardModel, differentiable in the start z_t.
    a* is a constant (Danskin), so this is exactly ∂(achieved)/∂z with the optimiser held fixed."""
    s = z_t.unsqueeze(0)
    for k in range(a_star_t.shape[0]):
        s = model(s, a_star_t[k].unsqueeze(0))
    return s.squeeze(0)


def hop_residual(model, base_step, mx, z_t, g_t, H, eps, rng, true_planner=None, n_avg=1):
    """R(z,g), REALITY-ANCHORED with a Danskin gradient.

    Our frozen base LEAKS through walls under multi-step MPPI (MPPI finds actions that thread the wall
    in imagination), so a base-only residual is ~0 even across a wall -> blind, no gradient. Fix: a
    STRAIGHT-THROUGH estimator -- the residual VALUE uses the TRUE-env achieved endpoint (correctly ~0.5
    at a room-center subgoal because the real low level gets stuck at the wall), while the GRADIENT flows
    through the differentiable base rollout Jacobian. So the energy SEES the wall, and its gradient pulls
    the subgoal toward where the hop actually gets stuck -- the doorway. (true_planner=None -> pure base,
    the old blind behavior, kept for ablation.) Returns (residual_tensor, achieved_np)."""
    a_star, base_ach = plan_hop_actions(base_step, mx,
                                        z_t.detach().numpy(), g_t.detach().numpy(), H, eps, rng)
    if len(a_star) == 0:
        return (z_t - g_t).norm(), z_t.detach().numpy()
    achieved = roll_diff(model, z_t, torch.tensor(a_star))          # differentiable in z_t (a* frozen)
    if true_planner is not None:
        zt, gt = z_t.detach().numpy(), g_t.detach().numpy()
        aches = [true_planner.reach(zt, gt, H, eps)[1] for _ in range(n_avg)]   # denoise stochastic reach
        true_ach = np.mean(aches, axis=0)
        achieved = achieved + (torch.tensor(true_ach, dtype=torch.float32) - achieved.detach())  # straight-through
        base_ach = true_ach
    return (achieved - g_t).norm(), np.asarray(base_ach)


# ------------------------------------------------------------ chain energy (teacher-forced Σ R)
def chain_energy(model, base_step, mx, chain, H, eps, rng, spacing_w=0.0, min_gap=0.0,
                 true_planner=None, n_avg=1):
    """chain: (K,2) torch, chain[0]=start and chain[-1]=goal are FIXED; interior points are free.
    L = Σ_i R(g_i, g_{i+1}) (+ optional spacing penalty). Returns (L, achieved_list)."""
    L = torch.zeros(())
    achieved = []
    for i in range(len(chain) - 1):
        r, ach = hop_residual(model, base_step, mx, chain[i], chain[i + 1], H, eps, rng,
                              true_planner, n_avg)
        L = L + r
        achieved.append(ach)
        if spacing_w > 0 and 0 < i < len(chain) - 1:      # gentle anti-collapse on interior gaps
            gap = (chain[i] - chain[i - 1]).norm()
            L = L + spacing_w * torch.relu(min_gap - gap)
    return L, achieved


# ------------------------------------------------------------ instance setup
def make_instance(env, L, seed):
    rng = np.random.default_rng(seed)
    env.reset(L=L, rng=rng)
    s, g = env.state.copy(), env.goal.copy()
    rooms, doors = env.path_doorways(s, g)
    return s, g, [np.asarray(d, float) for d in doors], rooms


def init_chain_room_centers(env, rooms, s, g):
    """Init interior subgoals at the ROOM CENTERS of the intermediate rooms on the path (the certified-
    FAIL placement). K points = start + centers(intermediate rooms) + goal."""
    interior = [env.room_center(r) for r in rooms[1:-1]]          # skip start & goal rooms
    pts = [np.asarray(s, float)] + interior + [np.asarray(g, float)]
    return np.asarray(pts, np.float32)


# ------------------------------------------------------------ execute a chain in the TRUE env (before/after)
def execute_success(base_step, true_step, mx, s, interior, g, eps, geps):
    ll = make_lowlevel(base_step, true_step, mx, np.random.default_rng(4242))
    ok, _ = ll.execute_chain(s, list(interior), g, H_LO, eps, geps)
    return bool(ok)


# ------------------------------------------------------------ junction lift of a point set
def points_lift(points, doors, env, tol=0.2):
    if not len(points):
        return 0.0, 0.0
    near = np.mean([min(np.linalg.norm(p - d) for d in doors) < tol for p in points])
    # base rate = fraction of random in-room points near a doorway (same tol)
    rng = np.random.default_rng(0)
    rand = np.mean([min(np.linalg.norm(env.sample_free_point(rng) - d) for d in doors) < tol
                    for _ in range(400)])
    return float(near), float(near / rand if rand > 0 else 0.0)


# ------------------------------------------------------------ the descent + figure
def run_migration(env, model, base_step, true_step, mx, L=3, seed=1, iters=60, lr=0.05,
                  eps=0.08, geps=0.08, smoke=False):
    s, g, doors, rooms = make_instance(env, L, seed)
    chain0 = init_chain_room_centers(env, rooms, s, g)
    K = len(chain0)
    print(f"[energy] instance L={L} seed={seed}: {K-2} interior slots @ room centers | doors={len(doors)}",
          flush=True)

    chain = torch.tensor(chain0.copy(), requires_grad=True)
    fixed_mask = torch.zeros(K, 2); fixed_mask[0] = 1; fixed_mask[-1] = 1   # freeze endpoints
    opt = torch.optim.Adam([chain], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.1)
    rng = np.random.default_rng(seed + 7)
    true_planner = make_lowlevel(base_step, true_step, mx, np.random.default_rng(4242))  # reality anchor
    n_avg = 1 if smoke else 3                 # denoise the stochastic true-env reach
    history = [chain0[1:-1].copy()]            # interior positions per iter (for the migration plot)
    losses = []
    for it in range(iters):
        opt.zero_grad()
        Lval, _ = chain_energy(model, base_step, mx, chain, H_LO, eps, rng,
                               spacing_w=0.5, min_gap=0.25, true_planner=true_planner, n_avg=n_avg)
        Lval.backward()
        with torch.no_grad():
            chain.grad = chain.grad * (1 - fixed_mask)       # never move start/goal
        opt.step(); sched.step()
        with torch.no_grad():                                # keep inside the arena
            chain.clamp_(1e-3, min(env.W, env.H) - 1e-3)
        history.append(chain.detach().numpy()[1:-1].copy())
        losses.append(float(Lval))
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            print(f"  [energy] it {it:3d} | chain_energy(Σ R)={float(Lval):.3f}", flush=True)

    interior0 = chain0[1:-1]
    interiorF = chain.detach().numpy()[1:-1]
    ok0 = execute_success(base_step, true_step, mx, s, interior0, g, eps, geps)
    okF = execute_success(base_step, true_step, mx, s, interiorF, g, eps, geps)
    near0, lift0 = points_lift(interior0, doors, env)
    nearF, liftF = points_lift(interiorF, doors, env)
    print(f"\n[energy] MIGRATION RESULT (L={L}, seed={seed}):", flush=True)
    print(f"   room-center init : execute_success={ok0}  | near-door frac={near0:.2f} lift={lift0:.2f}", flush=True)
    print(f"   after descent    : execute_success={okF}  | near-door frac={nearF:.2f} lift={liftF:.2f}", flush=True)
    print(f"   Σ R: {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)

    try:
        _plot(env, s, g, doors, history, interior0, interiorF, L, seed)
    except Exception as e:
        print(f"[energy] (plot skipped: {e})", flush=True)
    return dict(ok0=ok0, okF=okF, lift0=lift0, liftF=liftF, loss=losses,
                interior0=interior0, interiorF=interiorF, doors=doors)


def _plot(env, s, g, doors, history, interior0, interiorF, L, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for (x1, y1, x2, y2) in env.walls:                       # walls
        ax.plot([x1, x2], [y1, y2], color="k", lw=2)
    H = np.asarray(history)                                   # (iters+1, n_interior, 2)
    n = H.shape[1]
    for j in range(n):                                       # migration trails
        ax.plot(H[:, j, 0], H[:, j, 1], color="0.7", lw=1, zorder=1)
    ax.scatter([d[0] for d in doors], [d[1] for d in doors], marker="*", s=260,
               facecolor="gold", edgecolor="k", zorder=2, label="doorways (GT, unused by loss)")
    ax.scatter(interior0[:, 0], interior0[:, 1], c="crimson", s=90, zorder=3,
               label="init: room centers (fail)")
    ax.scatter(interiorF[:, 0], interiorF[:, 1], c="royalblue", s=90, zorder=4,
               label="after descent")
    ax.scatter([s[0]], [s[1]], c="green", s=120, marker="s", zorder=5, label="start")
    ax.scatter([g[0]], [g[1]], c="purple", s=120, marker="D", zorder=5, label="goal")
    ax.set_title(f"Rung-3 item 0: subgoals migrate room-centers → doorways (L={L})\n"
                 "pure energy descent, no labels in the loss")
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left")
    OUT.mkdir(exist_ok=True)
    p = OUT / f"rung3_energy_migration_L{L}_s{seed}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[energy] wrote migration figure -> {p}", flush=True)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    model, cfg_d, base_step = load_frozen_base(BASE)
    cfg = MazeConfig(**{**cfg_d, "geodesic_res": 24})
    env = BottleneckMaze(cfg); true_step = env.step_batch; mx = env.cfg.max_step
    LOG.mkdir(exist_ok=True)
    seeds = [1] if smoke else [1, 2, 3]
    outs = []
    for sd in seeds:
        outs.append(run_migration(env, model, base_step, true_step, mx, L=3, seed=sd,
                                   iters=25 if smoke else 100, lr=0.05, smoke=smoke))
    print("\n[energy] SUMMARY (room-center -> descent):")
    for sd, o in zip(seeds, outs):
        print(f"   seed {sd}: success {o['ok0']}->{o['okF']} | lift {o['lift0']:.2f}->{o['liftF']:.2f}",
              flush=True)
