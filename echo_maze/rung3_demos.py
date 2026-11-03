"""Rung 3 · emergent subgoal policy from EXPERT DEMOS + frozen dynamics (the data-efficient branch).

Pure REINFORCE (`rung3_reinforce.py`) is high-variance and won't move off ~0 here. This branch is
the robotics recipe: demos kill the variance, the FROZEN fine dynamics decide WHERE the subgoals go
to maximize planning, and (optionally) RL polishes.

  1. EXTRACT planning-optimal subgoals from optimal demos, using the frozen base P.
     For each expert trajectory we greedily segment it into the LONGEST chunks that stay
     reachable-in-one-hop under P (head straight at a candidate waypoint through the frozen model;
     reachable = it gets within eps within the low-level step budget H_lo). The break points are
     exactly where extending the hop would cross a wall — i.e. the doorways — but we never use the
     doorway LABELS to place them: the frozen dynamics' reachability decides. This yields, per demo,
     the subgoal set that makes planning easiest.

  2. BEHAVIOR-CLONE B on those (s, goal) -> displacement pairs. Supervised, low-variance, orders of
     magnitude fewer samples than REINFORCE, and ZERO true-env / low-level rollouts at train time
     (segmentation only imagines straight shots through the frozen base).

  3. (optional) POLISH with the planning/RL signal: warm-start REINFORCE from the BC policy.

Everything downstream (E=identity, P=frozen base, low-level MPC) is the SAME frozen stack as the
REINFORCE branch, and eval reuses the SAME rollout/evaluate — so success numbers are directly
comparable. Reference points: flat ~0.15, learned-HWM ~0.25, REINFORCE ~0.00, oracle ceiling ~0.85.

  python -m echo_maze.rung3_demos [--smoke] [--polish]
"""
import sys
from pathlib import Path
import numpy as np
import torch

from .config import MazeConfig
from .maze import BottleneckMaze
from .base_model import load_frozen_base
from .rung3_flavorB import ref_path
# Reuse the EXACT policy, frozen low level, rollout and eval from the REINFORCE branch so the
# comparison is apples-to-apples (only the way B is TRAINED differs).
from . import rung3_reinforce as R
from .rung3_reinforce import SubgoalPolicy, GeoGrid, evaluate, make_lowlevel

BASE = Path(__file__).resolve().parents[1] / "models" / "base_frozen.pt"
POLICY = Path(__file__).resolve().parents[1] / "models" / "rung3_bc_policy.pt"

H_LO = R.H_LO          # per-hop step budget = the certified single-hop band (10)
MAX_HOP = 1.6          # policy displacement bound; one room-crossing hop is up to ~1.4 (=H_LO*max_step)


# ------------------------------------------------------------ expert demos (free, optimal)
def gen_expert_demos(env, L_values, n_per_L, seed):
    """Optimal demos for free: BFS the room graph (oracle), thread the doorway centers, connect with
    straight segments. This is the shortest geometric route -- a valid expert trajectory. The doorway
    centers only shape the demo GEOMETRY (an optimal path threads doorways by definition); they are
    NOT used to place subgoals -- that's step 1's job, from the frozen dynamics alone."""
    rng = np.random.default_rng(seed); demos = []
    for L in L_values:
        got = 0
        while got < n_per_L:
            env.reset(L=L, rng=rng); s = env.state.copy(); g = env.goal.copy()
            rooms, doors = env.path_doorways(s, g)
            if not doors:
                continue
            path = ref_path(s, doors, g)
            if len(path) < 6:
                continue
            demos.append(dict(start=s, goal=g, path=path,
                              doors=[np.asarray(d, float) for d in doors], L=L))
            got += 1
    return demos


# ------------------------------------------------------------ one-hop reachability under the frozen P
# "Reachable in one hop" = the EXACT deployment planner (frozen MPC: plan in the base, ACT in the env)
# gets there within H_lo steps. This is the honest definition and makes every extracted subgoal
# executable-in-one-hop AT DEPLOYMENT by construction. Two subtleties learned the hard way:
#   * A raw straight-shot through the base is NOT a valid reachability oracle: pushed into a wall for
#     H_lo steps the smooth base LEAKS through it (its tiny per-step wall error accumulates), so it
#     calls far-through-wall waypoints "reachable". The bounded MPC does not.
#   * Planning AND acting in the base (pure imagination) is ALSO too optimistic: the base is a touch
#     rosy near walls, so imagination-reachable subgoals sit just past the true feasible frontier and
#     the deployment planner then misses them (measured: per-hop reach ~0.4 -> chained success ~0).
#     Acting in the env during segmentation closes that gap. It costs a BOUNDED number of frozen-planner
#     probe-rollouts per demo (binary search, ~L*log(hop) reaches) -- still orders of magnitude fewer
#     env steps than REINFORCE, and NO gradients-through-env / reward tuning. The dynamics MODEL still
#     decides WHERE (MPPI plans in the base); the env only advances the agent, exactly as at deploy.
def make_seg_planner(base_step, true_step, mx, seed=0):
    return make_lowlevel(base_step, true_step, mx, np.random.default_rng(seed))


def _reach_frac(planner, anchor, target, H_lo, eps, k):
    """Reach `target` k times with independent MPPI noise; return the fraction that arrived. A ROBUST
    reachability test -- the greedy frontier is stochastic (MPPI is noisy), and a subgoal reachable on
    only ONE lucky draw is a coin-flip at deployment (which uses a different seed). Requiring a strong
    majority keeps subgoals on the SOLID side of the frontier, so chained hops don't decay as 0.5^L."""
    return sum(planner.reach(anchor, np.asarray(target, float), H_lo, eps)[0] for _ in range(k)) / k


def segment_demo(planner, path, max_step, H_lo, eps, min_hop, k=3, need=1.0):
    """Greedily segment `path` into planning-optimal subgoals: from the current anchor, take the
    FARTHEST waypoint the frozen MPC ROBUSTLY reaches in one H_lo-step hop (>= `need` of `k` noisy
    attempts; binary search over the path index -- reachability is ~monotone in distance within a hop
    budget), make it a subgoal, repeat. The breaks land where extending the hop would need to thread a
    SECOND doorway -- i.e. at the junctions -- but placement comes only from the frozen dynamics, never
    from doorway labels. Returns the ordered subgoals EXCLUDING the goal (goal = terminal target)."""
    T = len(path); subs = []; a = 0
    budget = H_lo * max_step * 1.2          # can't reach farther than the hop budget anyway
    guard = 0
    while a < T - 1 and guard < 3 * T:
        guard += 1
        anchor = path[a]
        hi = a + 1                          # farthest index within the reach budget (upper search bound)
        while hi + 1 < T and np.linalg.norm(path[hi + 1] - anchor) <= budget:
            hi += 1
        lo, best = a + 1, a + 1             # a+1 (~one primitive step away) is trivially reachable
        while lo <= hi:                     # find the LARGEST robustly-reachable index in [a+1, hi]
            mid = (lo + hi) // 2
            if _reach_frac(planner, anchor, path[mid], H_lo, eps, k) >= need:
                best = mid; lo = mid + 1
            else:
                hi = mid - 1
        # snap tiny leading hops forward: if the farthest reachable barely moved, still commit to it
        if best >= T - 1:
            break                           # goal directly reachable: no more subgoals
        if np.linalg.norm(path[best] - anchor) < min_hop and best < T - 1:
            best = min(best + 1, T - 1)     # avoid a degenerate near-zero subgoal
            if best >= T - 1:
                break
        subs.append(path[best].copy()); a = best
    return subs


def _clip_disp(d, max_hop):
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d * np.minimum(1.0, (0.98 * max_hop) / (n + 1e-9))


def extract_all_subgoals(planner, demos, max_step, H_lo, eps, min_hop):
    """Segment every demo ONCE (the expensive frozen-MPC probing) and cache the subgoal list on it.
    Nested sweep subsets then just gather BC pairs from these cached subgoals -- no re-segmentation."""
    for i, dm in enumerate(demos):
        dm['subs'] = segment_demo(planner, dm['path'], max_step, H_lo, eps, min_hop)
    return [dm['subs'] for dm in demos]


def build_bc_dataset(demos, max_hop):
    """Turn (already-segmented) demos into supervised (state, goal)->displacement pairs.
    The states B is queried at during a rollout are [start, sub_1, ..., sub_last]; the matching
    targets are [sub_1, ..., sub_last, GOAL] (the last query aims the goal)."""
    S, G, D = [], [], []
    for dm in demos:
        anchors = [dm['start']] + dm['subs']
        targets = dm['subs'] + [dm['goal']]
        for s, t in zip(anchors, targets):
            S.append(s); G.append(dm['goal']); D.append(t - s)
    S = np.asarray(S, np.float32); G = np.asarray(G, np.float32)
    D = _clip_disp(np.asarray(D, np.float32), max_hop)
    return S, G, D


# ------------------------------------------------------------ behavior cloning
def behavior_clone(policy, S, G, D, epochs=400, lr=3e-3, bs=256, seed=0, verbose=True):
    """Supervised regression of B's mean displacement onto the extracted planning-optimal targets."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    St = torch.tensor(S); Gt = torch.tensor(G); Dt = torch.tensor(D)
    W = St.shape[0]; rng = np.random.default_rng(seed); nb = max(1, W // bs)
    for ep in range(epochs):
        perm = rng.permutation(W); acc = 0.0
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            pred = policy.mean_disp(St[idx], Gt[idx])
            loss = ((pred - Dt[idx]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); acc += loss.item()
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [bc] ep {ep:3d} | mse={acc/nb:.4f}", flush=True)
    return policy


# ------------------------------------------------------------ does the EXTRACTION localize at junctions?
def subgoal_localization(demos, all_subs, tol=0.2):
    """Sanity/thesis metric: are the EXTRACTED subgoals (placed by reachability, NOT labels) sitting
    at the doorways? lift = P(extracted subgoal near a doorway) / P(a random path point near one).
    >1 => the frozen dynamics recovered the junctions on their own."""
    def near_door(p, doors):
        return min(np.linalg.norm(p - d) for d in doors) < tol if doors else False
    n_sub = n_sub_door = 0; n_pt = n_pt_door = 0; counts = {}
    for dm, subs in zip(demos, all_subs):
        counts.setdefault(dm['L'], []).append(len(subs))
        for p in subs:
            n_sub += 1; n_sub_door += near_door(p, dm['doors'])
        for p in dm['path']:
            n_pt += 1; n_pt_door += near_door(p, dm['doors'])
    p_sub = n_sub_door / max(n_sub, 1); base = n_pt_door / max(n_pt, 1)
    mean_by_L = {L: float(np.mean(v)) for L, v in sorted(counts.items())}
    return dict(n_subgoals=n_sub, P_door_given_subgoal=p_sub, base_rate=base,
                lift=(p_sub / base if base > 0 else 0.0), mean_subgoals_by_L=mean_by_L)


# ------------------------------------------------------------ dynamics-corrected eval (B proposes, P disposes)
@torch.no_grad()
def snap_subgoal(planner, s, shat, H_lo, eps, fracs=(1.15, 1.0, 0.85, 0.7, 0.55)):
    """B's raw proposal shat can miss the (precision-critical) doorway by a hair -> the hop fails. Snap
    it to the FARTHEST point along B's proposed DIRECTION that the frozen planner actually reaches in
    one hop (test far->near, take the first that reaches). B decides the direction/scale; the frozen
    dynamics decide the exact executable point. No labels -- only the reachability signal we trust."""
    d = np.asarray(shat, float) - np.asarray(s, float)
    for f in fracs:
        cand = np.asarray(s, float) + f * d
        if planner.reach(np.asarray(s, float), cand, H_lo, eps)[0]:
            return cand
    return np.asarray(shat, float)      # nothing reached: fall back to the raw proposal


@torch.no_grad()
def evaluate_snapped(policy, base_step, true_step, env, cfg, mx, geogrid, L_values, n, seed=999):
    """Same eval as `evaluate`, but each hop snaps B's proposal to a reachable point under frozen P."""
    from .rung3_reinforce import sample_pairs, make_geo, cfg_W, cfg_H
    out = {}
    ll = make_lowlevel(base_step, true_step, mx, np.random.default_rng(4242))
    for L in L_values:
        ps = sample_pairs(env, L, n, seed=seed + L); ok = 0
        for (s0, g) in ps:
            s = np.asarray(s0, float).copy()
            for hop in range(3 * L + 4):
                st = torch.tensor(s, dtype=torch.float32); gt = torch.tensor(g, dtype=torch.float32)
                shat = s + policy.mean_disp(st, gt).numpy()
                shat = np.clip(shat, 1e-3, [cfg_W(cfg) - 1e-3, cfg_H(cfg) - 1e-3])
                subgoal = snap_subgoal(ll, s, shat, H_LO, cfg.subgoal_eps)
                _, s, _ = ll.reach(s, subgoal, H_LO, cfg.subgoal_eps)
                if np.linalg.norm(s - g) < cfg.goal_eps:
                    ok += 1; break
        out[L] = ok / len(ps)
    return out


# ------------------------------------------------------------ optional RL polish (BC bootstrap -> REINFORCE)
def polish_with_reinforce(policy, base_step, true_step, env, cfg, mx, geogrid, Ls, per_L,
                          updates, batch, seed=7):
    """Warm-start REINFORCE from the BC policy with a small exploration std (local polish)."""
    R.pool.clear(); R.geos.clear(); R.poolL.clear()
    for L in Ls:
        for (s0, g) in R.sample_pairs(env, L, per_L, seed=500 + L):
            R.pool.append((s0, g)); R.geos.append(R.make_geo(geogrid, g)); R.poolL.append(L)
    R.train(policy, base_step, true_step, env, R.geos, R.pool, cfg, mx,
            updates=updates, batch=batch, lr=1e-3, std0=0.25, std1=0.1, seed=seed)
    return policy


# ------------------------------------------------------------ main
if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    do_polish = "--polish" in sys.argv
    model, cfg_d, base_step = load_frozen_base(BASE)
    cfg = MazeConfig(**{**cfg_d, "geodesic_res": 24})
    env = BottleneckMaze(cfg); true_step = env.step_batch; mx = env.cfg.max_step
    Ls = (2, 3) if smoke else (2, 3, 4)
    eval_n = 12 if smoke else 40
    sweep_n = 10 if smoke else 20
    eps = cfg.subgoal_eps; min_hop = 0.25

    print("[demos] precomputing fine-geodesic adjacency (for eval shaping) ...", flush=True)
    geogrid = GeoGrid(env, res=30 if smoke else 40)

    # ---- big demo pool (fixed seed) so the data-efficiency subsets are nested ----
    pool_per_L = 20 if smoke else 80
    demos = gen_expert_demos(env, Ls, pool_per_L, seed=100)
    print(f"[demos] generated {len(demos)} optimal demos ({pool_per_L}/L).", flush=True)

    # ---- step 1: extract planning-optimal subgoals ONCE, under the frozen P (no labels used) ----
    print("[demos] extracting planning-optimal subgoals via the frozen MPC (imagined under P) ...",
          flush=True)
    seg_planner = make_seg_planner(base_step, true_step, mx, seed=0)
    all_subs = extract_all_subgoals(seg_planner, demos, mx, H_LO, eps, min_hop)
    loc = subgoal_localization(demos, all_subs)
    print(f"[demos] EXTRACTED-subgoal localization: lift={loc['lift']:.2f} "
          f"(P(door|subgoal)={loc['P_door_given_subgoal']:.2f} base={loc['base_rate']:.2f}) "
          f"| mean #subgoals by L={loc['mean_subgoals_by_L']} (ideal ~= L-1 hops-1)", flush=True)

    # ---- data-efficiency sweep: BC on N demos/L, eval success vs L ----
    Ns = [2, 5, 10] if smoke else [2, 5, 10, 20, 40, 80]
    Ns = [n for n in Ns if n <= pool_per_L]
    print(f"\n[demos] DATA-EFFICIENCY SWEEP (BC), eval n={sweep_n}/L  "
          f"| refs: flat~0.15 HWM~0.25 REINFORCE~0.00 oracle~0.85", flush=True)
    header = "  demos/L | " + " | ".join(f"L={L}" for L in Ls) + " | #BC pairs"
    print(header + "\n  " + "-" * (len(header) - 2), flush=True)
    best_policy = None
    for N in Ns:
        subset = [d for L in Ls for d in [x for x in demos if x['L'] == L][:N]]
        S, G, D = build_bc_dataset(subset, MAX_HOP)
        policy = SubgoalPolicy(max_hop=MAX_HOP)
        behavior_clone(policy, S, G, D, epochs=200 if smoke else 400, verbose=False)
        res = evaluate(policy, base_step, true_step, env, cfg, mx, geogrid, Ls, n=sweep_n, seed=999)
        row = f"  {N:7d} | " + " | ".join(f"{res[L]:.2f}" for L in Ls) + f" | {len(S)}"
        print(row, flush=True)
        best_policy = policy      # last (largest N) is the one we keep

    # ---- final eval of the full-data BC policy at the reporting n, and save ----
    res = evaluate(best_policy, base_step, true_step, env, cfg, mx, geogrid, Ls, n=eval_n, seed=999)
    print(f"\n[demos] FINAL BC policy (all {Ns[-1]} demos/L), eval n={eval_n}/L:", flush=True)
    for L in Ls:
        print(f"   L={L}: success={res[L]:.2f}", flush=True)
    torch.save({"state_dict": best_policy.state_dict(), "max_hop": MAX_HOP}, POLICY)
    print(f"[demos] saved BC policy -> {POLICY}", flush=True)

    # ---- step 3 (optional): RL polish on top of BC ----
    if do_polish:
        print("\n[demos] POLISH: warm-starting REINFORCE from the BC policy ...", flush=True)
        polish_with_reinforce(best_policy, base_step, true_step, env, cfg, mx, geogrid, Ls,
                              per_L=20 if smoke else 40,
                              updates=40 if smoke else 120, batch=16 if smoke else 24)
        res = evaluate(best_policy, base_step, true_step, env, cfg, mx, geogrid, Ls, n=eval_n, seed=999)
        print(f"[demos] AFTER POLISH, eval n={eval_n}/L:", flush=True)
        for L in Ls:
            print(f"   L={L}: success={res[L]:.2f}", flush=True)
        torch.save({"state_dict": best_policy.state_dict(), "max_hop": MAX_HOP}, POLICY)
