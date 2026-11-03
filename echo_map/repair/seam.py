"""Phase-3 seam hook: env-gated subgoal repair at the two-level planner boundary.

Enabled only when ECHO_REPAIR_CONDITION in {repair_trigger, repair_always, hwm_replan, hwm_replan_every}.
No-op (returns None) otherwise -> baseline trajectories untouched. Mirrors the instrument pattern.

Called from TwoLvlPlanner.plan() AFTER l1_result exists (so z_T = l1_result.pred_obs[-1] is available).
Returns g_exec (bs, D) to reset the l1 target to, or None to keep the original subgoal.

Mechanism logging (ECHO_REPAIR_CSV): per replan-segment, r_plan before/after, #fired, compute.
"""
from __future__ import annotations
import atexit, csv, os
import torch
from . import repair_op as _rop
from . import control_loop as _cl

_EVENTS = []
_NCALL = 0          # per-(segment,env) mechanism records
_WROTE = False

def _meansq(a, b):
    a = _rop.flatten_conv_output(a).float(); b = _rop.flatten_conv_output(b).float()
    return ((a - b) ** 2).mean(dim=-1)          # (bs,)

def condition():
    return os.environ.get("ECHO_REPAIR_CONDITION", "")

def enabled():
    return condition() in ("repair_trigger", "repair_always", "hwm_replan", "hwm_replan_every")

def _cfg():
    return _rop.RepairConfig(
        alpha=float(os.environ.get("ECHO_ALPHA", "1.0")),
        beta=float(os.environ.get("ECHO_BETA", "1.0")),
        eta=float(os.environ.get("ECHO_ETA", "1e-2")),
        n_inner=int(os.environ.get("ECHO_NINNER", "10")),
        refresh_actions=os.environ.get("ECHO_REFRESH", "1") == "1",
    )

def _flush():
    global _WROTE
    p = os.environ.get("ECHO_REPAIR_CSV")
    if not p or _WROTE or not _EVENTS:
        return
    _WROTE = True
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_EVENTS[0].keys())); w.writeheader(); w.writerows(_EVENTS)
    print(f"[echo-repair] wrote {len(_EVENTS)} events -> {p}")

atexit.register(_flush)

_XY_ROWS = []
_XY_WROTE = False


def _probe_xy(planner, z):
    """latent (bs, ...) -> maze (x,y), exactly as mppi_planner probes its own predictions."""
    prober = getattr(planner, "prober", None)
    if prober is None:
        return None
    with torch.no_grad():
        loc = prober(z)
        norm = getattr(planner, "normalizer", None)
        if norm is not None and hasattr(norm, "unnormalize_location"):
            loc = norm.unnormalize_location(loc)
    a = loc.detach().cpu().float().reshape(loc.shape[0], -1)[:, :2]
    return [[float(v[0]), float(v[1])] for v in a]


def _flush_xy():
    global _XY_WROTE
    p = os.environ.get("ECHO_REPAIR_XY_CSV")
    if not p or _XY_WROTE or not _XY_ROWS:
        return
    _XY_WROTE = True
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_XY_ROWS[0].keys())); w.writeheader(); w.writerows(_XY_ROWS)
    print(f"[echo-repair-xy] wrote {len(_XY_ROWS)} rows -> {p}")


atexit.register(_flush_xy)


def _log_repair_xy(two_lvl, seg_idx, z_cur, g0, g_rep, z_T, z_goal, fired, path):
    """One row per (segment, env): where the agent is, where the subgoal was, where repair moved it,
    where the low level expects to land, and where the true goal sits — all in maze coordinates."""
    pl = two_lvl.l2_planner
    cur, before, after, land, goal = (_probe_xy(pl, z_cur), _probe_xy(pl, g0), _probe_xy(pl, g_rep),
                                      _probe_xy(pl, z_T), _probe_xy(pl, z_goal))
    if before is None:
        return                                   # no prober available -> nothing to log
    cond = os.environ.get("ECHO_REPAIR_CONDITION", "hwm") or "hwm"
    for j in range(len(before)):
        _XY_ROWS.append(dict(
            condition=cond, seg=int(seg_idx), env=j, fired=int(bool(fired[j])),
            agent_x=cur[j][0] if cur else float("nan"), agent_y=cur[j][1] if cur else float("nan"),
            g_before_x=before[j][0], g_before_y=before[j][1],
            g_after_x=after[j][0] if after else float("nan"),
            g_after_y=after[j][1] if after else float("nan"),
            l1_pred_x=land[j][0] if land else float("nan"),
            l1_pred_y=land[j][1] if land else float("nan"),
            goal_x=goal[j][0] if goal else float("nan"), goal_y=goal[j][1] if goal else float("nan"),
        ))


def maybe_repair(two_lvl, l2_result, l1_result, seg_idx=-1):
    """Return g_exec (bs, D) or None. Frozen models; only the subgoal g is optimized."""
    cond = condition()
    if cond not in ("repair_trigger", "repair_always", "hwm_replan", "hwm_replan_every"):
        return None
    g0 = l2_result.pred_obs[1].detach()                    # (bs, D)  current subgoal
    z_T = l1_result.pred_obs[-1].detach()                  # (bs, D)  frozen predicted terminal (a*)
    z_cur = l1_result.pred_obs[0].detach()                 # (bs, D)  current encoded state
    z_goal = two_lvl.l2_planner.objective.target_enc.detach()   # (bs, D) encoded goal
    tau = float(os.environ.get("ECHO_TAU", "1e9"))
    macros_remaining = l2_result.actions[:, 1:].detach()   # (bs, m-1, adim)
    cfg = _cfg()
    r_plan_before = _meansq(z_T, g0)                       # (bs,) bottom-up: reachability
    # top-down goal-progress signal (model-derived): how much closer to goal the subgoal is than now.
    # >0 => subgoal advances toward goal; <=tau_prog => misplaced/stalled => repair.
    goal_prog = _meansq(z_cur, z_goal) - _meansq(g0, z_goal)   # (bs,)

    # top-down term (v1): pull g directly toward the goal latent -> term2 = beta * d(g, z_goal).
    # (Full spec rolls F_high through remaining macros; that conv/action-fusion coupling is deferred.)
    def roll_high(g):
        return g

    # planning runs under no_grad -> repair needs grad explicitly enabled.
    with torch.enable_grad():
        g_rep = _rop.repair_subgoal(g0, z_T, roll_high, z_goal, cfg, replan_fn=None, model=None)  # (bs, D)
    r_plan_after_rep = _meansq(z_T, g_rep)                 # (bs,)

    bs = g0.shape[0]
    g_exec = g0.clone()
    fired = torch.zeros(bs, dtype=torch.bool)
    trig_mode = os.environ.get("ECHO_TRIGGER", "reach")    # reach (default) | progress | both
    tau_prog = float(os.environ.get("ECHO_TAU_PROG", "0.0"))
    if cond == "repair_always":
        g_exec = g_rep; fired[:] = True
    elif cond == "repair_trigger":
        reach_bad = r_plan_before > tau                    # bottom-up: unreachable
        prog_bad = goal_prog < tau_prog                    # top-down: not goal-directed / stalled
        if trig_mode == "progress":
            mask = prog_bad
        elif trig_mode == "both":
            mask = reach_bad | prog_bad                    # two-pressure: repair if EITHER fails
        else:                                              # "reach" = original behavior
            mask = reach_bad
        g_exec[mask] = g_rep[mask]; fired = mask
    # (hwm_replan / hwm_replan_every handled in M3.2)

    # --- additive, env-gated: the correction in MAZE COORDINATES (what repair actually did) -----
    # Probes the latents through the planner's own location prober, so g_before / g_after / the
    # agent's own position land in the same (x,y) frame as the logged rollout. Read-only.
    xyp = os.environ.get("ECHO_REPAIR_XY_CSV")
    if xyp:
        try:
            _log_repair_xy(two_lvl, seg_idx, z_cur, g0, g_rep, z_T, z_goal, fired, xyp)
        except Exception as e:
            print(f"[echo-repair-xy] WARNING xy log failed, run continues: {e}", file=__import__("sys").stderr, flush=True)

    # mechanism log
    csvp = os.environ.get("ECHO_REPAIR_CSV")
    if csvp:
        for j in range(bs):
            _EVENTS.append(dict(
                condition=cond, seg=int(seg_idx), env=j,
                r_plan_before=float(r_plan_before[j]),
                r_plan_after=float(r_plan_after_rep[j]) if fired[j] else float(r_plan_before[j]),
                goal_prog=float(goal_prog[j]),
                trig_mode=os.environ.get("ECHO_TRIGGER", "reach"),
                fired=int(bool(fired[j])), tau=tau, tau_prog=tau_prog,
                n_inner=cfg.n_inner, alpha=cfg.alpha, beta=cfg.beta, eta=cfg.eta,
            ))
    global _NCALL
    _NCALL += 1
    if _NCALL <= 8:
        import sys
        print(f"[echo-repair] call#{_NCALL} cond={cond} fired={int(fired.sum().item())}/{bs} "
              f"rplan_before={float(r_plan_before.mean()):.5f} rplan_after={float(r_plan_after_rep.mean()):.5f} "
              f"|dg|={float((g_rep-g0).abs().mean()):.3e}", file=sys.stderr, flush=True)
    if not fired.any():
        return None                                        # nothing changed -> caller keeps original
    return g_exec.detach()
