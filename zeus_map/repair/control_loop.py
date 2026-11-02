"""Phase 3.2/3.3 — per-segment control loop + the six conditions.

Given, at a replan step: the two-level plan (l2_result -> subgoal g, l1_result -> a*, r_plan), and
callables to repair / re-plan the high level, decide which subgoal to actually execute and tally compute.

Conditions (spec 3.3), all on identical paired tuples:
  flat              : Stage-1 low-level-only (handled by the flat eval, not here)
  hwm               : never repairs; execute g as-is (native HWM)
  hwm_replan        : if r_plan > tau -> full high-level replan (blunt), execute new g; else g
  hwm_replan_every  : full high-level replan every segment regardless of r_plan (naive upper bound)
  repair_trigger    : ours -> if r_plan > tau: repair g; if still > tau: escalate (1x) to full replan
  repair_always     : ablation -> repair every segment regardless of r_plan

Compute is a first-class metric: every predictor forward pass counts. `fwd_passes_low/high` are
accumulated; repair adds n_inner (+refresh) high-level forwards, escalation adds a full high-level solve.
Guard: max 1 escalation/segment; if the fresh plan's first g also fails the trigger, execute best-effort.
"""
from __future__ import annotations
from dataclasses import dataclass, field

CONDITIONS = ("flat", "hwm", "hwm_replan", "hwm_replan_every", "repair_trigger", "repair_always")


@dataclass
class SegmentDecision:
    g_exec: object                 # subgoal latent to execute the low level toward
    r_plan_before: float
    r_plan_after: float            # after repair/replan (== before if untouched)
    n_repairs: int = 0
    n_escalations: int = 0
    fwd_passes_low: int = 0
    fwd_passes_high: int = 0


def decide_segment(
    condition: str,
    *,
    g,                             # l2_result.pred_obs[1]  (current subgoal)
    r_plan: float,                 # scalar for this env
    tau: float,
    repair_fn,                     # (g) -> (g_repaired, r_plan_after, fwd_high_added)   [None for non-repair]
    replan_high_fn,                # () -> (g_new, r_plan_new, fwd_high_added)            [None for hwm]
    low_plan_cost: int,            # fwd passes the low-level plan already spent this segment
) -> SegmentDecision:
    d = SegmentDecision(g_exec=g, r_plan_before=r_plan, r_plan_after=r_plan,
                        fwd_passes_low=low_plan_cost)

    if condition in ("flat", "hwm"):
        return d                                                   # execute g as-is

    if condition == "hwm_replan_every":
        g2, r2, fh = replan_high_fn()
        d.g_exec, d.r_plan_after, d.n_escalations, d.fwd_passes_high = g2, r2, 1, fh
        return d

    if condition == "hwm_replan":
        if r_plan > tau:
            g2, r2, fh = replan_high_fn()
            d.g_exec, d.r_plan_after, d.n_escalations, d.fwd_passes_high = g2, r2, 1, fh
        return d

    if condition == "repair_always" or (condition == "repair_trigger" and r_plan > tau):
        g2, r2, fh = repair_fn(g)
        d.g_exec, d.r_plan_after, d.n_repairs, d.fwd_passes_high = g2, r2, 1, fh
        # escalation (repair_trigger only): repaired subgoal still unreachable -> one full replan
        if condition == "repair_trigger" and r2 > tau:
            g3, r3, fh2 = replan_high_fn()
            d.fwd_passes_high += fh2
            if r3 <= tau or True:                                 # execute best-effort even if still >tau
                d.g_exec, d.r_plan_after, d.n_escalations = g3, r3, 1
        return d

    return d                                                       # repair_trigger with r_plan<=tau: as-is


@dataclass
class ComputeCounter:
    """Global per-episode compute tally (fwd passes at both levels, incl. repair/escalation)."""
    fwd_passes_low: int = 0
    fwd_passes_high: int = 0
    n_replans: int = 0
    n_repairs: int = 0
    n_escalations: int = 0

    def add(self, d: SegmentDecision):
        self.fwd_passes_low += d.fwd_passes_low
        self.fwd_passes_high += d.fwd_passes_high
        self.n_repairs += d.n_repairs
        self.n_escalations += d.n_escalations
