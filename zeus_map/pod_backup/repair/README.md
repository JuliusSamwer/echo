# Zeus-P1 `repair/` — Stage 2 (residual instrumentation, calibration, repair)

Reachability-residual → subgoal repair on the frozen HWM_PLDM checkpoints. Milestone order is STRICT:
M1 (reproduction, done) → **M2 (calibration)** → M3 (repair smoke + sweep) → M4 (full matrix + figures).

## The idea in one line
HWM's low-level planner computes a reachability estimate for every subgoal and throws it away. We
recover it (`r_plan`), and (Stage 2) use it to locally repair invalid/too-hard subgoals by latent-space
gradient descent — frozen models. Hypothesis: shifts the success-vs-horizon curve right at matched compute.

## Files
- `residuals.py`  — r_plan / r_exec (planner-cost metric), `SegmentLogger`, passive `record_segment` hook.
- `calibrate.py`  — Phase 2 sanity checks (Spearman, succ-vs-fail separation), τ, `CALIBRATION.md`.
- `repair_op.py`  — Phase 3.1 repair operator E(g)=α·d(ẑ_T,g)²+β·d(F_high(g),z_goal)²; frozen-param guards.
- `control_loop.py` — Phase 3.2 per-segment decide (trigger→repair→escalate) + compute counters; the 6 conditions.
- `stats.py`      — Wilson CIs, paired bootstrap, McNemar.
- `plots.py`      — the 6 figures (headline, success-vs-compute, calib, repair before/after, rates, rollout).
- `PATCH_INSTRUMENT.md` — the single upstream hook in `pldm/planning/mpc.py` (kept in `PATCHES.md`).
- `configs/`      — one YAML per condition; `base.yaml` holds τ + repair hyperparams.
- `tests/`        — frozen-params, grad-reaches-only-g, metric==RunningCost, instrument no-op. (all green)

## M2 — calibration (do this first, on the pod)
1. Install the instrumentation hook (`PATCH_INSTRUMENT.md`) — log it in `PATCHES.md`.
2. Re-run the `hwm` condition with a logger installed → `runs/hwm_segments.csv` (spec CSV schema).
   Assert bit-exact trajectories vs uninstrumented under fixed seed (test c).
3. `python -m repair.calibrate --csv runs/hwm_segments.csv --moderate_L <L> --out repair/CALIBRATION.md`
   - Sanity 1: Spearman(r_plan,r_exec) > ~0.3, else RED (r_plan not a proxy — stop).
   - Sanity 2: r_plan larger on failed episodes (AUC>0.6), else switch `residual.key: r_plan_norm`.
   - τ = p90 of r_plan over successful segments at moderate L → into `configs/base.yaml`.
M2 done when: logging has zero effect on numbers, calib figures exist, τ chosen + justified in CALIBRATION.md.

## M3/M4 — repair
- Conditions: `hwm`, `hwm_replan`, `hwm_replan_every`, `repair_trigger` (ours), `repair_always`.
  Spine = `repair_trigger` vs `hwm_replan(_every)` at matched/lower compute.
- Smoke every new condition on 5 mazes first. Sweep (3.4) on a 20-maze subset at 2 failure-L values:
  η∈{3e-3,1e-2,3e-2}, n_inner∈{5,10,25}, β/α∈{0.3,1,3}, τ∈{p75,p90,p97}, refresh on/off. Pick best on
  subset, run the full matrix ONCE. No test-set tuning.
- Accept/kill (M3–M4): GREEN if `repair_trigger` beats `hwm` with non-overlapping CIs on ≥2 L at ≤ compute
  of `hwm_replan_every`. YELLOW if `hwm_replan` matches at similar compute (story = cheap repair vs expensive
  replan; check low-compute regime). RED if repair moves nothing — debug order in the spec; do NOT paper over.

## Compute is first-class
Every predictor forward counts (both levels, every MPPI/CEM sample, every repair inner step). Reported per
(method, L). All success-vs-compute comparisons are compute-matched.
