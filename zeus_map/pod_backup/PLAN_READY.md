# Two-pressure repair — staged & ready (built 2026-07-28, pod off)

Everything below is BUILT and syntax/logic-checked locally. When the pod is up, it's push → smoke → run.
All changes are ADDITIVE / env-gated — existing pipeline behaves identically when the new flags are unset.

## What changed vs the n=18 pipeline
- `repair/instrument.py`: `on_episode_caa` now ALSO dumps per-step dist-to-goal when `ZEUS_PROGRESS_CSV` set. (existing caa untouched)
- `repair/seam.py`: new goal-progress signal `goal_prog = d(z_cur,z_goal) - d(g0,z_goal)`; new trigger modes via `ZEUS_TRIGGER=reach|progress|both` (default `reach` = old behavior) + `ZEUS_TAU_PROG`. Events CSV now also logs goal_prog/trig_mode.
- New: `progress_sep.py` (step-2 AUC), `run_diag_progress.sh` (step-2 run), `run_2p_repairs.sh` (step-3 runs), `twop_analyze.py` (step-3 stats).

## Resume sequence (when pod is up — new HOST/PORT from RunPod console)
```
# 0. reconnect + re-apt (ephemeral root disk) + push build
#    (re-apt block: build-essential libosmesa6-dev libgl1-mesa-dev libglew-dev libglfw3-dev xvfb ...)
bash push_to_pod.sh <HOST> <PORT>          # syncs changed files, verifies repair imports

# 1. STEP 2 — diagnostic run (~80 min, one hwm run, logs reachability+caa+per-step progress)
ssh ... 'cd /workspace/HWM_PLDM && setsid bash run_diag_progress.sh > /workspace/diag2.log 2>&1 </dev/null &'
#    watch: multi-chunk caa should reach env 17; no "need at least one array" crash.

# 2. STEP 2 analysis (offline, instant)
ssh ... 'cd /workspace/HWM_PLDM && source activate_env.sh && python progress_sep.py'
#    LOOK AT: do trajectory-shape metrics (stall_frac, progress_rate, last_third_closed) beat
#    reachability's AUC~0.64?  >=0.78 => reframe validated. Pick a usable trigger metric + threshold.
#    -> set TAU_PROG from the goal_prog distribution of failing vs succeeding segments (events/analysis).

# 3. STEP 3 — two-pressure repair (~4 hr, 3 conditions; baseline = diag hwm)
ssh ... 'cd /workspace/HWM_PLDM && TAU_PROG=<chosen> BETA_HI=3 setsid bash run_2p_repairs.sh > /workspace/2p.log 2>&1 </dev/null &'

# 4. STEP 3 analysis
ssh ... 'cd /workspace/HWM_PLDM && source activate_env.sh && python twop_analyze.py'
```

## What each result means
- **Ceiling (repair_alwaysB, strong β):** does aggressive goal-directed repair with PERFECT detection
  beat hwm a lot? If yes -> mechanism has headroom (effect was signal-limited, not mechanism-limited).
- **trig_progress:** does firing only on model-derived goal-progress stall RECOVER most of the ceiling
  gain cheaply? This is the recoverability / real-signal claim.
- **trig_both:** two-pressure (reach OR progress) — best of both, the intended contribution.

## ALWAYS after any run: pull results local (crash-proof)
```
# tar runs/*.csv + *_report.pt + logs on pod -> scp to zeus-p1/pod_backup/
```

## Honest notes
- `roll_high` is still identity (top-down term = pull g toward encoded true goal). Fine for now — the
  goal DIRECTION is model-derived through the frozen encoder (allowed). The oracle/model split we test is
  in the TRIGGER (detection), not the pull direction. A true multi-step F_high rollout is a later refinement.
- Step-2 metrics final_dist/min_dist are circular with the fail label — ignore them; judge on the
  trajectory-SHAPE metrics that a live trigger could actually use.
- n=18 still underpowered for significance; goal here is a BIGGER effect size, then n=40 confirms.
