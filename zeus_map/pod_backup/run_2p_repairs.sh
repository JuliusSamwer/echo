#!/usr/bin/env bash
# STEP 3: two-pressure repair conditions (hard, n=18, n_steps=150). Baseline = the diag hwm run.
# Env knobs (set from step-2 analysis before launching):
#   TAU_PROG   goal-progress trigger threshold (fire if goal_prog < TAU_PROG). default 0.0
#   BETA_HI    strong goal-direction weight for the ceiling always-repair. default 3
#   TAU_REACH  reachability trigger threshold (for the 'both' condition). default 0.06
set -x
cd /workspace/HWM_PLDM
source /workspace/HWM_PLDM/activate_env.sh
python setcfg.py 18 18 150

TAU_PROG="${TAU_PROG:-0.0}"
BETA_HI="${BETA_HI:-3}"
TAU_REACH="${TAU_REACH:-0.06}"

run_cond () {
  local name=$1
  rm -f "$ZEUS_CAA_CSV" "$ZEUS_REPAIR_CSV" "$ZEUS_PROGRESS_CSV"
  NENV=6 bash eval.sh hwm > /workspace/2p_${name}.log 2>&1
  echo "[2p] ${name} DONE $(date)"
}

echo "[2p] START $(date) TAU_PROG=$TAU_PROG BETA_HI=$BETA_HI TAU_REACH=$TAU_REACH"
export ZEUS_NINNER=10 ZEUS_ETA=1e-2 ZEUS_ALPHA=1

# --- Ceiling: always-repair with STRONG goal-direction (perfect detection, aggressive top-down) ---
export ZEUS_REPAIR_CONDITION=repair_always ZEUS_BETA=$BETA_HI
export ZEUS_REPAIR_CSV=/workspace/HWM_PLDM/runs/2p_alwaysB_events.csv
export ZEUS_CAA_CSV=/workspace/HWM_PLDM/runs/2p_alwaysB_caa.csv
export ZEUS_PROGRESS_CSV=/workspace/HWM_PLDM/runs/2p_alwaysB_progress.csv
run_cond alwaysB

# --- Recoverability: trigger on GOAL-PROGRESS stall (model-derived top-down signal) ---
export ZEUS_REPAIR_CONDITION=repair_trigger ZEUS_BETA=$BETA_HI ZEUS_TRIGGER=progress ZEUS_TAU_PROG=$TAU_PROG
export ZEUS_REPAIR_CSV=/workspace/HWM_PLDM/runs/2p_trigprog_events.csv
export ZEUS_CAA_CSV=/workspace/HWM_PLDM/runs/2p_trigprog_caa.csv
export ZEUS_PROGRESS_CSV=/workspace/HWM_PLDM/runs/2p_trigprog_progress.csv
run_cond trigprog

# --- Two-pressure: trigger if EITHER unreachable OR not goal-directed ---
export ZEUS_REPAIR_CONDITION=repair_trigger ZEUS_BETA=$BETA_HI ZEUS_TRIGGER=both ZEUS_TAU=$TAU_REACH ZEUS_TAU_PROG=$TAU_PROG
export ZEUS_REPAIR_CSV=/workspace/HWM_PLDM/runs/2p_trigboth_events.csv
export ZEUS_CAA_CSV=/workspace/HWM_PLDM/runs/2p_trigboth_caa.csv
export ZEUS_PROGRESS_CSV=/workspace/HWM_PLDM/runs/2p_trigboth_progress.csv
run_cond trigboth

echo "2P CHAIN COMPLETE $(date)"
