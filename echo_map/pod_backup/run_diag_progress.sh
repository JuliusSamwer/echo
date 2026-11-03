#!/usr/bin/env bash
# STEP 2: one instrumented HWM run (hard, n=18, n_steps=150) logging reachability (segments),
# per-episode closest-approach (caa -> fail labels), AND per-step dist-to-goal (progress).
# Doubles as the hwm baseline for the step-3 two-pressure comparison (same episodes).
set -x
cd /workspace/HWM_PLDM
source /workspace/HWM_PLDM/activate_env.sh
python setcfg.py 18 18 150          # top n_envs=18, hard.n_envs=18, n_steps=150 (multi-chunk fix in place)

rm -f runs/diag_segments.csv runs/diag_caa.csv runs/diag_progress.csv
unset ECHO_REPAIR_CONDITION ECHO_REPAIR_CSV ECHO_TAU ECHO_TRIGGER
export ECHO_SEGMENT_CSV=/workspace/HWM_PLDM/runs/diag_segments.csv
export ECHO_CAA_CSV=/workspace/HWM_PLDM/runs/diag_caa.csv
export ECHO_PROGRESS_CSV=/workspace/HWM_PLDM/runs/diag_progress.csv
export NENV=6

echo "[diag] START $(date)"
bash eval.sh hwm > /workspace/diag_progress.log 2>&1
cp "$(find checkpoint -iname "planning_l2_mpc_report*hard*" | head -1)" /workspace/diag_hwm_report.pt 2>/dev/null
echo "[diag] DONE $(date)  seg=$(wc -l < runs/diag_segments.csv) caa=$(wc -l < runs/diag_caa.csv) prog=$(wc -l < runs/diag_progress.csv)"
