#!/usr/bin/env bash
# M2 — residual calibration. Launch ONLY after M1 (HWM reproduction) is confirmed and the GPU is free.
# Runs an INSTRUMENTED hwm eval (ECHO_SEGMENT_CSV enables logging; identical trajectories otherwise),
# dumps per-segment (r_plan,r_exec,succeeded,L) to a CSV, then runs the calibration + sanity checks.
source /workspace/HWM_PLDM/activate_env.sh >/dev/null 2>&1
cd /workspace/HWM_PLDM
mkdir -p runs
export ECHO_METHOD=hwm
export ECHO_SEGMENT_CSV=/workspace/HWM_PLDM/runs/hwm_segments.csv
export NENV=6

echo "[m2] instrumented hwm run start $(date)"
bash eval.sh hwm > /workspace/eval_hwm_m2.log 2>&1
echo "[m2] hwm done; segments -> $ECHO_SEGMENT_CSV ($(wc -l < runs/hwm_segments.csv 2>/dev/null) rows)"

echo "[m2] calibration (moderate L = medium = 2) ..."
python -m repair.calibrate --csv runs/hwm_segments.csv --moderate_L 2 --out repair/CALIBRATION.md 2>&1 | tail -30
echo "[m2] DONE $(date). See repair/CALIBRATION.md + figures/calib_*.png"
