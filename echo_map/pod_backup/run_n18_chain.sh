#!/usr/bin/env bash
set -x
cd /workspace/HWM_PLDM
source /workspace/HWM_PLDM/activate_env.sh
echo "[chain] START $(date)"
run_cond () {
  local name=$1
  rm -f "$ECHO_CAA_CSV" "${ECHO_REPAIR_CSV:-/tmp/none_$name}"
  NENV=6 bash eval.sh hwm > /workspace/n18_${name}.log 2>&1
  cp "$(find checkpoint -iname "planning_l2_mpc_report*hard*" | head -1)" /workspace/n18_${name}_report.pt 2>/dev/null
  echo "[chain] ${name} DONE $(date)"
}
unset ECHO_REPAIR_CONDITION ECHO_REPAIR_CSV ECHO_TAU
export ECHO_CAA_CSV=/workspace/HWM_PLDM/runs/n18_hwm_caa.csv
run_cond hwm
export ECHO_REPAIR_CONDITION=repair_always ECHO_ALPHA=1 ECHO_BETA=1 ECHO_NINNER=10 ECHO_ETA=1e-2
export ECHO_REPAIR_CSV=/workspace/HWM_PLDM/runs/n18_always_events.csv ECHO_CAA_CSV=/workspace/HWM_PLDM/runs/n18_always_caa.csv
run_cond always
export ECHO_REPAIR_CONDITION=repair_trigger ECHO_TAU=0.06
export ECHO_REPAIR_CSV=/workspace/HWM_PLDM/runs/n18_trig_events.csv ECHO_CAA_CSV=/workspace/HWM_PLDM/runs/n18_trig_caa.csv
run_cond trig
echo "CHAIN COMPLETE $(date)"
