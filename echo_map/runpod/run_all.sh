#!/usr/bin/env bash
# Unattended M1 pipeline: flat (medium,hard) then HWM (medium,hard) at reduced envs, collect results.
# No set -e: if one eval hiccups, still run the other + collect whatever exists.
source /workspace/HWM_PLDM/activate_env.sh >/dev/null 2>&1
cd /workspace/HWM_PLDM
export NENV=12

echo "[run_all] FLAT (medium,hard) start $(date)"
bash eval.sh flat > /workspace/eval_flat_run.log 2>&1
echo "[run_all] FLAT done $(date)"

echo "[run_all] HWM (medium,hard) start $(date)"
bash eval.sh hwm > /workspace/eval_hwm_run.log 2>&1
echo "[run_all] HWM done $(date)"

echo "[run_all] collecting results $(date)"
python - > /workspace/RESULTS.txt 2>/dev/null <<'PY'
import torch, os
base="checkpoint/maze2d_large_diverse/maze2d_large_diverse"
def sr(f):
    try:
        r=torch.load(f, weights_only=False); return f"{r.success_rate:.3f} (n={getattr(r,'num_one_turns',0)+getattr(r,'num_two_turns',0)+getattr(r,'num_three_turns',0)})"
    except Exception as e: return f"ERR {e}"
print("=== ECHO-P1 M1 REPRODUCTION (reduced n_envs=12; easy-flat from earlier 40-env run) ===")
print("Paper targets:  FLAT easy 100 / med ~63 / hard 44   |   HWM med 95 / hard 83")
print("-"*64)
for lvl in ["easy","medium","hard"]:
    f=f"{base}/planning_l1_mpc_report_d4rl_{lvl}"
    if os.path.exists(f): print(f"FLAT  {lvl:6s}: {sr(f)}")
for lvl in ["medium","hard"]:
    f=f"{base}/planning_l2_mpc_report_d4rl_{lvl}"
    if os.path.exists(f): print(f"HWM   {lvl:6s}: {sr(f)}")
PY
cat /workspace/RESULTS.txt
echo "[run_all] ALL DONE $(date)"
