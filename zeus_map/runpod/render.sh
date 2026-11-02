#!/usr/bin/env bash
# Render images for a proprio dataset, parallelized + fully detached (survives ssh close).
# Usage:  bash render.sh <data_path> <n_workers>   (n_workers MUST divide the #splits exactly)
source /workspace/HWM_PLDM/activate_env.sh >/dev/null 2>&1
cd /workspace/HWM_PLDM
DP="$1"; NW="${2:-20}"
echo "[render] $DP with $NW workers at $(date +%H:%M:%S)"
for i in $(seq 0 $((NW-1))); do
  setsid python pldm_envs/diverse_maze/data_generation/render_data.py \
    --data_path "$DP" --workers_num "$NW" --worker_id "$i" \
    > "/workspace/rw${i}.log" 2>&1 < /dev/null &
done
echo "[render] launched $NW detached workers"
