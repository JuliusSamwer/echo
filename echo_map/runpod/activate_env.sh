#!/usr/bin/env bash
# Source this in EVERY new shell:  source activate_env.sh
ZP1="${ZP1_ROOT:-/workspace}"
source "$ZP1/miniconda3/etc/profile.d/conda.sh"
conda activate pldm
export MUJOCO_PY_MUJOCO_PATH="$ZP1/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ZP1/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_GL=egl          # headless GL for rendering the 98x98 obs; if EGL fails try: export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline     # no wandb login needed for eval
echo "[env] pldm active | MUJOCO_GL=$MUJOCO_GL | cuda=$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)"
