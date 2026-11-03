#!/usr/bin/env bash
# ECHO-P1 Stage-1 env setup on a RunPod Linux+CUDA box. Idempotent; fails loud.
# Run from the HWM_PLDM repo root. One-time; cached on the persistent /workspace volume.
set -euo pipefail
REPO="$(pwd)"
echo "[setup] repo=$REPO"

# ---------------------------------------------------------------- 1. system libs for mujoco-py 2.1
echo "[setup] apt system deps (mujoco-py / GL / build) ..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  wget git build-essential patchelf \
  libgl1-mesa-dev libgl1-mesa-glx libglew-dev libglfw3 libglfw3-dev \
  libosmesa6-dev xvfb libglib2.0-0 ca-certificates unzip

# Install EVERYTHING on the persistent /workspace volume (root disk is small + ephemeral).
ZP1="${ZP1_ROOT:-/workspace}"

# ---------------------------------------------------------------- 2. MuJoCo 2.1.0 binaries
MJ="$ZP1/.mujoco/mujoco210"
if [ ! -d "$MJ" ]; then
  echo "[setup] downloading mujoco210 ..."
  mkdir -p "$ZP1/.mujoco"
  wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mj210.tar.gz
  # /workspace is a networked (MFS) volume that forbids chown -> tar must not preserve ownership/perms
  tar -xzf /tmp/mj210.tar.gz -C "$ZP1/.mujoco" --no-same-owner --no-same-permissions
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MJ/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="$MJ"

# ---------------------------------------------------------------- 3. conda py3.9 env (repo: python 3.9)
if [ ! -d "$ZP1/miniconda3" ]; then
  echo "[setup] installing miniconda to $ZP1/miniconda3 ..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$ZP1/miniconda3"
fi
source "$ZP1/miniconda3/etc/profile.d/conda.sh"
# recent conda requires explicit ToS acceptance for the default channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
if ! conda env list | grep -q "^pldm "; then
  conda create -n pldm python=3.9 -y
fi
conda activate pldm

# ---------------------------------------------------------------- 4. python deps
echo "[setup] pip install requirements (this compiles mujoco-py on first import) ..."
python -m pip install --upgrade pip wheel setuptools cython==0.29.33
# Cython + mujoco-py must be present before d4rl builds; requirements pins both.
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" pip install -r requirements.txt
pip install -e .
# wandb 0.13.10 imports pkg_resources, which setuptools>=81 removed -> pin below 81
pip install "setuptools<81"
# huggingface_hub for the asset downloads
pip install -U "huggingface_hub>=0.23"

# ---------------------------------------------------------------- 5. verify
echo "[setup] verifying ..."
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import os; os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR","1"); os.environ.setdefault("MUJOCO_GL","egl")
import mujoco_py; print("mujoco_py OK")
import d4rl; print("d4rl OK")
PY
echo "[setup] DONE. Now: source activate_env.sh   (every new shell)"
