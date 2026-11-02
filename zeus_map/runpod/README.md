# Zeus-P1 on RunPod — Stage 1 baseline reproduction

Goal: run HWM_PLDM's flat + hierarchical eval on a Linux+CUDA box (this repo is CUDA + mujoco-py/d4rl,
which don't work on the arm64 Mac). Everything here is designed to keep **GPU-hours cheap** — the eval is
inference on frozen conv models, not training.

## Pick the pod (cheap)
- **GPU:** any 24 GB card is overkill-but-cheap. Community-cloud **RTX 3090 (~$0.22/hr)** or **RTX 4090
  (~$0.34/hr)** is ideal. Do NOT rent an A100 — we're doing MPPI inference, not training.
- **Template:** RunPod **"PyTorch 2.x, CUDA 12.x, Ubuntu 22.04"** official template.
- **Volume:** attach a **~50 GB persistent volume** mounted at `/workspace`. This is the key cost-saver —
  the conda env + mujoco + datasets survive pod stop/start, so you install mujoco-py ONCE. Without it you
  re-install every session (mujoco-py build is the slow part).
- Est. total for Stage 1: setup ~1–2 GPU-hr (one-time) + eval ~1–2 GPU-hr ⇒ **~$1–3**.

## Steps (run on the pod, in order)
```bash
cd /workspace
git clone https://github.com/kevinghst/HWM_PLDM.git
cd HWM_PLDM
# copy the three scripts from zeus-p1/runpod/ into this repo (scp, or paste), then:
bash setup.sh          # system libs + mujoco210 + conda(py3.9) + pip install -r requirements + pip install -e .
source activate_env.sh # exports MUJOCO/LD_LIBRARY_PATH and activates the conda env (source it every new shell)
bash download.sh       # pretrained ckpts + dataset from HuggingFace
bash eval.sh flat      # smoke: flat baseline on 'easy'
bash eval.sh hwm       # smoke: hierarchical on 'easy'
```

## What "done" looks like (M1)
- Both eval commands run to completion and print per-level success rates.
- `hwm` > `flat`, flat collapses at hard, hier degrades later (the paper's qualitative curve).
- Numbers within ~5–10 pts of the paper (RECON §6 — pull the targets from arXiv 2604.03208 Fig 1b).

## Known gotchas (fix on the box, they're cheap path/env errors not GPU-time)
- **mujoco-py first-import compiles** (cython). Slow ONCE; cached on the persistent volume after.
- **Rendering backend:** we set `MUJOCO_GL=egl`. If EGL fails headless, try `MUJOCO_GL=osmesa`
  (and `apt install libosmesa6-dev`). `activate_env.sh` has both, commented.
- **Config abs paths:** the yaml has `/scratch/wz1232/...` hardcoded (root_path, images_path). `eval.sh`
  overrides `root_path` and dataset paths via `--values`. If eval complains about a missing path, add the
  matching `--values key=/workspace/HWM_PLDM/...` override and re-run (path errors surface instantly).
- **Probe / starts_targets set:** eval reads `maze2d_large_diverse_probe/starts_targets_{5_8,9_12,13_16}.pt`.
  If the 25maps dataset snapshot doesn't include the probe set, check HF `kevinghst/*` for a probe dataset
  repo and add it to `download.sh`. (First eval run will tell us exactly which files it wants.)
- **wandb:** we set `WANDB_MODE=offline` so no login is needed.

## After M1
Report the numbers back (they go in RECON §6 as the reproduction check), then we start Phase 2 (residual
instrumentation) as a minimal patch at the seam in `two_lvl_planner.py` (RECON §4). Our code stays in
`zeus-p1/repair/`; upstream edits get logged in `zeus-p1/PATCHES.md`.
