#!/usr/bin/env bash
# Run the HWM_PLDM baseline eval (Stage 1 / M1). Usage:  bash eval.sh {flat|hwm}
# Run from repo root after `source activate_env.sh` and `bash download.sh`.
set -euo pipefail
MODE="${1:-hwm}"
REPO="$(pwd)"
PRE="$REPO/pldm/pretrained"
FLAT_CKPT="$PRE/3-9-1-seed248_epoch=3_sample_step=15465472.ckpt"
HWM_CKPT="$PRE/load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt"

cd "$REPO/pldm"
# Override the hardcoded /scratch image paths from the yaml to the images WE RENDERED (render.sh +
# postprocess -> images.zarr), exactly like their generate_all_datasets_og.sh produces. root_path
# resolves the ${root_path} data/probe/starts_targets paths. The main (25maps) training images are
# only iterated for L2 latent bounds (HWM eval); flat (L1) doesn't touch them, so leave it null here.
PROBE_IMG="$REPO/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe/images.npy"
MAPS_IMG="$REPO/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/images.npy"
# Main (25maps) ds is only used for next(iter(ds)) shape-inference at Trainer construction (model is
# loaded, not trained; planning uses the live env) -> point it at the already-rendered probe images so
# the sample is shape-correct, avoiding a 130GB render of the full training set that eval never reads.
# crop_length=85000: keeps the main ds under the 101k probe images we point it at (no out-of-range
# empty samples), while leaving the probe set ~full (86k) so the location prober trains properly.
# probing.max_samples=null -> use the config's epochs (5) for the prober, not the 400 that max_samples
# would force on the cropped set (90min -> ~6min). Prober is diagnostic; planning success is env-defined.
# NENV: reduced eval envs for a lighter-but-trustworthy run (paper uses 40). levels=medium,hard is the
# comparison where hierarchy matters (easy already reproduced at 97.5%). batch_size=NENV -> 1 batch/level.
NENV="${NENV:-12}"
PROBE_DATA="$REPO/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe/data.p"
# NO method/hyperparameter changes vs the paper. Only: (a) remap the authors' hardcoded /scratch paths to
# our rendered data, (b) reduce n_envs (fewer eval episodes -> wider CIs, same expected success), (c) the
# main ds is repointed to the probe split (its ONLY eval uses are obs-shape inference + L2 latent-bound
# stats, both distribution-level; the 25maps train images aren't shipped and rendering all 4.5M is the only
# other option). The read-only-mmap crash is fixed in-code by loading images to RAM (see PATCHES.md) - same
# data, not a paper change. num_workers, num_samples, horizon, replan, model: all paper defaults.
COMMON="eval_only=true root_path=$REPO data.d4rl_config.path=$PROBE_DATA data.d4rl_config.images_path=$PROBE_IMG eval_cfg.probing.train_images_path=$PROBE_IMG eval_cfg.probing.val_images_path=$PROBE_IMG eval_cfg.probing.max_samples=null eval_cfg.probing.max_samples_enc=null eval_cfg.d4rl_planning.levels=hard eval_cfg.d4rl_planning.n_envs_batch_size=$NENV"

if [ "$MODE" = "flat" ]; then
  echo "[eval] FLAT (1-level PLDM) ..."
  python train.py --config configs/diverse_maze/icml/large_diverse_25maps.yaml \
    --values $COMMON load_checkpoint_path="$FLAT_CKPT" 2>&1 | tee "$REPO/eval_flat.log"
elif [ "$MODE" = "hwm" ]; then
  echo "[eval] HWM (2-level hierarchical) ..."
  python train.py --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
    --values $COMMON load_l1_only=false load_checkpoint_path="$HWM_CKPT" 2>&1 | tee "$REPO/eval_hwm.log"
else
  echo "usage: bash eval.sh {flat|hwm}"; exit 1
fi
echo "[eval] done -> $REPO/eval_${MODE}.log  (grep for success rates per level)"
