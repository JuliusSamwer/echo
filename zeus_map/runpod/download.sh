#!/usr/bin/env bash
# Download pretrained HWM/PLDM checkpoints + the diverse-maze dataset from HuggingFace.
# Run from the HWM_PLDM repo root, after `source activate_env.sh`.
set -euo pipefail

echo "[download] pretrained checkpoints -> pldm/pretrained/ ..."
python pldm/download_ckpt_from_hf.py --out-dir "$(pwd)/pldm/pretrained"
# -> 3-9-1-seed248_epoch=3_sample_step=15465472.ckpt         (flat / PLDM, 1 level)
# -> load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt (HWM, 2 levels)

echo "[download] diverse-maze dataset -> pldm_envs/diverse_maze/datasets/ ..."
python pldm_envs/diverse_maze/data_generation/download_ds_from_hf.py \
  --repo-id kevinghst/maze2d-large-diverse-25maps

echo "[download] contents:"
ls -la pldm/pretrained || true
echo "---"
find pldm_envs/diverse_maze/datasets -maxdepth 2 -type f 2>/dev/null | head -40 || true

cat <<'NOTE'

[download] NOTE: eval also needs the PROBE start/target sets:
  maze2d_large_diverse_probe/starts_targets_{5_8,9_12,13_16}.pt  (easy/medium/hard)
If they are NOT under datasets/ after the snapshot above, look for a probe dataset on HF
(e.g. `huggingface-cli repo ls kevinghst` or the project page) and download it into
pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe/ . The first eval run will name the
exact file it wants if one is missing.
NOTE
