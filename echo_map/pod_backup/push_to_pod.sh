#!/usr/bin/env bash
# Push the two-pressure build to a freshly-resumed pod.  Usage: bash push_to_pod.sh <HOST> <PORT>
# Only touches the files this build changed/added — leaves everything else on /workspace intact.
set -e
HOST="$1"; PORT="$2"
[ -z "$PORT" ] && { echo "usage: bash push_to_pod.sh <HOST> <PORT>"; exit 1; }
KEY=~/.ssh/runpod_ed25519
R="root@$HOST"
SCP="scp -i $KEY -P $PORT -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
D=/workspace/HWM_PLDM

$SCP repair/instrument.py repair/seam.py "$R:$D/repair/"
$SCP progress_sep.py twop_analyze.py setcfg.py \
     run_diag_progress.sh run_2p_repairs.sh "$R:$D/"
echo "pushed. verifying import on pod..."
ssh -i $KEY -p $PORT -o ConnectTimeout=20 "$R" \
  "cd $D && source activate_env.sh >/dev/null 2>&1 && python -c 'import repair.instrument, repair.seam; print(\"repair imports OK\")'"
