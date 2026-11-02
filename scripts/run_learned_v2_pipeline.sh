#!/bin/bash
# Autonomous: wait for 10 properly-trained models -> eval (flat vs hier, paired) -> aggregate panel.
set -e
ROOT=/Users/juliius/Nobel30/Zeus
SCR=/Users/juliius/Nobel30/Zeus/models
MD=$SCR/learned_models_v2
ED=$SCR/learned_eval_v2
cd "$ROOT"

echo "[pipe] waiting for 10 models..."
while [ "$(ls $MD/maze_*.pt 2>/dev/null | wc -l | tr -d ' ')" -lt 10 ]; do sleep 20; done
echo "[pipe] 10 models ready. launching eval shards."

rm -rf "$ED"; mkdir -p "$ED"
for i in 0 1 2 3 4; do
  ZEUS_DEVICE=cpu ZEUS_THREADS=3 python -u scripts/rung2_learned_rigorous.py \
    --shard $i --n-shards 5 --mazes 10 --pairs 10 --out-dir "$ED" --model-dir "$MD" \
    > "$ED/log_$i.txt" 2>&1 &
done
wait
echo "[pipe] eval done. aggregating."
python scripts/aggregate_rigorous.py "$ED" "LEARNED model (rmse 0.08)" "rung2_learned_rigorous.png"
echo "[pipe] PIPELINE_DONE"
