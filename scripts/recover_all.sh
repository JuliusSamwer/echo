#!/bin/bash
# Recover the full rung 1->3 pipeline from scratch and persist EVERY result inside the repo.
# Models -> models/  ;  per-stage logs -> models/results/  ;  master doc -> RESULTS.md
# Sequential on purpose (all stages default to the MPS device; no contention).
ROOT=/Users/juliius/Nobel30/Zeus
MD=$ROOT/models/learned_models_v2
RES=$ROOT/models/results
cd "$ROOT"
mkdir -p "$MD" "$RES"

run () {  # run <name> <cmd...> : tee to log, never abort the whole recovery on one failure
  local name="$1"; shift
  echo "===== [$name] START $(date '+%H:%M:%S') ====="
  ( set +e; "$@" ) > "$RES/$name.log" 2>&1
  local code=$?
  echo "===== [$name] DONE exit=$code $(date '+%H:%M:%S') ====="
  echo "$name $code" >> "$RES/_exitcodes.txt"
}

: > "$RES/_exitcodes.txt"

# 0. Substrate (rung 2+3 shared) — train only if fewer than 10 models present.
n=$(ls "$MD"/maze_*.pt 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -lt 10 ]; then
  run substrate env ZEUS_DEVICE=mps python -u scripts/train_models.py --mazes 10 --model-dir "$MD"
else
  echo "[substrate] $n models already present — skipping train"
fi

# 1. Rung 1 — failure curve (trains its own cached model) -> figures/rung1_failure_curve.png
run rung1_failure_curve env ZEUS_DEVICE=mps python -u scripts/rung1_failure_curve.py

# 2. Rung 2 — learned flat-vs-hier rigorous eval -> figures/rung2_learned_rigorous.png
run rung2_pipeline bash scripts/run_learned_v2_pipeline.sh

# 3. Rung 3 — the de-risking ladder (each prints its own gate)
for step in rung3_step0 rung3_step1 rung3_step2 rung3_step3 rung3_step5 rung3_step6 rung3_stepA; do
  run "$step" env ZEUS_DEVICE=mps python -u "scripts/$step.py"
done

# ---- assemble the master results doc ----
python - "$RES" > "$ROOT/RESULTS.md" <<'PY'
import sys, pathlib
res = pathlib.Path(sys.argv[1])
order = ["substrate","rung1_failure_curve","rung2_pipeline",
         "rung3_step0","rung3_step1","rung3_step2","rung3_step3",
         "rung3_step5","rung3_step6","rung3_stepA"]
codes = {}
ec = res/"_exitcodes.txt"
if ec.exists():
    for ln in ec.read_text().split("\n"):
        if ln.strip(): k,v = ln.rsplit(" ",1); codes[k]=v
print("# Zeus — recovered results (rung 1-3)\n")
print("Regenerated from scratch after the scratchpad-loss incident. "
      "Models live in `models/` (persistent, inside the repo).\n")
print("## Stage status\n")
print("| stage | exit | log |")
print("|---|---|---|")
for s in order:
    log = res/f"{s}.log"
    status = "n/a" if s not in codes else ("ok" if codes[s]=="0" else f"FAIL({codes[s]})")
    print(f"| {s} | {status} | `models/results/{s}.log` |")
print()
for s in order:
    log = res/f"{s}.log"
    if not log.exists():
        continue
    print(f"\n## {s}\n")
    print("```")
    print(log.read_text().rstrip())
    print("```")
PY
echo "RECOVER_ALL_DONE — see $ROOT/RESULTS.md"
