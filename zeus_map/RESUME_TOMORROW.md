# Resume tomorrow — Zeus-P1

## State (2026-07-09 EOD)
- **M1 flat: DONE + backed up** (`zeus-p1/results/`, pod `/workspace/flat_backup/`):
  easy **0.975**, medium **0.700** (40-env) / 0.583 (6-env), hard **0.500**. Matches paper (100/63/44).
- **M1 HWM: NOT captured yet.** The L2 eval works and ran to L2-planning cleanly; the only blocker is the
  harness writing L2 report files only at the very end + the `levels`/`n_envs` `--values` overrides being
  silently ignored (OmegaConf nested-key issue). So it grinds L1+L2×medium+hard instead of just L2-medium.
- **Stage-2 scaffold: built + unit-tested** (`zeus-p1/repair/`), instrumentation hook applied to
  `pldm/planning/mpc.py` (no-op unless `ZEUS_SEGMENT_CSV` set). Instrument bug fixed (`model.level1.backbone`).
- Everything on pod `/workspace` (patched repo, rendered `images.npy`, conda env `pldm`, checkpoints).

## FIRST thing tomorrow: get the HWM L2 medium number (clean, ~30 min)
Root cause to fix first: the `--values eval_cfg.d4rl_planning.levels/medium.n_envs` overrides DON'T apply.
FIX by editing the yaml directly (already done for n_envs=6). Also set it to L2-medium-ONLY so it doesn't
run hard/L1:
```
# on pod, edit pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml:
#   d4rl_planning.levels: "medium"      (drop hard)
#   eval_l1: false  (already)   n_envs: 6 (already)
ssh ... ; cd /workspace/HWM_PLDM
NENV=6 setsid bash run_eval.sh hwm > /workspace/hwm.log 2>&1 < /dev/null &
# watch for planning_l2_mpc_report_d4rl_medium ; ~18min prober + ~12min planning
```
Expect HWM medium ≫ flat (paper 0.95 vs 0.70). If yes -> M1 confirmed.

## Then: Stage 2 (M2 calibration) — one command
`bash run_m2.sh`  (instrumented L2-medium -> runs/hwm_segments.csv -> repair.calibrate -> CALIBRATION.md)
Verify segments log clean (instrument's first real run). Then M3/M4 per `repair/README.md`.

## Reusable gotchas (so we don't re-hit them)
- Network volume (`/workspace`) forbids chown -> `tar --no-same-owner`.
- conda ToS auto-accept; `setuptools<81` (wandb/pkg_resources); DataLoader `num_workers=0` monkeypatch in
  train.py (mmap shared-mem crash); images MUST stay `mmap_mode="r"` (full RAM load OOMs the 125GB container).
- `--values` nested per-level overrides (levels/n_envs) DON'T apply -> edit the yaml directly.
- SSH launches: use `setsid ... < /dev/null &` in a SEPARATE ssh call; never `pkill -f run_X` in the same
  command that launches run_X (it matches the ssh string and self-kills).
- pgrep procs: use `pgrep -fc train.py` (not `-c`, which matches the process name "python").
