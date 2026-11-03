# PATCHES — upstream HWM_PLDM files we touch (keep MINIMAL, no refactor)

Ground rule: all ECHO-P1 code lives in `echo-p1/repair/`. Any edit to an upstream file in
`HWM_PLDM/` is logged here with file:line + one-line reason. Stage 1 aims for ZERO upstream edits
(config overrides only). Device/CUDA portability edits, if we end up needing them off-GPU, go here too.

| # | file | lines | reason | stage |
|---|------|-------|--------|-------|
| — | (none yet) | | | |

## Ready to apply (Stage 2 — scaffold built in `repair/`, NOT yet applied upstream)
- `pldm/planning/mpc.py::MPCEvaluator._perform_mpc` — the ONE hook: passive `repair.residuals.record_segment`
  call at each replan (`i % replan_every == 0`) after the level1/level2 split. Exact diff + attrs in
  `repair/PATCH_INSTRUMENT.md`. Zero effect unless a logger is installed; reads detached tensors only.
  Seam values: subgoal `g = planning_result.level2.pred_obs[1]`, low-level predicted terminal =
  `planning_result.level1.pred_obs[-1]`, metric = planner-cost mean-sq (`repair/residuals.latent_distance`).
- Phase-3 repair reuses the SAME site, guarded by condition (`repair/control_loop.decide_segment`).
- `repair/` unit tests green locally (frozen params, grad-reaches-only-g, metric==RunningCost, no-op hook).

## Applied (Stage 1 eval, infra-only — NO effect on paper results)
All to make the authors' checkpoint-eval run on our box with our rendered images. None change the
model, planner, hyperparameters, num_samples, horizon, or algorithm.
1. `pldm/train.py` (top): monkeypatch `torch DataLoader.__init__` -> force `num_workers=0` (drop
   prefetch/persistent/worker_init). The images.npy is loaded read-only (mmap/from_numpy); DataLoader
   worker processes crash moving it to shared memory ("Trying to resize storage that is not resizable").
   Single-process loading is byte-identical data. THIS is what unblocked both flat and HWM.
2. `pldm_envs/diverse_maze/d4rl.py`: `np.load(..., mmap_mode="r")` -> `np.ascontiguousarray(np.load(...))`
   and `.permute(0,3,1,2)` -> `.permute(0,3,1,2).contiguous()`. Contiguous/in-RAM images. Data-identical.
3. `pldm/data/utils.py::make_dataloader`: `num_workers = 0` (belt-and-suspenders w/ #1).
Config-only remaps (via --values, not code): root_path, images_path -> our rendered probe images.npy,
main-ds path -> probe split (only used for obs-shape + latent-bound stats), probing image paths,
probing.max_samples=null (use paper's 5 epochs), n_envs=12 (fewer eval episodes), levels=medium,hard.

## APPLIED — Stage-2 instrumentation hook (2026-07-09, pod: /workspace/HWM_PLDM)
`pldm/planning/mpc.py`: 4 insertions (import + `maybe_start` + `on_replan` + `on_episode_end`), all calls
into `repair/instrument.py`. NO-OP unless env `ECHO_SEGMENT_CSV` is set (baseline bit-exact). Verified:
`import pldm.planning.mpc` and `from repair import ...` both succeed on the pod. `repair/` copied to
/workspace/HWM_PLDM/repair (on the pldm pythonpath). M2 launch = `bash run_m2.sh` (after M1 confirmed + GPU free).
