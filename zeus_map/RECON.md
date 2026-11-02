# RECON — HWM_PLDM (Zeus-P1 Phase 0)

Repo: `https://github.com/kevinghst/HWM_PLDM` (cloned at `zeus-p1/HWM_PLDM/`).
Paper: "Hierarchical Planning with Latent World Models" (HWM), arXiv 2604.03208, project page
`kevinghst.github.io/HWM`. This repo is the **minimal PLDM (Diverse Maze) implementation**; the paper's
VJEPA2/DINO-WM results are elsewhere.

## TL;DR
- Models + dataset are **downloadable from HuggingFace** — no training required for Stage 1.
- The **subgoal seam** (all of Stage 2) is one line in `two_lvl_planner.py`. Fully understood (below).
- 🔴 **FEASIBILITY BLOCKER**: repo is **CUDA-hardcoded** (`.cuda()`), and the env needs **d4rl + mujoco-py
  2.1** to render the 98×98 image observations. This machine is **arm64 macOS, MPS-only, no CUDA**.
  Faithful reproduction realistically needs a **Linux + NVIDIA GPU** box. Decision required before Phase 1.

---

## 1. World models (encoder + predictor classes)
- Two-level H-JEPA in `pldm/models/hjepa.py` (`level1` = low, `level2` = high). JEPA in `pldm/models/jepa.py`.
- Encoder/backbone: conv `menet6` (`d4rl_a`), **image input 98×98×3** (`pldm/models/encoders/`). Obs are IMAGES.
- Predictor: conv (`conv2`, `d4rl_b_p`), residual, `action_dim=2`, `z_dim=0` (deterministic, no latent action at L1)
  (`pldm/models/predictors/`). `step_skip=4` (1 high-level macro = 4 primitive steps).
- **Checkpoints SHIPPED on HF** (`pldm/download_ckpt_from_hf.py`, repo `kevinghst/pldm-maze2d-large-diverse`):
  - `3-9-1-seed248_epoch=3_sample_step=15465472.ckpt` (full/L2)
  - `load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt` (L1 low-level)
  → land in `pldm/pretrained/` (dir does not exist yet; created by the script). **No training needed.**

## 2. Dataset
- HF dataset `kevinghst/maze2d-large-diverse-25maps` via `pldm_envs/.../download_ds_from_hf.py`
  → `pldm_envs/diverse_maze/datasets/`. Format: `data.p` (transitions) + `images.npy` (98×98). 25 maps.
- Eval start/target sets: `maze2d_large_diverse_probe/starts_targets_{5_8,9_12,13_16}.pt` (the difficulty knob).

## 3. Planners (per level) + loop
- Both levels use **MPPI** (`pldm/planning/planners/mppi_planner.py` + `mppi_torch.py`). SGD planner also present.
- Orchestration: `pldm/planning/planners/two_lvl_planner.py::TwoLvlPlanner.plan`.
- Episode/control loop: `pldm/planning/mpc.py::MPCEvaluator` — `_construct_h_planner` (builds TwoLvlPlanner,
  line ~73), `_perform_h_mpc` (hierarchical rollout, line ~216), `_perform_mpc` (flat rollout, line ~343).
- Schedule (config): `replan_every: 4`, `action_repeat: 4`, `step_skip: 4`, episode cap `n_steps: 500`.
- L1 MPPI: `num_samples` 125 (easy) / 250 (med,hard), `noise_sigma 5`, `lambda_ 0.0025`, `max_plan_length 128`.

## 4. ⭐ THE SUBGOAL SEAM (Stage-2 intervention point)
`two_lvl_planner.py::TwoLvlPlanner.plan`:
```python
l2_result = self.l2_planner.plan(current_state=backbone_output, plan_size=plan_size, repr_input=True)
...
self.l1_planner.reset_targets(l2_result.pred_obs[1].detach(), repr_input=True)   # <-- subgoal g
l1_result = self.l1_planner.plan(current_state=backbone_output, plan_size=self.l2_step_skip, repr_input=True, ...)
```
- **Subgoal `g` = `l2_result.pred_obs[1]`** — the high level's FIRST predicted latent (obs-component).
- It is set as the low level's target via `reset_targets` → `objective.set_target` → used by `RunningCost`.
- **Low-level cost / the distance to reuse for the residual** (`mppi_planner.py::RunningCost.__call__`):
  `diff = (state - target).pow(2); return diff.mean(-1)` — **mean-squared latent distance** to `target_enc`.
- ⇒ **`r_plan = mean((l1_result.pred_obs[-1] - g)**2)`** (predicted terminal latent under `a*` vs subgoal),
  using the SAME metric as the planner cost (per spec). `l1_result.pred_obs` shape `(plan_size+1, bs, ...)`,
  so `[-1]` = predicted terminal. `r_exec` = same distance but with the ACHIEVED (encoded) latent after execution.
- `pred_obs[1]` here is index into the horizon dim `(plan_size+1, ...)`; `[0]`=current, `[1]`=first predicted.

## 5. Success + difficulty knobs
- **Success is env-defined**: `mpc.py:556` reads `info["success"]` from the d4rl maze env (goal reached within
  the env's radius). Aggregated in `MPCData.success_history`.
- **Difficulty `L` = easy / medium / hard**, set by the start-target files (goal manhattan/block distance
  ~5–8 / 9–12 / 13–16). Also `min_block_radius`, `max_plan_length` (150/200/250). `n_envs=40` per level (repo
  default; spec wants N=100 — we can sample more start/target pairs from the same generator).
- Compute knob already present: L1 `mppi.num_samples` per level → makes the compute-matched comparison natural.

## 6. Reproduction targets (paper numbers) — Table 3 "Diverse Maze Across Task Horizons"
| Difficulty (goal dist D) | Flat PLDM | HWM (hier) |
|---|---|---|
| Easy  D∈[5,8]   | 100% | 100% |
| Medium D∈[9,12] | ~63%* | 95% |
| Hard  D∈[13,16] | 44%  | 83% |
- Compute: HWM uses **~3–4× less** test-time compute than flat (paper's efficiency claim).
- Headline: hardest setting hierarchy ≈ doubles success **44% → 83%**.
- ⚠️ Pulled from arXiv HTML via summarizer (some results are chart-images). Easy (100/100) and Hard
  (44/83) agree across two fetches = FIRM anchors. *Medium-flat disputed (63% vs 84%); "+32pp at medium"
  quote ⇒ 63%. Verify exact digits vs the PDF, or just against our own M1 eval (our numbers = the check).
- Acceptance M1 = within ~5–10 pts of these, same qualitative curve (flat collapses at hard, hier holds).
- Source: arXiv 2604.03208, project page kevinghst.github.io/HWM.

## 7. 🔴 Feasibility (the decision)
- `.cuda()` hardcoded in: `two_lvl_planner.py` (5), `mppi_planner.py` (3), `objectives_v2.py` (2),
  `d4rl/hmpc.py` (1). ~11 sites → portable to `device`-agnostic, but that's a patch (violates "no refactor";
  acceptable as listed patches). This box: **arm64 mac, torch 2.10, MPS avail, CUDA false**.
- Env deps: `pldm_envs/diverse_maze/maze_draw.py` → `import d4rl.pointmaze` + `gym`; `requirements.txt` pins
  `d4rl==1.1`, `mujoco-py==2.1.2.14`, `mujoco==3.2.6`. **mujoco-py 2.1 on Apple Silicon is notoriously
  broken** (needs mujoco210 binary + cython + GL). Rendering 98×98 obs REQUIRES this — unavoidable for eval.
- **Conclusion**: running HWM's real eval here is high-risk/likely-infeasible. Options:
  1. **Linux + NVIDIA GPU (recommended)** — cloud box; repo runs closest to as-shipped; only download ckpts+ds.
  2. **Port to Mac/MPS** — patch `.cuda()`→device AND solve d4rl/mujoco-py on arm64 (the hard part). High risk.
  3. **Defer Stage 1** — build Stage-2 repair logic against a mocked seam / unit-test level now, wire to the
     real repo once compute is sorted. Keeps momentum but M1 (reproduction) stays open.

## Ground-rules setup (done)
- Our code will live in `zeus-p1/repair/` (residuals.py, repair_op.py, control_loop.py, eval_harness.py,
  plots.py). Upstream patches tracked in `zeus-p1/PATCHES.md`. No upstream refactor.
