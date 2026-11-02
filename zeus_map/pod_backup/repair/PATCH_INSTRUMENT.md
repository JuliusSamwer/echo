# Instrumentation patch — the ONE upstream hook (Phase 2)

All residual logging is a single passive call inside the per-segment loop. Zero effect on baseline
numbers: it runs only when `repair.residuals.set_logger(...)` has installed a logger, and reads only
DETACHED tensors. Unit test (c) asserts instrumented `hwm` reproduces uninstrumented trajectories
bit-exact under a fixed seed.

## Where
`pldm/planning/mpc.py :: MPCEvaluator._perform_mpc`, inside `for i in tqdm(range(max_steps), ...)`,
at the replan branch `if i % self.config.replan_every == 0:` — right after:

```python
planning_result = planner.plan(...)          # TwoLvlPlanningResult when bilevel_planning
if bilevel_planning:
    planning_result_l2 = planning_result.level2
    planning_result   = planning_result.level1
```

## The hook (add these lines; nothing else changes)
```python
from repair import residuals as _resid            # top of file

# --- inside the replan branch, after the level1/level2 split, bilevel only ---
if bilevel_planning and _resid.get_logger() is not None:
    # r_plan is computed here; r_exec/segment_succeeded are filled at the NEXT replan (segment end)
    self._pending_seg = dict(
        l2_result=planning_result_l2, l1_result=planning_result,
        g=residuals.subgoal_from_l2(planning_result_l2).detach(),
        segment=i // self.config.replan_every,
    )

# --- at segment end (next replan, or loop end): we have the achieved obs of the previous segment ---
# encode current obs -> z_achieved_obs (obs-component), then:
if bilevel_planning and _resid.get_logger() is not None and getattr(self, "_pending_seg", None):
    ps = self._pending_seg
    _resid.record_segment(
        method=self._method, L=self._L, maze_id=self._maze_ids, env_idx=self._env_base,
        episode=self._episode, segment=ps["segment"],
        l2_result=ps["l2_result"], l1_result=ps["l1_result"],
        z_achieved_obs=z_achieved_obs, z_current_obs=z_prev_obs,
        segment_succeeded=(_resid.r_exec_from_achieved(z_achieved_obs, ps["l2_result"])
                           < self._subgoal_eps).int().tolist(),
    )
```
`z_achieved_obs` = obs-component of `self.model.backbone(<current obs>)` — the encoder is already invoked
for the next plan(); reuse its `backbone_output.obs_component` (no extra forward). `self._method`,
`self._L`, `self._maze_ids`, `self._env_base`, `self._episode`, `self._subgoal_eps` are set by
`eval_harness` when it constructs the evaluator (a few attribute assignments, no logic change).

At episode end call `logger.set_episode_success(env_idx, episode, info["success"])` for each env.

## Phase 3 (repair) hook — same site, guarded by condition
When `self._condition != "hwm"`, replace `planning_result` (level1's target subgoal) via
`control_loop.decide_segment(...)`: call `l1_planner.reset_targets(g_exec)` and re-plan the low level
once against the (possibly repaired/replanned) `g_exec` before executing. `repair_fn` builds
`make_roll_high(model, macros_remaining)` from `planning_result_l2.actions[:, 1:]`; `replan_high_fn`
re-invokes `h_planner.l2_planner.plan(...)`. All extra forwards tallied into `fwd_passes_high`.
