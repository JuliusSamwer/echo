# ECHO — full project overview (problem, thesis, every rung, exact I/O)

*A reference to build rung 3 on. Every claim here is grounded in the code as of 2026-07-07.
Shapes are written as tensors; `D` = latent dim (=2 with the identity encoder), `A` = action dim (=2).*

---

## 0. The one-sentence thesis

> **A flat world model + sampling planner is enough to reach goals in short-horizon
> mazes, but breaks predictably as the horizon grows. The fix — temporal abstraction —
> should be *earned by planning*, not handed over as a fixed stride or discovered by
> self-supervised compression. Rung by rung we prove the floor, find where it breaks,
> add hierarchy, and finally ask whether the abstraction can *emerge* at the maze's
> decision points or wherever it is most useful for the overall performance.**

---

## 1. Problem setup

**Task.** Zero-shot, reward-free goal reaching. Given a start observation and a goal
observation, produce actions that drive the agent to the goal. No reward signal ever —
planning is done by imagining rollouts in a learned model and minimizing distance to the
goal in latent space.

**The invariant interface (the "seam").** Everything is three modules in a line:

```
obs ──E──► z ──P──► ẑ' ──MPC──► action
     encoder  forward model   planner
```

- **E — Encoder** (`echo/models/encoder.py`): `obs → z`. Invertible.
- **P — Forward model** (`echo/models/world_model.py`): `P(z, a) → Δz`, so `z' = z + P(z,a)`.
- **MPC — Planner** (`echo/planning/mpc.py`): sampling MPC that searches action sequences
  to minimize latent distance to the goal.

This seam is *load-bearing for the research*: each rung swaps exactly one module and
nothing else. Rung 2 swaps the planner (adds a hierarchical controller). Graduating to
pixels later swaps `E`. The rest stays byte-identical, so comparisons are apples-to-apples.

**Why it stays "maze-specific."** The observation is the agent's **position only** `(x,y)` —
**walls are not in the observation**. So a single flat `P` can only be correct for *one*
maze (the transition depends on walls it can't see). That limitation is deliberate: it is
exactly the motivation to later graduate `E` to a pixel/latent encoder where walls are
observable. At rungs 0–3 we train/eval on **one fixed maze with held-out (start, goal) pairs**.

---

## 2. The two environments

### 2a. `PointMaze` — the grid maze (rungs 0, 1, 2, and the *first* rung-3 experiments)
`echo/envs/point_maze.py`

- **Geometry:** a **perfect maze** (recursive-backtracker spanning tree) on a grid of unit
  cells. Unique path between any two cells ⇒ BFS cell-distance `L` is well defined. Position
  `p=(x,y)` lives in continuous `[0,W]×[0,H]`; cell `(cx,cy)` center is `(cx+.5, cy+.5)`.
- **Observation:** `obs = (x, y) ∈ ℝ²` (state-only; **no walls**). `obs_dim = 2`.
- **Action:** `a = (dx, dy)`, clipped to `[-a_max, a_max]`, `a_max = 0.4` (< 1 cell, so a step
  crosses at most one wall per axis). `act_dim = 2`.
- **Dynamics:** move per-axis (x then y), each blocked independently at walls → natural
  **wall sliding**. Deterministic.
- **Reward:** always 0. `done`/`info["success"]` fire when `‖p − goal‖ < goal_eps` (0.2 cells).
- **`L` (difficulty knob):** BFS optimal path length start→goal. **Rung 1 sweeps this.**

### 2b. `BottleneckMaze` — the certified rung-3 maze (new, `echo_maze/`)
Built from `~/Downloads/echo_env_buildspec.md`. This is a **cleaner rung-3 substrate** where
we can *prove* that boundary placement matters before training anything.

- **Geometry:** continuous **multi-room** maze — an `n_rooms_x × n_rooms_y` grid of rooms
  (default 3×3, `room_size=1.0`) separated by walls with **doorways** (gaps of
  `doorway_width`). Doorways are **off-center** (`doorway_jitter=0.34`) — this is the
  certified setting (see §7). Ground-truth doorway coordinates and a room-adjacency graph
  are known exactly.
- **Observation:** `obs = concat(state, goal) ∈ ℝ⁴` (full observability v0). State is `(x,y)`.
- **Action:** displacement, L2-clipped to `max_step = 0.14`. (`double_integrator` reserved.)
- **Dynamics:** vectorized ray-vs-wall-segment collision; move stops just before the nearest
  crossing. `step_batch(states[N,2], actions[N,2]) → next[N,2]` moves N candidates at once.
- **`L`:** number of doorways on the shortest **room**-path start→goal (reuses the difficulty axis).
- **Ground truth for probes:** `path_doorways(start,goal) → (rooms, doorway_centers)` gives the
  known-optimal boundary set (the doorways you *must* pass through).

---

## 3. The shared substrate — exact I/O

### E — `IdentityEncoder` (`encoder.py`)
Affine, invertible normalizer. **latent == normalized state.**
- `encode(obs) = (obs − mean) / scale` → `z`
- `decode(z) = z·scale + mean` → `obs`
- `from_bounds(low, high)`: `mean=(low+high)/2`, `scale=(high−low)/2`.
- `latent_dim = 2`. (An `MLPEncoder` exists only to make the graduation seam concrete; unused.)

### P — `ForwardModel` / `WorldModel` (`world_model.py`)
`P(z, a) → Δz`, **residual**: `z' = z + P(z,a)`. Architecture:
```
a_n = a / act_scale                       # act_scale = a_max, normalizes action to O(1)
Δz  = act_skip(a_n)  +  MLP([z, a_n])      # act_skip: linear, zero-init (learns free-space Δz≈a)
                                           # MLP: n_layers=3, hidden=256, GELU, zero-init head
```
- The linear skip captures trivial free-space motion; the MLP only fits the **sparse wall
  corrections** (the hard discontinuity where a passage is blocked). Zero-init ⇒ model
  starts at `Δz=0` (standard residual init).
- `step(z,a) = z + P(z,a)`. `rollout(z0[B,D], actions[B,T,A]) → [B,T,D]` (autoregressive; the
  hot loop the planner hammers).

### Training P — `train_world_model` (`train.py`)
Offline transitions sliced into windows of `(K+1 states, K actions)`, `K = rollout_len = 8`.
```
Loss = w_onestep · mean‖(z_in + P(z_in, a)) − z_tgt‖²      # teacher-forced 1-step, all K
     + w_rollout · mean‖ autoregressive_rollout − z_tgt‖²   # K-step unrolled from z_0
```
Everything in latent space. Adam (lr 1e-3, cosine), `epochs=200` (full). **Only `P` is
trained; `E` is fixed.** Achieved per-step error ≈ **0.09–0.11 cells** (see `maze_*.pt`).

### MPC — `plan(...)` (`mpc.py`), CEM or MPPI
**Inputs:** `model`, world-space `start` (+ `goal` or `goal_latent`), horizon `H`, samples `N`.
**Procedure (per call):**
1. Encode `z0 = E(start)`, `z_goal = E(goal)` (or use `goal_latent` directly — subgoals live in
   latent space, so this is unchanged under a visual encoder later).
2. Sample `N` action sequences `(N,H,A) ~ clip(mean + std·𝒩, −a_max, a_max)`.
3. Roll the model: `traj = rollout(z0, samples) → (N,H,D)`; `dist = ‖traj − z_goal‖ → (N,H)`.
4. **Cost (goal energy), `cost_mode`:**
   - `"terminal"`: `dist[:, −1]` (final-step distance).
   - `"min"` (**default**): `dist.min(dim=1)` — "did the rollout *ever* reach the goal?" Robust
     to overshoot and to wall detours (a straight terminal cost gets stuck pressing walls).
   - `"terminal+mean"`: terminal + `running_cost_w`·mean.
5. Refine: **CEM** keeps `elite_frac` best, refits `mean/std` (std floored at `min_std_frac·a_max`
   so elites keep exploring); **MPPI** does softmax-weighted mean. `cem_iters=5`.
6. Return best action sequence `(H, A)`. `return_residual=True` also returns the best achievable
   latent distance — the **reachability** signal reused everywhere downstream.

**`RecedingHorizonController`:** plan `H`, execute `exec_k` (=1), warm-start the mean (shift
forward), replan. Has **stuck-escape**: after `patience` stalled steps, cold-start the plan and
inject extra exploration for a few replans (converts deadlock failures into successes).
`cost_per_plan = N·H` is the reported planner-cost metric.

---

## 4. Rung 0 — the floor  ✅ PASSED

**Claim:** flat `E→P→MPC` clears easy mazes ≥ 95%.
**Setup:** identity `E`, one flat `P`, `RecedingHorizonController`. One fixed maze, held-out
(start,goal). **Gate:** success ≥ 95%. Passed (see `figures/successful_run.*`).
**Diagnostics (in priority order) when it fails:** (1) data coverage, (2) model `rollout_mse`,
(3) planner budget `H/N/cem_iters`. The rollout-error diagnostic is non-negotiable: it tells you
instantly whether a failure is the *model* or the *planner*.

---

## 5. Rung 1 — the failure curve  ✅ (regenerated 2026-07-06)

**Claim:** hold model + budget fixed; vary only `L`; flat success collapses → this env now
*needs* hierarchy. `scripts/rung1_failure_curve.py`.

- **Setup:** ONE 9×9 maze, ONE frozen flat model (RMSE@1 = 0.091), **fixed** planner budget
  `H=25, N=256, cem_iters=5`. Sweep `L ∈ {2,3,4,5,6,8,10,12}`, 20 held-out pairs each. Episode
  step-limit scales with `L` (so we measure planner capability, not a wall-clock cutoff).
- **Result (regenerated):**

  | L | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
  |---|---|---|---|---|---|---|----|----|
  | success | 100% | 85% | 80% | 65% | 70% | 50% | 10% | 5% |
  | rollout_mse | .014 | .15 | .29 | .58 | 1.87 | 1.86 | 3.30 | 5.93 |

  Monotone collapse; rollout error climbs ~400× across the sweep. **The proof that a fixed-budget
  flat planner cannot handle long horizons** — deceptive junctions (where the path must first move
  *away* from the goal for longer than `H`) defeat it. → figure `figures/rung1_failure_curve.png`.

---

## 6. Rung 2 — imposed hierarchy (HWM-style)  ✅ (substrate regenerated)

**Claim:** a learned **coarse world model** planning over **macro-actions** recovers long-horizon
success where flat collapses. This swaps only the **planner** (adds a high level); `E`, `P` seam
unchanged. Substrate = 10 per-maze flat models `models/learned_models_v2/maze_{0..9}.pt`
(`scripts/train_models.py`, 100% coverage, rmse@1 ≈ 0.11) — **rung 2 and rung 3 both load these.**

### The coarse model — `CoarseWorldModel` (`coarse_model.py`)
Fixed stride `k` (given, not learned). Three learned pieces, **shares the frozen encoder**:
- **`E_a` — macro-action encoder (VAE):** a `k`-step action chunk `(…,k,A) → q(m|a)=𝒩(μ,σ)`,
  `m ∈ ℝ^macro_dim`. KL-to-`𝒩(0,I)` makes the macro space **samplable** — the high-level planner
  draws `m ~ 𝒩(0,I)` and stays on the manifold where `G` is reliable.
- **`G` — coarse dynamics:** `(z_t, m) → Δz` **over k steps in one shot**: `z_{t+k} = z_t + G(z_t,m)`.
  One-shot is the whole point — re-rolling the flat model `k` times recompounds error (why the
  naive hierarchy failed).
- **`D_a` — macro decoder:** `m → k` actions, an anti-posterior-collapse reconstruction term.
- Interface: `coarse_step(z,m)`, `coarse_rollout(z0, macros[B,M,macro_dim]) → [B,M,D]`.

### The controller — `HWMController` (`planning/hwm_hierarchical.py`)
Two-level receding horizon:
1. **High level (CEM over macro-latents):** sample macros `(N_hi, M, macro_dim) ~ 𝒩(mean, std)`;
   roll the coarse model → waypoints `(N_hi, M, D)`; cost = `min_m ‖waypoint − z_goal‖`; refit
   elites. **Subgoal = the first coarse-predicted latent** `z_sub = coarse_step(z0, m0)`.
2. **Low level:** flat `plan(low, goal_latent=z_sub, H=H_lo, N=N_lo, cost_mode="min")` produces
   primitives to reach the subgoal.
3. **Replan** the subgoal when it's reached (`‖z_t − z_sub‖ < subgoal_eps`) or the timer hits `k`.
   Total cost = `N_lo·H_lo·n_plans + high-level cost`.

- **Eval:** `scripts/rung2_learned_rigorous.py` — paired flat-vs-hier over 10 mazes × pairs, 95%
  CIs (`figures/rung2_learned_rigorous.png`). (⏸️ not re-run since regen; substrate is ready.)

---

## 7. Rung 3 — *emergent* abstraction (the frontier)

**The question rung 3 exists to answer:**
> Can the abstraction level — *where* to place temporal boundaries — **emerge**, landing at the
> maze's **decision points (junctions/doorways)**, and does that require **planning pressure**
> rather than self-supervised compression?

No fixed stride `k`. A **boundary mechanism** decides, per step, whether that step is a chunk
boundary. Chunk endpoints become the waypoints/subgoals. All of this trains on a **frozen
substrate** (`E`, `P1` never touched).

### The pieces
- **`BoundaryNet B` (`boundary.py`):** reads `[z_t, z_t − z_{t−1}] → logit → g_t ∈ [0,1]`.
  Gate via **Gumbel-sigmoid** (binary-concrete), with a straight-through `hard` option so it
  **self-sharpens** toward binary as temperature `τ` anneals. `g_t` = "is `t` a boundary?"
- **`EmergentCoarseModel` (`emergent_coarse.py`):** holds `B` + a VAE action encoder `E_a`
  (over a **chunk summary** `[mean action, log chunk-length]`, so it handles **variable-length**
  chunks) + `D_a` + coarse dynamics `G: (z_b, m) → Δz` to the *next boundary*. **Exposes the exact
  same planner-facing interface as `CoarseWorldModel`** (`coarse_step/coarse_rollout/macro_std/k`)
  ⇒ the rung-2 `HWMController` runs **unchanged**. Emergent-vs-imposed is a pure model swap.

### The metric — **junction lift**
`lift = P(junction | boundary) / P(junction | any step)`. `lift > 1` means boundaries
**cluster at decision points**. Scored against ground-truth junctions (grid maze) or
`path_doorways` (bottleneck maze).

### Three training regimes (the scientific contrast)
1. **SSL-only — the CONTROL** (`train_emergent.train_emergent`, "Approach 1"):
   - **sufficiency:** differentiable soft expected-next-boundary. From step `a`, chunk ends at
     `a+j` w.p. `p_j = g_{a+j}·∏_{i<j}(1−g_{a+i})`; loss `Σ_j p_j ‖z_a + G(z_a,m_j) − z_{a+j}‖`.
     This is what gives `B` gradient: `g` rises where the coarse model predicts well (shorter
     chunks predict better → `g` up; **rate** pushes `g` down).
   - **closure (Chapman–Kolmogorov):** on hard (straight-through) boundaries, roll `P2` across
     boundaries vs roll frozen `P1` across the real fine steps; coarse latents must match fine
     latents **at boundary times**. The load-bearing property.
   - **rate:** penalize mean gate (or `(mean g − target_rate)²` to *force* a chunk length).
   - **+ VAE:** KL to `𝒩(0,I)` + macro reconstruction (samplable macro space).
   - **Prediction (why it's the control):** even forced to chunk, boundaries land ~randomly,
     **lift ≈ 1.0** — SSL/compression alone *cannot* place boundaries at decision points.
2. **Planning-pressure — the TEST** (`train_emergent.train_planning_pressure`, "Step 6"):
   - `B` is driven by the **reachability residual** `_reachability(low, z_a, targets)`: greedily
     roll the frozen `P1` *straight at* each future latent target and read the distance after that
     target's step count. The residual is small when a chunk is reachable by a dumb low-level
     controller and **spikes when the chunk would have to go through a wall / round a turn** — that
     spike **is** the junction signal.
   - **B objective:** `L_plan = Σ_j p_j·(residual_j − β·steps_j)` → `p` concentrates on the
     *longest reachable* chunk (break only where extending would hit a wall). `P2` trained by
     sufficiency (gates **detached**) + closure, so only `B` moves under planning pressure. `τ`
     anneals to sharpen gates.
   - **Hypothesis:** this places boundaries at junctions, which SSL couldn't (**lift > 1**).
3. **REINFORCE probe — the de-risk** (`scripts/rung3_stepA.py`):
   - The single binary question: *does the **real** planning residual move boundaries toward
     junctions at all?* Uses the BFS **reference path** (its turns are the junctions), `B` samples
     boundaries → subgoals, a **bounded executor** (`H_lo`) hops subgoal→subgoal via the real frozen
     planner, reward `= −‖final − goal‖ − λ·(#boundaries)` (the count penalty forces *few, well-placed*
     boundaries). REINFORCE on `B`'s Bernoulli boundary samples. **PASS = junction lift > 1.15.**

### Status of rung 3
- The grid-maze steps 0–6 + A are written (`scripts/rung3_step*.py`, `rung3_stepA.py`) and were the
  active work before the model-loss incident. They load the frozen `maze_*.pt` substrate.
- **Open question still open:** whether planning pressure / the REINFORCE residual actually clears
  the junction-lift gate. That is the crux the whole rung hinges on.

---

## 8. The new certified rung-3 substrate (`echo_maze/`)

**Why it exists.** On the grid maze, "junction lift" is only meaningful *if placing a subgoal at a
junction actually helps vs. placing it elsewhere*. Nothing guaranteed that. `echo_maze` adds a
**certification probe** that proves placement is causally load-bearing **before any training** — for
the cost of zero gradient steps. Build order gates itself: env → executor → **probe (the gate)** →
frozen base → rung-3 interface.

### Pieces (all in `echo_maze/`)
- **`config.py` / `maze.py`:** the `BottleneckMaze` (§2b) + geometry, vectorized collision, lazy
  geodesic field, room-graph routing, ground-truth doorways.
- **`executor.py` — `BoundedMPPI`:** receding-horizon MPPI with a **hard step budget `H`**. Generic
  over a batched `step` fn, so the **same** executor drives the true env (probe) and the frozen
  learned model (rung 3). `reach(s, subgoal, H)` = try to reach a subgoal within `H` primitive steps;
  `execute_chain(start, subgoals, goal, H)` = hop through subgoals then the goal. **The step budget is
  what makes doorways bite.**
- **`probe.py` / `run_probe.py` — the gate.** Hand-place boundaries three ways, measure success under
  the bounded executor, **no learning**:
  - **GT** = the doorway centers on the shortest route (known-optimal placement).
  - **RANDOM** = `L` random free points.
  - **ADVERSARIAL** = the `L` route-room centers — *right route, wrong precise placement.*
  - **PASS** = `success(GT) ≥ 0.8` **and** `GT−RANDOM` and `GT−ADVERSARIAL` both > 0 with paired-
    bootstrap 95% CIs excluding 0, **at all L**.
- **`base_model.py` / `train_base.py`:** `ForwardModel` `s' = s + f(s,a)` (residual MLP, SiLU),
  trained on doorway-biased transitions, then **frozen**. `validate_base` gates on open-space error
  `< 1e-2` **and** wall-refusal (must not hallucinate through walls). → `models/base_frozen.pt`.
- **`rung3.py`:** interface **stubs only** (`BoundaryGate`, `CoarseModel`, `reachability_residual`) —
  bodies deliberately unwritten pending the design fork (§9).

### Certification result (100 instances/L, L∈{2,3,4}) — **PASSED**
Certified config = **off-center doorways** (`doorway_jitter=0.34`, `doorway_width=0.18`), **`H=10`**,
`MPPIConfig(n_samples=384, n_iters=3, plan_horizon=14)`.

| L | GT | RANDOM | ADV | GT−ADV (95% CI) |
|---|----|--------|-----|-----------------|
| 2 | 0.92 | 0.01 | 0.59 | +0.33 [+0.22,+0.44] |
| 3 | 0.80 | 0.00 | 0.53 | +0.27 [+0.14,+0.40] |
| 4 | 0.82 | 0.00 | 0.38 | +0.44 [+0.32,+0.56] |

`PASS at H in [10]` only — a narrow, well-isolated **placement-critical band**: `H=8` leaves GT
unreliable (band floor); `H≥11` collapses the GT−ADV gap because the low-level threads doorways
unaided (at `H=16`, ADV=1.00). **Why off-center matters:** with *centered* doorways a room-center
subgoal is collinear with the gap, so ADVERSARIAL threads the door for free and placement never
bites — jitter breaks that collinearity. Full record: `echo_maze/CERTIFIED.md`,
`figures/probe_overlay.png`, `models/results/echo_maze_probe.log`.

---

## 9. Where we are & the open decision

**Done:** env certified (placement is load-bearing — the gate rung 3 needed). **Not done:**
(1) train + freeze + validate the base forward model (`python -m echo_maze.train_base`); (2) write
the `rung3.py` bodies + the 3-condition measurement harness (success vs `L`; junction lift vs
`path_doorways`; coarse-only closure; boundary count vs `L`). **Decision gate for rung 3:** match the
HWM-hier baseline within CIs **and** junction lift > 1.

**The one design fork to settle before writing rung-3 bodies — the placement signal:**
- **(A) Learned differentiable surrogate `R_ψ(s,g)`** — a small net trained to match the bounded
  executor's reachability residual over the frozen base, differentiable in `g`, so gradients flow
  straight into the boundary net. Clean gradients; costs one extra model to fit/validate.
- **(B) REINFORCE** on boundary samples (what grid-maze `stepA` used). No extra model; higher variance,
  slower convergence.

Both are named in `echo_maze/rung3.py`. The env, executor, frozen-base interface, and ground-truth
junctions are all in place for whichever we pick.

---

## Appendix — file map

| Path | What |
|---|---|
| `echo/envs/point_maze.py` | grid perfect-maze env (rungs 0–2, first rung-3) |
| `echo/models/encoder.py` | `IdentityEncoder` (E) |
| `echo/models/world_model.py` | `ForwardModel`/`WorldModel` (P), residual, rollout |
| `echo/train.py` | flat-P training (1-step + K-step rollout loss) |
| `echo/planning/mpc.py` | `plan` (CEM/MPPI), `RecedingHorizonController` |
| `echo/models/coarse_model.py` | imposed-`k` HWM coarse model (rung 2) |
| `echo/planning/hwm_hierarchical.py` | `HWMController` (two-level planner) |
| `echo/models/boundary.py` | `BoundaryNet` B (Gumbel-sigmoid gate) |
| `echo/models/emergent_coarse.py` | emergent coarse model (B + E_a + G) |
| `echo/train_emergent.py` | SSL, planning-pressure, `_reachability` |
| `scripts/train_models.py` | trains the 10-maze substrate `maze_*.pt` |
| `scripts/rung1_failure_curve.py` | rung-1 sweep |
| `scripts/rung2_learned_rigorous.py` | rung-2 paired eval |
| `scripts/rung3_step{0..6}.py`, `rung3_stepA.py` | rung-3 de-risking ladder |
| `echo_maze/` | certified bottleneck-maze substrate (env, executor, probe, base, stubs) |
| `echo_maze/CERTIFIED.md` | rung-3 env certification record |
| `models/`, `models/results/` | persistent checkpoints + logs (never scratchpad) |
