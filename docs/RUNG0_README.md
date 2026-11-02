# Zeus — Rung 0

The floor of a world-model planning ladder. **No abstraction, no hierarchy, no
latent s-axis, no learned encoder.** Just a flat forward model plus a sampling
planner that has to solve easy mazes near-100%. If this doesn't hold, nothing
above it can. The point of rung 0 is what it leaves out: strip everything
exciting, prove the loop closes.

## The interface (so it graduates without a rewrite)

```
obs ──E──► z ──P──► ẑ' ──MPC──► action
     encoder  forward model   planner
```

Same `E → P → MPC` seam the PLDM / DINO-WM / HWM stack uses. At rung 0:

- **E** (`models/encoder.py`) is a trivial *normalizing identity* — affine,
  invertible, `latent == state`. The seam is real, so swapping in a learned
  PLDM/DINO-WM encoder later is a swap, not a rewrite.
- **P** (`models/world_model.py`) is one small MLP, `P(z, a) → Δz`, residual.
- **MPC** (`planning/mpc.py`) is CEM or MPPI, goal-conditioned, receding horizon.

## The five pieces

| # | Piece | File | What it is |
|---|-------|------|-----------|
| 1 | Environment | `envs/point_maze.py` | ~Hand-rolled continuous point-maze, state obs. Parameterized by **L = optimal path length** (the knob rung 1 sweeps). Gym-like API → PLDM Diverse Maze is the swap-in target. |
| 2 | Data | `data/collect.py` | Offline noisy random-walk trajectories. No rewards. |
| 3 | Forward model | `models/world_model.py` | Flat MLP, 1-step + short autoregressive rollout loss (`train.py`). |
| 4 | Planner | `planning/mpc.py` | Sampling MPC (CEM/MPPI), energy `‖z_goal − rollout_final‖`. |
| 5 | Eval harness | `eval/harness.py` | success rate **+ rollout error + planner cost** (the two diagnostics). |

## Run it

```bash
pip install -r requirements.txt          # torch + numpy, nothing else
python tests/test_smoke.py               # unit checks (walls block, shapes, roundtrip)
python scripts/run_rung0.py --smoke      # ~seconds; proves the loop closes
python scripts/run_rung0.py --full       # the real gate run
python scripts/run_rung0.py --full --planner mppi --path-len 4
```

## The gate

**Rung 0 passes when flat model + planner clears easy mazes ≥ 95%.** Only then do
you touch rung 1 (fix capacity + planner budget, sweep `L`, find where flat breaks
— the failure curve). The runner prints PASS/FAIL.

### When it fails, the diagnostics tell you which of three — in order

The second diagnostic (model rollout error) is **non-negotiable**: it tells you
instantly whether a failure is the model or the planner, instead of guessing.

1. **Data coverage** — model never saw that region. → `coverage` < ~100%; raise
   `goal_directed_frac` / `n_train_episodes`.
2. **Model accuracy** — `rollout_mse` high. → more data / capacity / epochs.
3. **Planner budget** — `rollout_mse` low but planning still fails. → raise
   `H` / `n_samples` / `cem_iters`.

Fix the one the diagnostics point at; don't move on.

## A note on "easy mazes" (plural) vs. one maze

Rung 0 trains and evals on **one fixed maze with held-out (start, goal) pairs**.
Why not many mazes? With **state-only** obs `(x, y)`, the transition depends on
walls that aren't in the observation, so a single flat `P` is necessarily
maze-specific. That limitation *is* the motivation for graduating to a
pixel/latent encoder (where walls are observable) — exactly the s-axis the upper
rungs add. The code accepts multiple mazes for when rung 1 sweeps `L`; just
retrain per maze (or graduate E) so the model can see what it's planning in.

## Config

Everything is one dataclass (`config.py`); a run is fully described by it.
`Rung0Config.smoke()` / `.full()` are the two presets. Key knobs the spec calls
out are exposed: `path_len (L)`, `horizon (H)`, `n_samples (N)`, `cem_iters`,
`elite_frac`, `exec_k`.
