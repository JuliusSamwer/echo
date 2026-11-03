# echo_maze — certification record

**Status: CERTIFIED** (placement is causally load-bearing). Probe passed 2026-07-06.

## Certified config (now the package defaults)
- Geometry: `doorway_jitter=0.34`, `doorway_width=0.18`, 3×3 rooms, `room_size=1.0`, `max_step=0.14`.
- Executor: **H=10** step budget, `MPPIConfig(n_samples=384, n_iters=3, plan_horizon=14)`.
- Probe: 100 instances/L, L∈{2,3,4}, paired bootstrap 95% CIs.

## Result (H=10, the certified operating point)
| L | GT | RANDOM | ADV | GT−ADV (95% CI) |
|---|----|--------|-----|-----------------|
| 2 | 0.92 | 0.01 | 0.59 | +0.33 [+0.22, +0.44] |
| 3 | 0.80 | 0.00 | 0.53 | +0.27 [+0.14, +0.40] |
| 4 | 0.82 | 0.00 | 0.38 | +0.44 [+0.32, +0.56] |

GT ≥ 0.80 at all L; GT−RANDOM and GT−ADVERSARIAL both strictly positive with CIs excluding 0. PASS.

## The band (why H=10 specifically)
`PASS at H in [10]` only. H=8 → GT unreliable (band floor). H≥11 → GT−ADV gap collapses
as the bounded low-level threads doorways unaided (H=16: ADV=1.00, gap ≤0). Narrow band,
exactly as predicted: placement matters only when one hop covers ~one room but cannot skip a door.

## Why off-center doorways were required
With centered doorways a doorway sits on the shared center-line of its two rooms, so a
room-center subgoal (ADVERSARIAL) is collinear with the gap and gets threaded for free →
placement goes free and the probe (correctly) FAILs. `doorway_jitter=0.34` breaks that
collinearity, which is what makes precise doorway placement causally matter.

## Artifacts
- Full sweep log: `models/results/echo_maze_probe.log`
- Qualitative overlay (GT threads doors vs ADV stalls): `figures/probe_overlay.png`

## Next (not yet done)
1. Phase 4 — train + freeze + validate the base forward model (`python -m echo_maze.train_base`).
2. Rung-3a — implement `rung3.py` bodies after the placement-signal design call (surrogate R_ψ vs REINFORCE).
