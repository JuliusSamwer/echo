# CALIBRATION.md — Phase 2 (residual calibration)

Source: `runs/hwm_segments.csv`  · key: `r_plan`  · moderate L = 2  · n_segments = 528

## Sanity 1 — r_plan vs r_exec correlation (must be positive)
Spearman rho = **0.602**  (want > ~0.3). OK.

## Sanity 2 — r_plan larger on FAILED episodes (must separate)
P(r_plan_fail > r_plan_succ) = **nan**  (want > 0.5; >0.6 = clean separation).
n_succ_segments=0  n_fail_segments=0. See `figures/calib_rplan_dist.png`.

## Trigger threshold
tau = 90th pct of r_plan over SUCCESSFUL episodes at L=2  ->  **tau = nan**
(store per-config; sweep {p75,p90,p97} in Phase 3.4).

## Verdict: YELLOW (heavy overlap -> use normalized residual r_plan_norm)
