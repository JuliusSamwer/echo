"""Phase 2 — calibration & the two sanity checks. Runs on the per-segment CSV from the instrumented HWM.

  python -m repair.calibrate --csv runs/hwm_segments.csv --moderate_L 2 --out repair/CALIBRATION.md

Produces:
  - Sanity 1: Spearman(r_plan, r_exec). Low/insignificant => r_plan is NOT a usable reachability proxy
    (project-level RISK — flagged loudly).
  - Sanity 2: distribution of r_plan on segments of FAILED vs SUCCEEDED episodes; Mann-Whitney one-sided
    (failed stochastically larger). Overlap => switch to the normalized residual.
  - tau = 90th percentile of r_plan over segments of SUCCESSFUL episodes at moderate L (global, per-config).
  - figures: r_plan distributions (succ vs fail); r_plan-vs-r_exec scatter.
  - CALIBRATION.md summarizing choices + the M2 verdict.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("r_plan", "r_exec", "r_plan_norm"):
            r[k] = float(r[k])
        for k in ("L", "segment_succeeded", "episode_succeeded"):
            r[k] = int(float(r[k]))
    return rows


def _spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()); ry = (ry - ry.mean())
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _mannwhitney_u_greater(a, b):
    """One-sided: is `a` stochastically GREATER than `b`? Returns AUC (P(a>b)) in [0,1]; ~0.5 = overlap."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b]); ranks = np.argsort(np.argsort(allv)) + 1
    ra = ranks[:len(a)].sum()
    U = ra - len(a) * (len(a) + 1) / 2
    return float(U / (len(a) * len(b)))          # AUC = P(a > b)


def calibrate(csv_path, moderate_L, out_md, key="r_plan"):
    rows = load(csv_path)
    rp = [r[key] for r in rows]; re = [r["r_exec"] for r in rows]
    rho = _spearman(rp, re)

    succ = [r[key] for r in rows if r["episode_succeeded"] == 1]
    fail = [r[key] for r in rows if r["episode_succeeded"] == 0]
    auc = _mannwhitney_u_greater(fail, succ)      # want > 0.5 (failed segments have larger r_plan)

    succ_modL = [r[key] for r in rows if r["episode_succeeded"] == 1 and r["L"] == moderate_L]
    tau = float(np.percentile(succ_modL, 90)) if succ_modL else float("nan")

    try:
        _plots(rows, key, csv_path)
    except Exception as e:
        print(f"[calibrate] plots skipped: {e}")

    verdict = ("GREEN" if (rho > 0.3 and auc > 0.6) else
               "YELLOW (heavy overlap -> use normalized residual r_plan_norm)" if rho > 0.15 else
               "RED (r_plan not a usable reachability proxy — do NOT proceed)")
    md = f"""# CALIBRATION.md — Phase 2 (residual calibration)

Source: `{csv_path}`  · key: `{key}`  · moderate L = {moderate_L}  · n_segments = {len(rows)}

## Sanity 1 — r_plan vs r_exec correlation (must be positive)
Spearman rho = **{rho:.3f}**  (want > ~0.3). {'OK' if rho > 0.3 else 'WEAK/FAIL — flag'}.

## Sanity 2 — r_plan larger on FAILED episodes (must separate)
P(r_plan_fail > r_plan_succ) = **{auc:.3f}**  (want > 0.5; >0.6 = clean separation).
n_succ_segments={len(succ)}  n_fail_segments={len(fail)}. See `figures/calib_rplan_dist.png`.

## Trigger threshold
tau = 90th pct of {key} over SUCCESSFUL episodes at L={moderate_L}  ->  **tau = {tau:.4f}**
(store per-config; sweep {{p75,p90,p97}} in Phase 3.4).

## Verdict: {verdict}
"""
    Path(out_md).write_text(md)
    print(md)
    return dict(rho=rho, auc=auc, tau=tau, verdict=verdict)


def _plots(rows, key, csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = Path(csv_path).resolve().parent.parent / "figures"; figdir.mkdir(exist_ok=True)
    succ = [r[key] for r in rows if r["episode_succeeded"] == 1]
    fail = [r[key] for r in rows if r["episode_succeeded"] == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.percentile([r[key] for r in rows], 99), 40)
    ax.hist(succ, bins=bins, alpha=0.6, density=True, label="succeeded episodes", color="royalblue")
    ax.hist(fail, bins=bins, alpha=0.6, density=True, label="failed episodes", color="crimson")
    ax.set_xlabel(key); ax.set_ylabel("density"); ax.legend()
    ax.set_title("Calibration: r_plan on succeeded vs failed episodes")
    fig.savefig(figdir / "calib_rplan_dist.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([r[key] for r in rows], [r["r_exec"] for r in rows], s=6, alpha=0.3)
    ax.set_xlabel("r_plan (pre-exec)"); ax.set_ylabel("r_exec (post-exec)")
    ax.set_title("r_plan vs r_exec (reachability proxy check)")
    fig.savefig(figdir / "calib_rplan_vs_rexec.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[calibrate] wrote figures -> {figdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--moderate_L", type=int, default=2)
    ap.add_argument("--out", default="repair/CALIBRATION.md")
    ap.add_argument("--key", default="r_plan", choices=["r_plan", "r_plan_norm"])
    a = ap.parse_args()
    calibrate(a.csv, a.moderate_L, a.out, a.key)
