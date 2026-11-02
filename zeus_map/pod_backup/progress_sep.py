"""Step-2 offline analysis: does GOAL-PROGRESS-STALL separate the failing episodes,
where reachability (AUC~0.64) could not?

Reads:
  --progress runs/diag_progress.csv   (per-step: condition,L,env_idx,step,dist_to_goal)
  --caa      runs/diag_caa.csv         (per-episode closest_approach -> fail label)

For each episode builds the dist-to-true-goal trajectory (main planning stage), computes
several stall/progress metrics, and reports AUC(fail>succ) for each. AUC far from 0.5
(either direction) => that metric separates failures. Compare best here vs reachability's ~0.64.

Usage: python progress_sep.py --progress runs/diag_progress.csv --caa runs/diag_caa.csv
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np


def fail_labels(caa_path, thr=0.3):
    by = defaultdict(list)
    for r in csv.DictReader(open(caa_path)):
        by[int(r["env_idx"])].append(float(r["closest_approach"]))
    return {e: (1 if min(v) >= thr else 0) for e, v in by.items()}


def trajectories(prog_path):
    # env_idx -> list of (step, dist); take the LONGEST stage per env (the main planning rollout)
    raw = defaultdict(lambda: defaultdict(list))   # env -> step -> [dist,...]
    for r in csv.DictReader(open(prog_path)):
        raw[int(r["env_idx"])][int(r["step"])].append(float(r["dist_to_goal"]))
    out = {}
    for e, steps in raw.items():
        d = np.array([np.mean(steps[t]) for t in sorted(steps)])
        out[e] = d
    return out


def metrics(d):
    T = len(d)
    if T < 3:
        return {}
    start, final, mn = float(d[0]), float(d[-1]), float(d.min())
    third = max(1, T // 3)
    return dict(
        final_dist=final,
        total_closed=start - final,
        frac_closed=(start - final) / (start + 1e-9),
        late_backslide=final - mn,                       # how far it drifted after its best
        stall_frac=float(np.mean(d[1:] >= d[:-1] - 1e-6)),  # fraction of non-improving steps
        last_third_closed=float(d[-third] - final),       # progress in final third
        progress_rate=(start - final) / T,
        min_dist=mn,
    )


def auc_fail_gt(vals, lab):
    fv = [vals[e] for e in vals if lab.get(e) == 1]
    sv = [vals[e] for e in vals if lab.get(e) == 0]
    if not fv or not sv:
        return float("nan")
    c = sum(1 for a in fv for b in sv if a > b)
    t = sum(1 for a in fv for b in sv if a == b)
    return (c + 0.5 * t) / (len(fv) * len(sv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", default="runs/diag_progress.csv")
    ap.add_argument("--caa", default="runs/diag_caa.csv")
    a = ap.parse_args()
    lab = fail_labels(a.caa)
    traj = trajectories(a.progress)
    M = {e: metrics(traj[e]) for e in traj if len(traj[e]) >= 3}
    keys = sorted({k for m in M.values() for k in m})
    nf = sum(lab.get(e, 0) for e in M)
    print(f"episodes={len(M)}  fail={nf}  succ={len(M)-nf}   [reachability baseline AUC ~0.64]")
    print(f"{'metric':<20}{'AUC(fail>succ)':<16}separates?")
    best = (0.5, "")
    for k in keys:
        vals = {e: M[e][k] for e in M if k in M[e]}
        au = auc_fail_gt(vals, lab)
        sep = "" if au != au else ("STRONG" if abs(au - 0.5) >= 0.25 else ("weak" if abs(au - 0.5) >= 0.12 else "no"))
        print(f"{k:<20}{au:<16.3f}{sep}")
        if au == au and abs(au - 0.5) > abs(best[0] - 0.5):
            best = (au, k)
    print(f"\nBEST separator: {best[1]}  AUC={best[0]:.3f}  "
          f"(vs reachability ~0.64 -> {'BETTER, reframe validated' if abs(best[0]-0.5) > 0.14 else 'not clearly better'})")


if __name__ == "__main__":
    main()
