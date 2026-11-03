"""Phase 3.6 — the exact figure list, produced from the per-condition segment/episode CSVs.
  python -m repair.plots --runs runs/ --out figures/
Figures: (1) success vs L w/ CI bands (headline), (2) success vs planning compute per failure-L,
(3) r_plan succ-vs-fail dist (from calibrate), (4) r_plan before-vs-after repair scatter, (5) trigger/
escalation rates vs L, (6) one qualitative rollout (coarse plan, failed subgoal, repaired, executed path).
Figs 3 is emitted by calibrate.py; 6 needs a saved rollout trace from eval_harness (--dump_rollout)."""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
from repair.stats import wilson_ci


def _load_segments(runs_dir):
    rows = []
    for f in glob.glob(f"{runs_dir}/*_segments.csv"):
        for r in csv.DictReader(open(f)):
            rows.append(r)
    return rows


def _episode_success_by_method_L(rows):
    """-> {(method,L): [0/1 episode outcomes]} deduped by (method,env_idx,episode)."""
    seen = {}
    for r in rows:
        key = (r["method"], int(float(r["L"])), int(float(r["env_idx"])), int(float(r["episode"])))
        seen[key] = int(float(r["episode_succeeded"]))
    out = defaultdict(list)
    for (m, L, _e, _ep), s in seen.items():
        out[(m, L)].append(s)
    return out


def plot_all(runs_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir); out.mkdir(exist_ok=True)
    rows = _load_segments(runs_dir)
    if not rows:
        print(f"[plots] no *_segments.csv in {runs_dir}"); return
    byML = _episode_success_by_method_L(rows)
    methods = sorted({m for (m, _L) in byML}); Ls = sorted({L for (_m, L) in byML})

    # (1) success vs L, CI bands
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in methods:
        ys, los, his = [], [], []
        for L in Ls:
            o = byML.get((m, L), [])
            p, lo, hi = wilson_ci(int(sum(o)), len(o)) if o else (np.nan, np.nan, np.nan)
            ys.append(p); los.append(p - lo); his.append(hi - p)
        ax.errorbar(Ls, ys, yerr=[los, his], marker="o", capsize=3, label=m)
    ax.set_xlabel("horizon L"); ax.set_ylabel("success"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7); ax.set_title("Success vs L (Wilson 95% CI)")
    fig.savefig(out / "fig1_success_vs_L.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # (2) success vs compute (per failure-L): total fwd passes per episode vs success
    comp = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [fwd_sum, n_seg] per (method,L,env,ep)
    ep_succ = {}
    for r in rows:
        m, L = r["method"], int(float(r["L"])); e, ep = int(float(r["env_idx"])), int(float(r["episode"]))
        fwd = float(r.get("fwd_passes_low", 0)) + float(r.get("fwd_passes_high", 0))
        comp[(m, L)][(e, ep)][0] += fwd; comp[(m, L)][(e, ep)][1] += 1
        ep_succ[(m, L, e, ep)] = int(float(r["episode_succeeded"]))
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in methods:
        for L in Ls:
            eps = comp.get((m, L), {})
            if not eps:
                continue
            fwd = np.mean([v[0] for v in eps.values()])
            succ = np.mean([ep_succ[(m, L, e, ep)] for (e, ep) in eps])
            ax.scatter(fwd, succ, s=60); ax.annotate(f"{m}·L{L}", (fwd, succ), fontsize=6)
    ax.set_xlabel("planning compute (fwd passes / episode)"); ax.set_ylabel("success")
    ax.set_title("Success vs compute (compute-matched comparison)")
    fig.savefig(out / "fig2_success_vs_compute.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # (4) r_plan before vs after repair
    bp = [(float(r["r_plan"]), float(r["r_plan_after_repair"])) for r in rows
          if float(r.get("r_plan_after_repair", -1)) >= 0]
    if bp:
        b, a = zip(*bp)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(b, a, s=8, alpha=0.4); lim = max(max(b), max(a))
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("r_plan before repair"); ax.set_ylabel("r_plan after repair")
        ax.set_title("Repair reduces r_plan (below identity = good)")
        fig.savefig(out / "fig4_repair_before_after.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # (5) trigger/escalation rates vs L
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in methods:
        rep, esc = [], []
        for L in Ls:
            sub = [r for r in rows if r["method"] == m and int(float(r["L"])) == L]
            n = max(len(sub), 1)
            rep.append(sum(int(float(r.get("n_repairs", 0))) for r in sub) / n)
            esc.append(sum(int(float(r.get("n_escalations", 0))) for r in sub) / n)
        ax.plot(Ls, rep, marker="o", label=f"{m} repair/seg")
        ax.plot(Ls, esc, marker="s", ls="--", label=f"{m} escal/seg")
    ax.set_xlabel("L"); ax.set_ylabel("rate per segment"); ax.legend(fontsize=6)
    ax.set_title("Trigger / escalation rates vs L")
    fig.savefig(out / "fig5_rates_vs_L.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[plots] wrote figures -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/")
    ap.add_argument("--out", default="figures/")
    a = ap.parse_args()
    plot_all(a.runs, a.out)
