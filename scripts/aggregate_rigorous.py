"""Aggregate sharded rigorous results -> failure/recovery curve with 95% CIs.

Reports, per L: flat and hier success mean +/- 95% CI (Wilson), and the PAIRED
hier-flat difference with its 95% CI. A difference 'counts' only when its CI
excludes 0 (the bar the critique set). Saves figures/rung2_rigorous.png + a table.
"""
import sys, csv, glob, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(exist_ok=True)
DIR = sys.argv[1] if len(sys.argv) > 1 else None
LABEL = sys.argv[2] if len(sys.argv) > 2 else "EXACT model"          # e.g. "LEARNED model"
PNG_NAME = sys.argv[3] if len(sys.argv) > 3 else "rung2_rigorous.png"
assert DIR, "usage: aggregate_rigorous.py <shard-dir> [label] [out.png]"

rows = []
for fp in glob.glob(str(Path(DIR) / "shard_*.csv")):
    with open(fp) as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append((int(d["maze"]), int(d["L"]), int(d["pair"]), int(d["flat"]), int(d["hier"])))
assert rows, f"no rows found in {DIR}"
Ls = sorted(set(r[1] for r in rows))


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return p, center - half, center + half


def paired_diff_ci(diffs, z=1.96):
    d = np.asarray(diffs, dtype=float)
    n = len(d); m = d.mean()
    se = d.std(ddof=1)/math.sqrt(n) if n > 1 else float("nan")
    return m, m - z*se, m + z*se, n


print(f"{'L':>3} | {'n':>4} | {'flat % (95% CI)':>20} | {'hier % (95% CI)':>20} | {'hier-flat (95% CI)':>22} | sig")
print("-" * 100)
data = {}
for L in Ls:
    rl = [r for r in rows if r[1] == L]
    n = len(rl)
    fk = sum(r[3] for r in rl); hk = sum(r[4] for r in rl)
    fp, flo, fhi = wilson(fk, n); hp, hlo, hhi = wilson(hk, n)
    dm, dlo, dhi, _ = paired_diff_ci([r[4]-r[3] for r in rl])
    sig = "YES" if (dlo > 0 or dhi < 0) else "no"
    data[L] = dict(n=n, fp=fp, flo=flo, fhi=fhi, hp=hp, hlo=hlo, hhi=hhi, dm=dm, dlo=dlo, dhi=dhi, sig=sig)
    print(f"{L:>3} | {n:>4} | {fp*100:5.1f} [{flo*100:4.1f},{fhi*100:5.1f}] | "
          f"{hp*100:5.1f} [{hlo*100:4.1f},{hhi*100:5.1f}] | "
          f"{dm*100:+5.1f} [{dlo*100:+5.1f},{dhi*100:+5.1f}] | {sig}")

# --- plot with CI bands ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
fp = np.array([data[L]["fp"] for L in Ls])*100
flo = np.array([data[L]["flo"] for L in Ls])*100; fhi = np.array([data[L]["fhi"] for L in Ls])*100
hp = np.array([data[L]["hp"] for L in Ls])*100
hlo = np.array([data[L]["hlo"] for L in Ls])*100; hhi = np.array([data[L]["hhi"] for L in Ls])*100
ax.plot(Ls, fp, "o-", color="#d62728", lw=2.2, label="flat WM + MPC")
ax.fill_between(Ls, flo, fhi, color="#d62728", alpha=0.15)
ax.plot(Ls, hp, "s-", color="#1f77b4", lw=2.2, label="hierarchical (HWM-style)")
ax.fill_between(Ls, hlo, hhi, color="#1f77b4", alpha=0.15)
# mark significant separations
for L in Ls:
    if data[L]["sig"] == "YES":
        ax.scatter([L], [data[L]["hp"]*100], s=160, facecolors="none", edgecolors="#1f77b4", lw=2, zorder=5)
n_per = int(np.median([data[L]["n"] for L in Ls]))
ax.set_xlabel("optimal path length  L  (BFS cells)")
ax.set_ylabel("success rate (%)")
_sub = ("model error = 0 -> flat collapse is purely planner-horizon"
        if "EXACT" in LABEL else "flat fails from model error + horizon; this is the rung-3 baseline")
ax.set_title(f"Rigorous rung-2: {LABEL}, {n_per} mazes×pairs/L, 95% CI\n"
             f"{_sub}; circled = hier-flat CI excludes 0", fontsize=9.5)
ax.set_ylim(-2, 102); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
png = OUT / PNG_NAME
fig.savefig(png, dpi=145); plt.close(fig)
print(f"\nwrote {png}")
