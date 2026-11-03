"""Build the ECHO project-update figures from the raw logs/CSVs in this repo.

Every number plotted here is recomputed from files on disk (paths below), except the
HWM-paper reference numbers (Table values quoted from the HWM paper) and the M1 reproduction
numbers (echo_map/results/M1_RESULTS.txt + STATUS doc), which are typed in with their source.

    python scripts/make_echo_figures.py      # writes figures/echo/*.png
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = f"{ROOT}/echo_map/pod_backup/runs"
P1 = f"{ROOT}/echo_p1/runs"
OUT = f"{ROOT}/figures/echo"
os.makedirs(OUT, exist_ok=True)

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHTGRAY = "#8a8985", "#d6d5d0"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": LIGHTGRAY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"figures/echo/{name}")


# ----------------------------------------------------------------- helpers on the pod CSVs
def fail_labels(path, thr=0.3):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r["env_idx"])].append(float(r["closest_approach"]))
    return {e: int(min(v) >= thr) for e, v in by.items()}


def auc(vals, lab):
    fv = [vals[e] for e in vals if lab.get(e) == 1]
    sv = [vals[e] for e in vals if lab.get(e) == 0]
    c = sum(a > b for a in fv for b in sv) + 0.5 * sum(a == b for a in fv for b in sv)
    return c / (len(fv) * len(sv))


def fired(path):
    return sum(int(r["fired"]) for r in csv.DictReader(open(path)))


# ================================================================= 1. HWM reproduction
def fig_reproduction():
    levels = ["easy", "medium", "hard"]
    flat_ours, flat_paper = [0.975, 0.70, 0.50], [1.00, 0.63, 0.44]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=[3, 1.3]))
    x, w = np.arange(3), 0.36
    ax = axes[0]
    b1 = ax.bar(x - w / 2 - 0.01, flat_ours, w, color=BLUE, label="ours (reproduced)")
    b2 = ax.bar(x + w / 2 + 0.01, flat_paper, w, color=LIGHTGRAY, label="HWM paper")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():g}",
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, levels); ax.set_ylim(0, 1.12); ax.set_ylabel("success rate")
    ax.set_title("Flat PLDM (level 1 only)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax = axes[1]
    b1 = ax.bar([-w / 2 - 0.01], [1.00], w, color=BLUE)
    b2 = ax.bar([w / 2 + 0.01], [0.95], w, color=LIGHTGRAY)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}",
                ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks([0], ["medium"]); ax.set_ylim(0, 1.12); ax.set_xlim(-0.6, 0.6)
    ax.set_title("HWM (2-level)")
    ax.text(0, -0.2, "ours n=6 envs", ha="center", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    fig.suptitle("M1 · HWM_PLDM reproduction on D4RL diverse maze (frozen public checkpoints)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "m1_reproduction.png")


# ================================================================= 2. H-JEPA / LeWM / HWM training (echo_p1)
def read_log(name):
    rows = list(csv.DictReader(open(f"{P1}/{name}/seed0/log.csv")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def fig_p1():
    runs = [("flat", "flat LeWM (1 level)", GRAY), ("hjepa", "H-JEPA (2 levels, SIGReg on both)", BLUE),
            ("hwm", "HWM-style (E2 = identity, no SIGReg L2)", ORANGE)]
    logs = {k: read_log(k) for k, _, _ in runs}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    panels = [("val_l1", "Level-1 val prediction loss", True), ("rank1", "Level-1 effective rank (of 64)", False),
              ("rank2", "Level-2 effective rank (of 64)", False)]
    for ax, (key, title, logy) in zip(axes, panels):
        for k, label, c in runs:
            L = logs[k]
            if key not in L:
                continue
            ax.plot(L["step"], smooth(L[key]) if logy else L[key], color=c, lw=2, label=label)
        ax.set_title(title); ax.set_xlabel("training step (of 20k planned)")
        if logy:
            ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    for k, label, c in runs:
        L = logs[k]
        for ax, key in ((axes[1], "rank1"), (axes[2], "rank2")):
            if key in L:
                ax.annotate(f"{L[key][-1]:.0f}", (L["step"][-1], L[key][-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    fig.suptitle("echo_p1 · H-JEPA / LeWM / HWM reimplementation on FourRoom (seed 0, runs stopped at 6–8k steps)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "p1_training.png")


# ================================================================= 3. signal separation AUCs
def fig_signals():
    lab = fail_labels(f"{RUNS}/diag_caa.csv")
    seg = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{RUNS}/diag_segments.csv")):
        for k in ("r_plan", "r_exec", "r_plan_norm"):
            seg[int(r["env_idx"])][k].append(float(r[k]))
    prog = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{RUNS}/diag_progress.csv")):
        prog[int(r["env_idx"])][int(r["step"])].append(float(r["dist_to_goal"]))
    traj = {e: np.array([np.mean(s[t]) for t in sorted(s)]) for e, s in prog.items()}

    rows = []
    for sig, nice in (("r_plan", "r_plan (planner residual)"), ("r_exec", "r_exec (executed residual)"),
                      ("r_plan_norm", "r_plan_norm (normalised)")):
        best = max((auc({e: f(seg[e][sig]) for e in seg}, lab), an) for an, f in
                   (("mean", np.mean), ("max", np.max), ("final", lambda v: v[-1]), ("sum", np.sum)))
        rows.append((f"{nice} · best agg = {best[1]}", best[0], "reach"))
    stall = {e: float(np.mean(d[1:] >= d[:-1] - 1e-6)) for e, d in traj.items()}
    rows.append(("stall_frac (oracle x,y goal progress)", auc(stall, lab), "oracle"))
    nf = sum(lab.values())

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [BLUE if r[2] == "oracle" else GRAY for r in rows][::-1]
    ax.barh(names, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0.51, -0.45, "chance", fontsize=8, color=MUTED, ha="left")
    ax.set_xlim(0, 1.05); ax.set_xlabel("AUC (failed episode > succeeded episode)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What predicts HWM failure?  (hard, n={len(lab)} episodes: {nf} fail / {len(lab)-nf} succeed)")
    fig.text(0.01, -0.06, "Reachability barely separates failures; goal-progress stall does — but only measured with "
             "oracle (x,y). The latent-space\nequivalent d(z, z_goal) separates at ≈ chance (STATUS doc §5.4) → "
             "motivates the learned temporal-distance head R(z,g).", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "m3_signal_auc.png")
    return rows


# ================================================================= 4. repair ledger (n=18, paired)
def fig_repairs():
    conds = [
        ("HWM baseline", "n18_hwm", None, "v1/v2"),
        ("repair_always (reach)", "n18_always", "n18_always", "v1"),
        ("repair_trigger (reach, τ)", "n18_trig", "n18_trig", "v1"),
        ("repair_always, strong β", "2p_alwaysB", "2p_alwaysB", "v2"),
        ("trigger on goal-progress", "2p_trigprog", "2p_trigprog", "v2"),
        ("trigger on both", "2p_trigboth", "2p_trigboth", "v2"),
    ]
    base = fail_labels(f"{RUNS}/n18_hwm_caa.csv")
    data = []
    for name, caa, ev, ver in conds:
        lab = fail_labels(f"{RUNS}/{caa}_caa.csv")
        nfail = sum(lab.values())
        rescued = sum(base[e] == 1 and lab[e] == 0 for e in base)
        broke = sum(base[e] == 0 and lab[e] == 1 for e in base)
        nrep = fired(f"{RUNS}/{ev}_events.csv") if ev else 0
        data.append((name, nfail, rescued, broke, nrep, ver))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    names = [d[0] for d in data][::-1]
    succ = [(18 - d[1]) / 18 for d in data][::-1]
    cols = [LIGHTGRAY if d[0] == "HWM baseline" else (BLUE if d[5] == "v1" else AQUA) for d in data][::-1]
    ax.barh(names, succ, color=cols, height=0.55)
    for i, d in enumerate(data[::-1]):
        tag = "" if d[0] == "HWM baseline" else f"   +{d[2]} rescued / −{d[3]} broke"
        ax.text(succ[i] + 0.01, i, f"{18-d[1]}/18{tag}", va="center", fontsize=8.5, color=INK2)
    ax.axvline((18 - data[0][1]) / 18, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0, 1.0); ax.set_xlabel("episode success rate (same 18 hard episodes, paired)")
    ax.grid(axis="y", visible=False)
    ax.set_title("M3 · test-time subgoal repair on frozen HWM")
    ax = axes[1]
    rep = [(d[0], d[4], d[5]) for d in data if d[4] > 0][::-1]
    ax.barh([r[0] for r in rep], [r[1] for r in rep], color=[BLUE if r[2] == "v1" else AQUA for r in rep], height=0.55)
    for i, r in enumerate(rep):
        ax.text(r[1] + 8, i, str(r[1]), va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 720); ax.set_xlabel("repairs fired (of 612 segments)")
    ax.grid(axis="y", visible=False); ax.set_title("repair compute")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="v1 reachability-triggered (Jul 13)"),
                        Patch(color=AQUA, label="v2 two-pressure (Jul 28)")],
               frameon=False, fontsize=8.5, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.06))
    fig.text(0.01, -0.05, "Hard level, n_steps=150 (budget-cut regime). None significant: best McNemar p = 0.625. "
             "Trigger matches/beats always-repair at 2.6× fewer repairs.", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "m3_repair_ledger.png")
    return data


# ================================================================= 5. schematics
def box(ax, xy, w, h, text, fc, ec=INK2, fs=9.5, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.35)


def arrow(ax, p, q, c=INK2, lw=1.6, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=c, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4.6); ax.axis("off"); ax.grid(False)
    ax.text(0.2, 4.4, "ECHO: closed-loop hierarchy", fontsize=13, fontweight="bold")
    box(ax, (0.2, 2.8), 2.0, 0.9, "goal", "#f1f0ec", fs=11)
    box(ax, (3.2, 2.8), 2.4, 0.9, "High level\nproposes subgoal", "#e3eefb", ec=BLUE, fs=11)
    box(ax, (6.6, 2.8), 2.2, 0.9, "Repair\nadjusts subgoal", "#e2f5ee", ec=AQUA, fs=11)
    box(ax, (0.2, 0.5), 2.0, 0.9, "current state", "#f1f0ec", fs=11)
    box(ax, (3.2, 0.5), 2.4, 0.9, "Low level\nreaches subgoal", "#e3eefb", ec=BLUE, fs=11)
    box(ax, (6.6, 0.5), 2.2, 0.9, "Environment", "#f1f0ec", fs=11)
    arrow(ax, (2.2, 3.25), (3.2, 3.25)); arrow(ax, (2.2, 0.95), (3.2, 0.95))
    arrow(ax, (5.6, 0.95), (6.6, 0.95))
    arrow(ax, (3.9, 2.8), (3.9, 1.4), lw=2)
    ax.text(3.75, 2.1, "subgoal", fontsize=10, ha="right", va="center", color=INK2)
    arrow(ax, (5.0, 1.4), (7.0, 2.8), c=ORANGE, lw=2.6, rad=-0.15)
    ax.text(6.25, 1.9, "residual\n(the echo)", fontsize=10.5, color=ORANGE, ha="left", va="center", fontweight="bold")
    arrow(ax, (6.6, 3.25), (5.6, 3.25), c=AQUA, lw=2)
    save(fig, "schematic_architecture.png")


def fig_timeline():
    stages = [
        ("Jun 23", "Rung 0", "flat world model\n+ MPC, ≥95% easy", "PASS", BLUE),
        ("Jul 6", "Rung 1", "flat breaks with\nhorizon (100→5%)", "PASS", BLUE),
        ("Jul 7", "Rung 2", "hierarchy fixes it\n(exact model)", "PASS*", BLUE),
        ("Jul 8", "Rung 3", "emergent subgoals\n(RL / BC / energy)", "NULL", GRAY),
        ("Jul 9", "Closed loop", "repair on toy maze\n0.38→0.87 at L=4", "PASS", BLUE),
        ("Jul 10", "HWM repro", "PLDM + HWM\non D4RL maze", "PASS", BLUE),
        ("Jul 10–13", "M2 / M3", "residual ρ=0.60\nrepair 7→5 fails", "WEAK", ORANGE),
        ("Jul 28", "M3.5", "two-pressure\nrepair", "NULL", GRAY),
        ("Oct 7", "H-JEPA", "LeWM, H-JEPA, HWM\nreimplemented", "IN PROG.", ORANGE),
        ("next", "v3 gate", "learned R(z,g)\non frozen latents", "TODO", LIGHTGRAY),
    ]
    fig, ax = plt.subplots(figsize=(16, 3.8))
    n = len(stages)
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-2.2, 1.5); ax.axis("off"); ax.grid(False)
    ax.plot([-0.3, n - 0.7], [0, 0], color=LIGHTGRAY, lw=3, zorder=0)
    for i, (d, name, desc, verdict, c) in enumerate(stages):
        ax.scatter([i], [0], s=170, color=c, zorder=3, edgecolor=SURF, linewidth=2)
        ax.text(i, 0.45, d, ha="center", fontsize=8.5, color=MUTED)
        ax.text(i, 0.85, name, ha="center", fontsize=10, fontweight="bold")
        ax.text(i, -0.45, desc, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.3)
        ax.text(i, -1.55, verdict, ha="center", va="top", fontsize=8.5, fontweight="bold",
                color=INK if c != LIGHTGRAY else MUTED,
                bbox=dict(boxstyle="round,pad=0.25", fc=c if c not in (LIGHTGRAY,) else "#f1f0ec", ec="none", alpha=0.25))
    ax.axvspan(-0.5, 3.5, ymin=0.02, ymax=0.98, color="#f6f5f2", zorder=-1)
    ax.axvspan(3.5, 4.5, ymin=0.02, ymax=0.98, color="#eef4fc", zorder=-1)
    ax.axvspan(4.5, n - 0.5, ymin=0.02, ymax=0.98, color="#f6f5f2", zorder=-1)
    ax.text(1.5, -2.15, "echo_maze · own toy mazes (state obs)", ha="center", fontsize=8.5, color=MUTED)
    ax.text(4.0, -2.15, "bridge", ha="center", fontsize=8.5, color=MUTED)
    ax.text(7.0, -2.15, "echo_map / echo_p1 · published world models (pixels)", ha="center", fontsize=8.5, color=MUTED)
    ax.text(-0.5, 1.4, "Project timeline, 2026", fontsize=12, fontweight="bold")
    save(fig, "timeline.png")


if __name__ == "__main__":
    fig_reproduction()
    fig_p1()
    sig = fig_signals()
    for r in sig:
        print(f"  AUC {r[1]:.3f}  {r[0]}")
    rep = fig_repairs()
    for d in rep:
        print("  ", d)
    fig_architecture()
    fig_timeline()
