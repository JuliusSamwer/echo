"""Step-3 analysis: baseline (diag hwm) vs two-pressure repair conditions.
failure rate + Wilson 95% CI, paired McNemar vs hwm, repair-fire counts (compute).

Usage: python twop_analyze.py            (expects runs/ populated by run_diag + run_2p)
"""
import csv, os, math
from collections import defaultdict
from math import comb

BASE = "/workspace/HWM_PLDM/runs/"
HWM_CAA = BASE + "diag_caa.csv"
CONDS = [   # (label, caa_csv, events_csv or None)
    ("repair_alwaysB", BASE + "2p_alwaysB_caa.csv",  BASE + "2p_alwaysB_events.csv"),
    ("trig_progress",  BASE + "2p_trigprog_caa.csv", BASE + "2p_trigprog_events.csv"),
    ("trig_both",      BASE + "2p_trigboth_caa.csv", BASE + "2p_trigboth_events.csv"),
]


def load(f):
    if not os.path.exists(f):
        return {}
    by = defaultdict(list)
    for r in csv.DictReader(open(f)):
        by[int(r["env_idx"])].append(float(r["closest_approach"]))
    return {e: min(v) for e, v in by.items()}


def fail(d, thr=0.3):
    return {e: (1 if d[e] >= thr else 0) for e in d}


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def mcnemar(h, r):
    envs = set(h) & set(r)
    b = sum(1 for e in envs if h[e] == 1 and r[e] == 0)   # hwm fail, repair ok (rescued)
    c = sum(1 for e in envs if h[e] == 0 and r[e] == 1)   # hwm ok, repair fail (broke)
    n = b + c
    pv = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n))
    return b, c, pv


def fires(f):
    if not f or not os.path.exists(f):
        return None
    return sum(1 for r in csv.DictReader(open(f)) if r.get("fired") == "1")


H = load(HWM_CAA); hf = fail(H)
k = sum(hf.values()); n = len(hf); p, lo, hi = wilson(k, n)
print(f"{'HWM (baseline)':<18} fail {k:2d}/{n:2d} = {p:.2f}  [{lo:.2f},{hi:.2f}] 95%CI")
for label, caa, ev in CONDS:
    d = load(caa)
    if not d:
        print(f"{label:<18} (no data yet)")
        continue
    df = fail(d); k2 = sum(df.values()); n2 = len(df); p2, lo2, hi2 = wilson(k2, n2)
    b, c, pv = mcnemar(hf, df)
    fc = fires(ev)
    fcs = f"fires={fc}" if fc is not None else ""
    print(f"{label:<18} fail {k2:2d}/{n2:2d} = {p2:.2f}  [{lo2:.2f},{hi2:.2f}]  "
          f"rescued={b} broke={c} McNemar p={pv:.3f}  {fcs}")
