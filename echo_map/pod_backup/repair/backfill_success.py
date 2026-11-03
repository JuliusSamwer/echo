"""Back-fill episode_succeeded in a segment CSV from the authoritative MPCReport.success.
Matches by (L, env_idx); report.success is indexed by global env_idx (0..n_envs-1).
Usage: python -m repair.backfill_success --csv X.csv --report R1 [--report R2] --out Y.csv
"""
import argparse, csv, torch
from collections import Counter

def level_from_name(p):
    s = p.lower()
    return 3 if "hard" in s else 2 if "medium" in s else 1 if "easy" in s else 0

def load_success(p):
    r = torch.load(p, map_location="cpu", weights_only=False)
    return [int(x) for x in r.success]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    succ_by_L = {}
    for rp in a.report:
        L = level_from_name(rp); succ_by_L[L] = load_success(rp)
        print(f"  report L={L}: success={succ_by_L[L]}")
    rows = list(csv.DictReader(open(a.csv)))
    n_set = n_miss = 0
    for row in rows:
        L = int(float(row["L"])); ei = int(float(row["env_idx"]))
        s = succ_by_L.get(L)
        if s is not None and 0 <= ei < len(s):
            row["episode_succeeded"] = s[ei]; n_set += 1
        else:
            n_miss += 1
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    dist = dict(Counter(int(r["episode_succeeded"]) for r in rows))
    print(f"set={n_set} miss={n_miss} episode_succeeded_dist={dist} -> {a.out}")

if __name__ == "__main__":
    main()
