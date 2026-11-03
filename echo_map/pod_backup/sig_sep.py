import csv, os
from collections import defaultdict
import numpy as np
def caa_out(f):
    by=defaultdict(list)
    for r in csv.DictReader(open(f)): by[int(r["env_idx"])].append(float(r["closest_approach"]))
    return {e:(1 if min(v)>=0.3 else 0) for e,v in by.items()}
def load_seg(f):
    by=defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f)):
        e=int(r["env_idx"])
        for k in ("r_plan","r_exec","r_plan_norm"): by[e][k].append(float(r[k]))
    return by
def auc(fv,sv):
    if not fv or not sv: return float("nan")
    c=sum(1 for a in fv for b in sv if a>b); t=sum(1 for a in fv for b in sv if a==b)
    return (c+0.5*t)/(len(fv)*len(sv))
base="/workspace/HWM_PLDM/runs/"
out=caa_out(base+"diag_caa.csv"); seg=load_seg(base+"diag_segments.csv")
aggs={"mean":np.mean,"max":np.max,"final":lambda v:v[-1],"sum":np.sum}
nf=sum(out.values())
print("episodes=%d  fail=%d  succ=%d"%(len(out),nf,len(out)-nf))
print("%-12s %-6s %s"%("signal","agg","AUC(fail>succ) [0.5=no sep, ->1 separates]"))
best=(0,"","")
for sig in ("r_plan","r_exec","r_plan_norm"):
    for an,af in aggs.items():
        vals={e:af(seg[e][sig]) for e in seg if seg[e][sig]}
        fv=[vals[e] for e in vals if out.get(e)==1]; sv=[vals[e] for e in vals if out.get(e)==0]
        a=auc(fv,sv)
        print("%-12s %-6s %.3f"%(sig,an,a))
        if a==a and abs(a-0.5)>abs(best[0]-0.5): best=(a,sig,an)
print("BEST separator: %s/%s AUC=%.3f"%(best[1],best[2],best[0]))
