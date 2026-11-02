import csv, os, math
from collections import defaultdict
from math import comb
def load(f):
    if not os.path.exists(f): return {}
    by=defaultdict(list)
    for r in csv.DictReader(open(f)): by[int(r["env_idx"])].append(float(r["closest_approach"]))
    return {e:min(v) for e,v in by.items()}
def wilson(k,n):
    if n==0: return (0.0,0.0,0.0)
    p=k/n; z=1.96; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (p,max(0.0,c-h),min(1.0,c+h))
def fires(f):
    if not os.path.exists(f): return 0
    return sum(1 for r in csv.DictReader(open(f)) if r.get("fired")=="1")
def fail(d,thr=0.3): return {e:(1 if d[e]>=thr else 0) for e in d}
def mcnemar(h,r):
    envs=set(h)&set(r)
    b=sum(1 for e in envs if h[e]==1 and r[e]==0)
    c=sum(1 for e in envs if h[e]==0 and r[e]==1)
    n=b+c
    pv=1.0 if n==0 else min(1.0, 2*sum(comb(n,i) for i in range(min(b,c)+1))/(2**n))
    return b,c,pv
base="/workspace/HWM_PLDM/runs/"
H=load(base+"n18_hwm_caa.csv"); A=load(base+"n18_always_caa.csv"); T=load(base+"n18_trig_caa.csv")
print("episodes: hwm=%d always=%d trig=%d"%(len(H),len(A),len(T)))
hf=fail(H); Af=fail(A); Tf=fail(T)
for nm,d in [("HWM",hf),("repair_always",Af),("repair_trigger",Tf)]:
    if not d: print("%-15s no data"%nm); continue
    k=sum(d.values()); n=len(d); p,lo,hi=wilson(k,n)
    print("%-15s fail %2d/%2d = %.2f  [%.2f,%.2f] 95%%CI"%(nm,k,n,p,lo,hi))
for nm,rf in [("always",Af),("trigger",Tf)]:
    if rf: b,c,pv=mcnemar(hf,rf); print("  vs HWM %-8s rescued=%d broke=%d McNemar p=%.4f"%(nm,b,c,pv))
af=fires(base+"n18_always_events.csv"); tf=fires(base+"n18_trig_events.csv")
print("repairs fired: always=%d trigger=%d  (%.1fx fewer)"%(af,tf,af/max(tf,1)))
