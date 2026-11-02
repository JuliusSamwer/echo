import sys, re
top, hardn, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
p="pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml"; s=open(p).read()
i=s.index("  h_d4rl_planning:"); j=s.index("    easy:", i)
tb=re.sub(r"n_envs: \d+","n_envs: %d"%top, s[i:j], count=1); s=s[:i]+tb+s[j:]
k=s.index("    hard:"); hb=s[k:]
hb=re.sub(r"n_steps: \d+","n_steps: %d"%steps, hb, count=1)
hb=re.sub(r"n_envs: \d+","n_envs: %d"%hardn, hb, count=1)
open(p,"w").write(s[:k]+hb)
