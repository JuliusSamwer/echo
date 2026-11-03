p="pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml"
s=open(p).read()
i=s.index("  h_d4rl_planning:")
j=s.index("    easy:", i)
block=s[i:j]
if "n_envs: 85" in block:
    open(p,"w").write(s[:i]+block.replace("n_envs: 85","n_envs: 18",1)+s[j:]); print("FIXED: top-level n_envs 85->18")
else:
    print("already fixed or 85 not found; top block n_envs lines:", [l for l in block.splitlines() if "n_envs" in l])
