"""Load the saved BC policy and compare PLAIN vs DYNAMICS-SNAPPED eval head-to-head.

Plain BC on this placement-critical maze is too imprecise (B's small regression error misses the
doorway). Snapping B's proposed subgoal to the farthest reachable point under the frozen P should
recover the reachable-chain performance. This isolates: did B learn the right COARSE direction?

  python -m zeus_maze.eval_snapped
"""
from pathlib import Path
import numpy as np
import torch

from .config import MazeConfig
from .maze import BottleneckMaze
from .base_model import load_frozen_base
from .rung3_reinforce import SubgoalPolicy, evaluate, GeoGrid
from .rung3_demos import evaluate_snapped, BASE, POLICY

if __name__ == "__main__":
    model, cfg_d, base_step = load_frozen_base(BASE)
    cfg = MazeConfig(**{**cfg_d, "geodesic_res": 24})
    env = BottleneckMaze(cfg); true_step = env.step_batch; mx = env.cfg.max_step
    Ls = (2, 3, 4); n = 40
    geogrid = GeoGrid(env, res=40)

    d = torch.load(POLICY, weights_only=False)
    policy = SubgoalPolicy(max_hop=d.get("max_hop", 1.6))
    policy.load_state_dict(d["state_dict"])
    print(f"[eval_snapped] loaded {POLICY} (max_hop={d.get('max_hop')})", flush=True)

    print(f"[eval_snapped] PLAIN eval (n={n}/L) ...", flush=True)
    plain = evaluate(policy, base_step, true_step, env, cfg, mx, geogrid, Ls, n=n, seed=999)
    print(f"[eval_snapped] SNAPPED eval (n={n}/L) ...", flush=True)
    snap = evaluate_snapped(policy, base_step, true_step, env, cfg, mx, geogrid, Ls, n=n, seed=999)

    print("\n[eval_snapped] refs: flat~0.15 HWM~0.25 REINFORCE~0.00 oracle(GT@H10)=0.92/0.80/0.82")
    print("   L |  PLAIN | SNAPPED")
    for L in Ls:
        print(f"   {L} |  {plain[L]:.2f}  |  {snap[L]:.2f}", flush=True)
