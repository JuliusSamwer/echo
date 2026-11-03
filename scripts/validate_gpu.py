"""Validate + benchmark the batched evaluator against the sequential one."""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.models.true_model import TrueDynamicsModel
from echo.planning.mpc import PlannerParams, RecedingHorizonController
from echo.eval.gpu_eval import BatchedMaze, batched_flat_eval

cfg = Rung0Config.full(); cfg.grid = (9, 9); cfg.cost_mode = "min"
cfg.horizon, cfg.n_samples, cfg.cem_iters = 25, 256, 5
params = PlannerParams.from_cfg(cfg)
rng = np.random.default_rng(1000)
env = PointMaze.generate((9, 9), path_len=12, rng=rng, a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)

# (1) DYNAMICS bit-exactness: BatchedMaze.step vs env.step
print("=== (1) dynamics exactness ===")
bm_cpu = BatchedMaze(env, device="cpu")
maxerr = 0.0
for _ in range(15):
    sc = (int(rng.integers(9)), int(rng.integers(9)))
    s = env.reset(start_cell=sc); acts = rng.uniform(-0.4, 0.4, (40, 2))
    true = []
    for t in range(40):
        s, *_ = env.step(acts[t]); true.append(s.copy())
    p = torch.tensor(env._cell_center(sc), dtype=torch.float32)[None]
    got = []
    for t in range(40):
        p = bm_cpu.step(p, torch.tensor(acts[t], dtype=torch.float32)[None]); got.append(p[0].numpy())
    maxerr = max(maxerr, np.abs(np.array(got) - np.array(true)).max())
print(f"max abs diff vs env over 15x40 steps: {maxerr:.2e}  ({'OK' if maxerr < 1e-2 else 'FAIL'})")

# (2) SUCCESS-RATE parity: sequential (TrueDynamicsModel) vs batched (BatchedMaze), same pairs
print("\n=== (2) success-rate parity (perfect model) ===")
tm = TrueDynamicsModel(env)
for L in [6, 10]:
    pr = np.random.default_rng(7000 + L)
    pairs = []
    while len(pairs) < 30:
        scg = env.sample_start_goal(L, pr); env.start_cell, _ = scg[0], env.set_goal(scg[1])
        if env.shortest_path_len() == L: pairs.append(scg)
    max_steps = 25 + 8 * L
    # sequential
    seq = []
    for sc, gc in pairs:
        env.start_cell = sc; env.set_goal(gc); env.max_steps = max_steps
        ctrl = RecedingHorizonController(tm, params, H=25, N=256); ctrl.reset()
        s = env.reset(); info = {"success": False}
        while env._t < max_steps:
            a = ctrl.step(s, env.goal, np.random.default_rng(42)); s, _, dn, info = env.step(a[0])
            if dn: break
        seq.append(info["success"])
    # batched
    starts = [env._cell_center(sc) for sc, _ in pairs]
    goals = [env._cell_center(gc) for _, gc in pairs]
    suc = batched_flat_eval(bm_cpu, bm_cpu, starts, goals, params, 25, 256, max_steps, cfg.goal_eps, "cpu", seed=42)
    print(f"L={L:2d}: sequential {np.mean(seq):.2f}  |  batched {np.mean(suc):.2f}  "
          f"(n={len(pairs)}, diff {abs(np.mean(seq)-np.mean(suc)):.2f})")

# (3) BENCHMARK: sequential-CPU vs batched-CPU vs batched-MPS
print("\n=== (3) speed: 30 pairs at L=10, perfect model ===")
L = 10; max_steps = 25 + 8 * L
pr = np.random.default_rng(99); pairs = []
while len(pairs) < 30:
    scg = env.sample_start_goal(L, pr); env.start_cell = scg[0]; env.set_goal(scg[1])
    if env.shortest_path_len() == L: pairs.append(scg)
starts = [env._cell_center(sc) for sc, _ in pairs]; goals = [env._cell_center(gc) for _, gc in pairs]

t = time.time()
for sc, gc in pairs:
    env.start_cell = sc; env.set_goal(gc); env.max_steps = max_steps
    ctrl = RecedingHorizonController(tm, params, H=25, N=256); ctrl.reset()
    s = env.reset()
    while env._t < max_steps:
        a = ctrl.step(s, env.goal, np.random.default_rng(1)); s, _, dn, _ = env.step(a[0])
        if dn: break
seq_t = time.time() - t

t = time.time()
batched_flat_eval(bm_cpu, bm_cpu, starts, goals, params, 25, 256, max_steps, cfg.goal_eps, "cpu", seed=1)
cpu_b_t = time.time() - t

mps_b_t = float("nan")
if torch.backends.mps.is_available():
    bm_mps = BatchedMaze(env, device="mps")
    batched_flat_eval(bm_mps, bm_mps, starts, goals, params, 25, 256, max_steps, cfg.goal_eps, "mps", seed=1)  # warmup
    t = time.time()
    batched_flat_eval(bm_mps, bm_mps, starts, goals, params, 25, 256, max_steps, cfg.goal_eps, "mps", seed=1)
    mps_b_t = time.time() - t

print(f"sequential CPU (30 episodes loop): {seq_t:6.2f}s")
print(f"batched     CPU (1 call, B=30):    {cpu_b_t:6.2f}s   ({seq_t/cpu_b_t:.1f}x vs seq)")
print(f"batched     MPS (1 call, B=30):    {mps_b_t:6.2f}s   ({seq_t/mps_b_t:.1f}x vs seq)")
