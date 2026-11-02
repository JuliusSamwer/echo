"""Cheap correctness checks for each piece, runnable without training a full model.

Run:  python -m pytest tests/ -q     (or just: python tests/test_smoke.py)
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zeus.config import Rung0Config
from zeus.envs.point_maze import PointMaze
from zeus.data.collect import collect_dataset, dataset_bounds
from zeus.models.encoder import IdentityEncoder
from zeus.models.world_model import build_world_model
from zeus.planning.mpc import PlannerParams, RecedingHorizonController, plan, coarse_only_plan


def test_maze_walls_block_and_path_len():
    rng = np.random.default_rng(0)
    env = PointMaze.generate((5, 5), path_len=3, rng=rng)
    assert env.shortest_path_len() == 3
    # walls keep the point inside the box
    env.reset()
    for _ in range(200):
        s, _, done, _ = env.step(rng.uniform(-env.a_max, env.a_max, size=2))
        assert 0.0 <= s[0] <= env.W and 0.0 <= s[1] <= env.H
        if done:
            env.reset()


def test_collision_against_a_known_wall():
    # 2x1 maze with NO passage between the two cells: a point shoved right must not cross x=1.
    open_right = np.zeros((2, 1), dtype=bool)
    open_up = np.zeros((2, 1), dtype=bool)
    env = PointMaze(open_right, open_up, (0, 0), (1, 0), a_max=0.4)
    env.reset()
    for _ in range(50):
        s, *_ = env.step(np.array([0.4, 0.0]))
    assert s[0] < 1.0, "point leaked through a wall"


def test_encoder_roundtrip():
    enc = IdentityEncoder.from_bounds([0, 0], [5, 5])
    x = torch.tensor([[1.3, 4.2], [0.0, 5.0]])
    assert torch.allclose(enc.decode(enc.encode(x)), x, atol=1e-5)


def test_world_model_rollout_shapes():
    enc = IdentityEncoder.from_bounds([0, 0], [5, 5])
    wm = build_world_model(enc, act_dim=2, hidden_dim=32, n_layers=2)
    z0 = torch.zeros(2)
    actions = torch.zeros(7, 4, 2)
    traj = wm.rollout(z0, actions)
    assert traj.shape == (7, 4, 2)


def test_planner_signature_and_bounds():
    cfg = Rung0Config.smoke(); cfg.n_samples = 64; cfg.horizon = 6
    rng = np.random.default_rng(1)
    env = PointMaze.generate(cfg.grid, cfg.path_len, rng)
    eps = collect_dataset(env, 10, 30, cfg.explore_momentum, cfg.explore_noise, 0.0, rng)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low, high)
    wm = build_world_model(enc, 2, 32, 2)
    params = PlannerParams.from_cfg(cfg)

    # standalone functional signature: plan(model, start, goal, H, N) -> actions
    actions = plan(wm, env.reset(), env.goal, cfg.horizon, cfg.n_samples, params=params, rng=rng)
    assert actions.shape == (cfg.horizon, 2)
    assert np.all(np.abs(actions) <= cfg.a_max + 1e-6)

    # coarse_only_plan hook exists and is callable (no-op delegate at rung 0)
    coarse = coarse_only_plan(wm, env.reset(), env.goal, cfg.horizon, cfg.n_samples, params=params, rng=rng)
    assert coarse.shape == (cfg.horizon, 2)

    ctrl = RecedingHorizonController(wm, params, H=cfg.horizon, N=cfg.n_samples)
    ctrl.step(env.reset(), env.goal, rng)
    assert ctrl.stats.cost_per_plan == cfg.n_samples * cfg.horizon


def test_hierarchical_controller_and_coarse_only():
    from zeus.planning.hierarchical import HierarchicalController
    cfg = Rung0Config.smoke()
    rng = np.random.default_rng(2)
    env = PointMaze.generate(cfg.grid, cfg.path_len, rng)
    eps = collect_dataset(env, 10, 30, cfg.explore_momentum, cfg.explore_noise, 0.0, rng)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low, high)
    wm = build_world_model(enc, 2, 32, 2, act_scale=cfg.a_max)
    params = PlannerParams.from_cfg(cfg)

    # full hierarchical step -> bounded primitive actions, same interface as flat controller
    hc = HierarchicalController(wm, params, k=4, M=4, N_hi=32, N_lo=32, hi_iters=2, lo_iters=2)
    a = hc.step(env.reset(), env.goal, rng)
    assert a.shape[1] == 2 and np.all(np.abs(a) <= cfg.a_max + 1e-6)

    # coarse-only closure diagnostic: returns the raw macro plan (k constant actions)
    hco = HierarchicalController(wm, params, k=4, M=4, N_hi=32, N_lo=32,
                                 hi_iters=2, lo_iters=2, coarse_only=True)
    ac = hco.step(env.reset(), env.goal, rng)
    assert ac.shape == (4, 2) and np.all(np.abs(ac) <= cfg.a_max + 1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all smoke tests passed")
