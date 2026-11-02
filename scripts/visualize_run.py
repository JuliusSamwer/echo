"""Render a successful rung-0 episode: static trajectory PNG + GIF.

Uses the winning config (min-over-horizon energy, H20 N512). Loads the cached
model if present, else trains one. Rolls episodes until it finds a success and
captures the executed trajectory (plus the planner's imagined rollout at t=0).
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
from zeus.train import train_world_model
from zeus.planning.mpc import PlannerParams, RecedingHorizonController, plan
from zeus.eval.viz import plot_trajectory, animate_trajectory

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
CACHE = Path("/Users/juliius/Nobel30/Zeus/models/"
             "model_cache.pt")

cfg = Rung0Config.full()
cfg.cost_mode = "min"; cfg.horizon = 20; cfg.n_samples = 512
torch.manual_seed(cfg.seed)
rng = np.random.default_rng(cfg.seed)

env = PointMaze.generate(cfg.grid, cfg.path_len, rng, a_max=cfg.a_max,
                         goal_eps=cfg.goal_eps, max_steps=cfg.max_steps)

if CACHE.exists():
    model = torch.load(CACHE, weights_only=False)["model"]; print("loaded cached model")
else:
    eps = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len, cfg.explore_momentum,
                          cfg.explore_noise, cfg.goal_directed_frac, rng)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, cfg.device, act_scale=cfg.a_max)
    train_world_model(model, eps, cfg, verbose=False)
    torch.save({"model": model}, CACHE); print("trained model")

params = PlannerParams.from_cfg(cfg)
ctrl = RecedingHorizonController(model, params, H=cfg.horizon, N=cfg.n_samples)

# roll held-out (start,goal) pairs until one succeeds
for attempt in range(60):
    sc, gc = env.sample_start_goal(cfg.path_len, rng)
    env.start_cell = sc; env.set_goal(gc)
    ctrl.reset()
    s = env.reset(); goal = env.goal
    traj = [s.copy()]
    imagined = None
    done = False
    while env._t < cfg.max_steps:
        seq = ctrl.step(s, goal, rng)
        if imagined is None:  # capture the model's imagined rollout from the first plan
            z0 = model.encode(torch.as_tensor(s, dtype=torch.float32))
            a = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
            imagined = model.decode(model.rollout(z0, a)[0]).detach().cpu().numpy()
            imagined = np.vstack([s, imagined])
        for k in range(min(cfg.exec_k, len(seq))):
            s, _, done, info = env.step(seq[k]); traj.append(s.copy())
            if done: break
        if done: break
    if done and info["success"]:
        traj = np.asarray(traj)
        L = env.shortest_path_len()
        title = f"Zeus rung-0 success  |  L={L}  start={sc} goal={gc}  |  {len(traj)-1} steps"
        png = OUT / "successful_run.png"
        gif = OUT / "successful_run.gif"
        plot_trajectory(env, traj, goal, str(png), title=title, imagined=imagined)
        animate_trajectory(env, traj, goal, str(gif), title=title)
        print(f"SUCCESS after {attempt+1} sampled pair(s): start={sc} goal={gc}, {len(traj)-1} steps")
        print(f"  wrote {png}")
        print(f"  wrote {gif}")
        break
else:
    print("no success found in 60 attempts (unexpected at ~90%)")
