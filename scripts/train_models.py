"""Train N per-maze learned world models on the GPU, cached + resumable.

Single process by design — the earlier 5-process CPU run thrashed (torch threads
oversubscribed). One process on the resolved GPU (cuda>mps>cpu) avoids contention.
Models are cached to <model-dir>/maze_{m}.pt and skipped if present, so this is
safe to stop/resume. Same maze gen seeds (1000+m) + training recipe as the rigorous
baseline, so these models ARE the rung-3 substrate.

  ECHO_DEVICE=mps python scripts/train_models.py --mazes 100 --model-dir DIR
  # CUDA box, shard across GPUs:
  ECHO_DEVICE=cuda:0 python scripts/train_models.py --start 0  --end 50  --model-dir DIR &
  ECHO_DEVICE=cuda:1 python scripts/train_models.py --start 50 --end 100 --model-dir DIR &
"""
import sys, argparse, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo.config import Rung0Config
from echo.device import resolve_device, device_report
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset, dataset_bounds, coverage_report
from echo.models.encoder import IdentityEncoder
from echo.models.world_model import build_world_model
from echo.train import train_world_model

ap = argparse.ArgumentParser()
ap.add_argument("--mazes", type=int, default=100)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--model-dir", type=str, required=True)
ap.add_argument("--device", type=str, default="auto")   # or ECHO_DEVICE env
ap.add_argument("--threads", type=int, default=2)        # torch threads (CPU)
ap.add_argument("--episodes", type=int, default=400)
ap.add_argument("--epochs", type=int, default=35)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--rollout-len", type=int, default=6)    # training unroll length (total FLOP driver)
args = ap.parse_args()

torch.set_num_threads(args.threads)
dev = resolve_device(args.device)
print(f"[train] {device_report()} | using '{dev}'", flush=True)

GRID = (9, 9)
cfg = Rung0Config.full(); cfg.grid = GRID; cfg.device = dev
cfg.n_train_episodes = args.episodes; cfg.episode_len = 80; cfg.goal_directed_frac = 0.3
cfg.epochs = args.epochs; cfg.batch_size = args.batch; cfg.rollout_len = args.rollout_len

mdir = Path(args.model_dir); mdir.mkdir(parents=True, exist_ok=True)
end = args.end if args.end is not None else args.mazes
todo = [m for m in range(args.start, end) if not (mdir / f"maze_{m}.pt").exists()]
print(f"[train] {end - args.start} mazes in range, {len(todo)} to train (rest cached)", flush=True)

times = []
for i, m in enumerate(todo):
    t0 = time.time()
    gen_rng = np.random.default_rng(1000 + m)
    env = PointMaze.generate(GRID, path_len=12, rng=gen_rng,
                             a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=200)
    eps = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len, cfg.explore_momentum,
                          cfg.explore_noise, cfg.goal_directed_frac, np.random.default_rng(2000 + m))
    cov = coverage_report(eps, env)
    low, high = dataset_bounds(eps)
    enc = IdentityEncoder.from_bounds(low - 0.5, high + 0.5)
    torch.manual_seed(m)
    model = build_world_model(enc, 2, cfg.hidden_dim, cfg.n_layers, dev, act_scale=cfg.a_max)
    train_world_model(model, eps, cfg, verbose=False)
    # 1-step accuracy probe
    s = np.concatenate([e.states[:-1] for e in eps]); a = np.concatenate([e.actions for e in eps])
    s2 = np.concatenate([e.states[1:] for e in eps])
    with torch.no_grad():
        z = model.encode(torch.tensor(s, dtype=torch.float32, device=dev))
        pred = model.decode(model.step(z, torch.tensor(a, dtype=torch.float32, device=dev))).cpu().numpy()
    rmse1 = float(np.sqrt(np.mean((pred - s2) ** 2)))
    torch.save({"model": model.to("cpu"), "rmse1": rmse1, "coverage": cov}, mdir / f"maze_{m}.pt")
    dt = time.time() - t0; times.append(dt)
    eta = np.mean(times) * (len(todo) - i - 1)
    print(f"[train] maze {m:3d} ({i+1}/{len(todo)}) cov={cov:.0%} rmse@1={rmse1:.3f} "
          f"{dt:.1f}s | ETA {eta/60:.1f} min", flush=True)

print(f"[train] done: {len(todo)} models, mean {np.mean(times) if times else 0:.1f}s/model", flush=True)
