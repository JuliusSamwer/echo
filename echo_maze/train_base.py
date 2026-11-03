"""Entry: train the base forward model on the certified env, validate, then freeze.

  python -m echo_maze.train_base            # full (30k steps, 200k transitions)
  python -m echo_maze.train_base --smoke    # fast sanity

Artifact -> models/base_frozen.pt (inside the repo, persistent).
Do NOT run this until run_probe.py PASSes -- an uncertified env makes the base useless.
"""
import sys
from pathlib import Path
import torch

from .config import MazeConfig
from .maze import BottleneckMaze
from .base_model import train_base, validate_base, freeze
from echo.device import resolve_device

OUT = Path(__file__).resolve().parents[1] / "models"
OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    dev = str(resolve_device("auto"))
    env = BottleneckMaze(MazeConfig())
    steps = 2000 if smoke else 60_000
    n_data = 20_000 if smoke else 400_000
    model = train_base(env, steps=steps, n_data=n_data, device=dev)
    metrics = validate_base(env, model, device=dev)
    model = freeze(model)
    out = OUT / "base_frozen.pt"
    torch.save({"state_dict": model.state_dict(), "metrics": metrics,
                "cfg": env.cfg.__dict__}, out)
    print(f"[base] saved frozen base -> {out}  (validated ok={metrics['ok']})")
