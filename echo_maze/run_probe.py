"""Entry: certify the env. Sweep H to locate the placement-critical band, emit overlays.

  python -m echo_maze.run_probe            # full certification (H sweep, 100 instances)
  python -m echo_maze.run_probe --smoke    # fast sanity (few instances, H=8)

Reading the sweep: you want an H where GT >> RANDOM ~= ADVERSARIAL with a clean CI gap.
As H grows the gap collapses (the low-level routes doorways unaided) -- that collapse
point is the top of the placement-critical band; pick H comfortably below it. If NO H
gives a gap, geometry is too permissive: narrow doorway_width, raise room_size relative
to max_step, or add rooms -- then re-run. Do not proceed to Phase 4 until PASS at all L.
"""
import sys
from pathlib import Path
import numpy as np

from .config import MazeConfig
from .maze import BottleneckMaze
from .executor import MPPIConfig, BoundedMPPI
from .probe import run_probe, summarize, plot_overlay, _adversarial_subgoals

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


def _emit_overlays(maze_cfg, mppi_cfg, H=8, L=3, seed=0):
    """One qualitative figure: GT threading doorways vs ADVERSARIAL stalling."""
    cfg = MazeConfig(**{**maze_cfg.__dict__, "seed": seed})
    env = BottleneckMaze(cfg)
    rng = np.random.default_rng(seed + 777)
    env.reset(L=L, rng=rng)
    start, goal = env.state.copy(), env.goal.copy()
    rooms, gt_doors = env.path_doorways(start, goal)
    m = MPPIConfig(**{**mppi_cfg.__dict__, "max_step": cfg.max_step})
    ex = BoundedMPPI(env.step_batch, m, rng=np.random.default_rng(seed + 888))
    _, _, gt_traj, _ = ex.execute_chain(start, gt_doors, goal, H, cfg.subgoal_eps, cfg.goal_eps)
    adv_sg = _adversarial_subgoals(env, rooms)
    _, _, adv_traj, _ = ex.execute_chain(start, adv_sg, goal, H, cfg.subgoal_eps, cfg.goal_eps)
    path = OUT / "probe_overlay.png"
    plot_overlay(env, start, goal,
                 {"GT (doorways)": (gt_doors, gt_traj),
                  "ADVERSARIAL (room centers)": (adv_sg, adv_traj)},
                 path, title=f"Placement probe  (H={H}, L={L})")
    print(f"[probe] wrote overlay -> {path}")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    maze_cfg = MazeConfig()                       # defaults sit in the target regime
    mppi_cfg = MPPIConfig(max_step=maze_cfg.max_step)

    # certified band is around H=10; sweep straddles it to show the gap open then collapse
    H_values = (10,) if smoke else (8, 10, 11, 12, 16)
    L_values = (2, 3) if smoke else (2, 3, 4)
    n_inst = 20 if smoke else 100

    _emit_overlays(maze_cfg, mppi_cfg, H=10, L=3)
    band_pass = {}
    for H in H_values:
        print(f"\n===== H = {H} =====")
        res = run_probe(maze_cfg, mppi_cfg, L_values=L_values, n_instances=n_inst, H=H)
        band_pass[H] = summarize(res)
    good = [H for H, ok in band_pass.items() if ok]
    print(f"\n[probe] PASS at H in {good}" if good else
          "\n[probe] NO H PASSED — geometry too permissive; tighten doorway_width / room_size / add rooms")
