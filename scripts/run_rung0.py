"""End-to-end rung-0 run: collect -> train P -> plan with MPC -> eval the gate.

Usage:
    python scripts/run_rung0.py --smoke      # ~seconds, proves the loop closes
    python scripts/run_rung0.py --full       # the real >=95% gate run
    python scripts/run_rung0.py --full --planner mppi --path-len 4

Rung 0 PASSES when success >= 95% on easy held-out (start, goal) pairs.
On failure, read the diagnostics in this order (printed below):
    data coverage  ->  model accuracy (rollout_mse)  ->  planner budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo.config import Rung0Config
from echo.envs.point_maze import PointMaze
from echo.data.collect import collect_dataset, dataset_bounds, coverage_report
from echo.models.encoder import IdentityEncoder
from echo.models.world_model import build_world_model
from echo.train import train_world_model
from echo.planning.mpc import PlannerParams, RecedingHorizonController
from echo.eval.harness import evaluate, probe_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--planner", choices=["cem", "mppi"], default=None)
    ap.add_argument("--path-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Rung0Config.smoke() if args.smoke or not args.full else Rung0Config.full()
    if args.planner: cfg.planner = args.planner
    if args.path_len is not None: cfg.path_len = args.path_len
    if args.seed is not None: cfg.seed = args.seed

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    print("=" * 72)
    print(f"ECHO rung 0  |  planner={cfg.planner}  L={cfg.path_len}  grid={cfg.grid}  seed={cfg.seed}")
    print("=" * 72)

    # --- environment: one fixed easy maze (state obs => dynamics are maze-specific) ---
    env = PointMaze.generate(cfg.grid, cfg.path_len, rng,
                             a_max=cfg.a_max, goal_eps=cfg.goal_eps, max_steps=cfg.max_steps)
    print(f"[env] maze {cfg.grid}, start={env.start_cell} goal={env.goal_cell}, "
          f"true path len={env.shortest_path_len()}")

    # --- data ---
    print("[data] collecting offline random-walk trajectories ...")
    episodes = collect_dataset(env, cfg.n_train_episodes, cfg.episode_len,
                               cfg.explore_momentum, cfg.explore_noise,
                               cfg.goal_directed_frac, rng)
    cov = coverage_report(episodes, env)
    n_trans = sum(len(e.actions) for e in episodes)
    print(f"[data] {len(episodes)} episodes, {n_trans} transitions, cell coverage={cov:.0%}")
    if cov < 0.95:
        print("[data] WARNING: coverage < 95% — consider goal_directed_frac>0 (diagnostic #1).")

    # --- model: E (normalizing identity) -> P (flat MLP) ---
    low, high = dataset_bounds(episodes)
    pad = 0.5
    encoder = IdentityEncoder.from_bounds(low - pad, high + pad)
    model = build_world_model(encoder, act_dim=2, hidden_dim=cfg.hidden_dim,
                              n_layers=cfg.n_layers, device=cfg.device, act_scale=cfg.a_max)
    n_params = sum(p.numel() for p in model.P.parameters())
    print(f"[model] forward model P: {n_params} params, latent_dim={model.latent_dim}")
    print("[train] fitting P (1-step + multi-step rollout loss) ...")
    history = train_world_model(model, episodes, cfg)
    print(f"[train] final {cfg.rollout_len}-step rollout loss = {history['rollout'][-1]:.2e}")

    # --- clean model diagnostics (decoupled from planner) ---
    probe = probe_model(env, model, cfg, rng, horizon=cfg.horizon)
    print(f"[probe] model open-loop RMSE @1/{cfg.horizon//2}/{cfg.horizon} steps = "
          f"{probe['per_step_rmse'][0]:.3f} / {probe['per_step_rmse'][cfg.horizon//2-1]:.3f} / "
          f"{probe['per_step_rmse'][-1]:.3f} cells  |  wall-tunnel rate = {probe['tunnel_rate']:.1%}")

    # --- planner + eval ---
    params = PlannerParams.from_cfg(cfg)
    controller = RecedingHorizonController(model, params, H=cfg.horizon, N=cfg.n_samples)
    print(f"[eval] {cfg.n_eval_episodes} held-out (start,goal) pairs at L={cfg.path_len} ...")
    report = evaluate(env, model, controller, cfg, rng)

    print("-" * 72)
    print(f"RESULT  {report.summary()}")
    passed = report.success_rate >= cfg.pass_threshold
    if passed:
        print(f"GATE: PASS  ({report.success_rate:.1%} >= {cfg.pass_threshold:.0%})  -> rung 1 unlocked.")
    else:
        print(f"GATE: FAIL  ({report.success_rate:.1%} < {cfg.pass_threshold:.0%})")
        # ordered triage
        if cov < 0.95:
            print("  triage: data coverage gap first (raise goal_directed_frac / n_episodes).")
        elif report.mean_rollout_mse > 1e-2:
            print(f"  triage: MODEL accuracy — rollout_mse={report.mean_rollout_mse:.2e} is high "
                  "(more data / capacity / epochs).")
        else:
            print(f"  triage: PLANNER budget — rollout_mse={report.mean_rollout_mse:.2e} is low, "
                  "so raise H / n_samples / cem_iters.")
    print("=" * 72)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
