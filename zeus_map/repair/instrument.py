"""Glue so the upstream `mpc.py` diff is 3 lines. Activated only when env var ZEUS_SEGMENT_CSV is set;
otherwise every call is a no-op (baseline runs untouched, trajectories bit-exact).

Env:
  ZEUS_SEGMENT_CSV  path to dump the per-segment CSV (setting it enables instrumentation)
  ZEUS_METHOD       method label for the records (default "hwm")

A "segment" = one replan interval. At replan step i, the freshly-read obs_t + proprio is the ACHIEVED
state of the previous segment (=> r_exec for prev) AND the start state of the new segment (=> r_plan for
new, computed from that replan's l2/l1 results). One backbone encode per segment; deterministic (eval
mode) so it consumes no RNG and does not alter the rollout.
"""
from __future__ import annotations
import atexit
import os
import torch

from . import residuals as _resid

_STARTED = False


def maybe_start():
    global _STARTED
    if _STARTED:
        return
    _STARTED = True
    csv_path = os.environ.get("ZEUS_SEGMENT_CSV")
    if not csv_path:
        return                                   # instrumentation disabled -> pure baseline
    logger = _resid.SegmentLogger(method=os.environ.get("ZEUS_METHOD", "hwm"))
    _resid.set_logger(logger)
    atexit.register(lambda: logger.to_csv(csv_path) and None)
    print(f"[zeus-instrument] segment logging ON -> {csv_path}")


def _level_to_L(prefix) -> int:
    """map planner prefix like 'd4rl_medium' -> difficulty index (easy=1,medium=2,hard=3)."""
    s = str(prefix or "").lower()
    return {"easy": 1, "medium": 2, "hard": 3}.get(s.split("_")[-1], 0)


@torch.no_grad()
def _encode_obs(ev, obs_t, proprio_pos, proprio_vel, locations):
    """obs -> obs-component latent, exactly as the planner encodes (backbone), detached."""
    model = ev.model.level1  # HJEPA: encoder lives on level1
    if proprio_pos is not None and proprio_vel is not None:
        proprio = torch.cat([proprio_pos, proprio_vel], dim=-1).cuda()
    elif proprio_pos is not None:
        proprio = proprio_pos.cuda()
    elif proprio_vel is not None:
        proprio = proprio_vel.cuda()
    else:
        proprio = None
    bb = model.backbone(obs_t.cuda(), proprio=proprio,
                        locations=locations.cuda() if locations is not None else None)
    return bb.obs_component.detach()


def _xy(t):
    """(.., 2) tensor -> list of [x, y] floats, tolerant of a squeezable middle dim."""
    import numpy as np
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    a = a.reshape(a.shape[0], -1)[:, :2]
    return [[float(v[0]), float(v[1])] for v in a]


_PLAN_ROWS = []


def _flush_plan():
    p = os.environ.get("ZEUS_PLAN_CSV")
    if not p or not _PLAN_ROWS:
        return
    import csv
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_PLAN_ROWS[0].keys()))
        w.writeheader(); w.writerows(_PLAN_ROWS)
    print(f"[zeus-plan] wrote {len(_PLAN_ROWS)} planned-waypoint rows -> {p}", flush=True)


atexit.register(_flush_plan)


def log_plan_geometry(ev, l2_result, l1_result, seg_idx):
    """ADDITIVE, env-gated (ZEUS_PLAN_CSV): where the high level is actually POINTING, in maze (x,y).

    `l2_result.locations` is (H+1, bs, 2) — the planner's own probe of every predicted high-level
    waypoint into maze coordinates (mppi_planner already computes it via the location prober).
    Index 1 is THE SUBGOAL handed to the low level; 2.. are the rest of the intended chain.
    `l1_result.locations[-1]` is where the low level predicts it will actually land.
    Read-only on detached tensors — no extra model forward, no RNG, no effect on the rollout.
    """
    if not os.environ.get("ZEUS_PLAN_CSV"):
        return
    try:
        if l2_result is None or getattr(l2_result, "locations", None) is None:
            return
        L = _level_to_L(getattr(ev, "prefix", None))
        base = getattr(ev, "_zeus_env_base", 0)
        cond = os.environ.get("ZEUS_REPAIR_CONDITION", "hwm") or "hwm"
        chain = l2_result.locations                      # (H+1, bs, ..., 2)
        H = chain.shape[0]
        land = (_xy(l1_result.locations[-1])
                if l1_result is not None and getattr(l1_result, "locations", None) is not None else None)
        for h in range(H):
            pts = _xy(chain[h])
            for j, (x, y) in enumerate(pts):
                _PLAN_ROWS.append(dict(
                    condition=cond, L=L, env_idx=int(base + j), segment=int(seg_idx),
                    waypoint=h,                          # 0 = current state, 1 = SUBGOAL, 2.. = rest of chain
                    x=x, y=y,
                    l1_pred_x=(land[j][0] if land else float("nan")),
                    l1_pred_y=(land[j][1] if land else float("nan")),
                ))
    except Exception as e:
        print(f"[zeus-plan] WARNING plan-geometry log failed, run continues: {e}", flush=True)


def on_replan(ev, obs_t, proprio_pos, proprio_vel, locations, l2_result, l1_result, i):
    """Called at every bilevel replan (i % replan_every == 0) after the level split. No-op if disabled."""
    log_plan_geometry(ev, l2_result, l1_result, i // max(1, getattr(ev.config, "replan_every", 1)))
    if _resid.get_logger() is None or l2_result is None:
        return
    z_here = _encode_obs(ev, obs_t, proprio_pos, proprio_vel, locations)   # achieved(prev) == start(new)
    seg_idx = i // ev.config.replan_every
    prev = getattr(ev, "_zeus_seg", None)
    if prev is not None:
        _resid.record_segment(
            method=os.environ.get("ZEUS_METHOD", "hwm"),
            L=_level_to_L(getattr(ev, "prefix", None)),
            maze_id=getattr(ev, "_zeus_maze_ids", 0),
            env_idx=getattr(ev, "_zeus_env_base", 0),
            episode=0, segment=prev["seg"],
            l2_result=prev["l2"], l1_result=prev["l1"],
            z_achieved_obs=z_here, z_current_obs=prev["z0"],
            segment_succeeded=(_resid.r_exec_from_achieved(z_here, prev["l2"])
                               < getattr(ev, "_zeus_subgoal_eps", 1e9)).int().tolist(),
        )
    ev._zeus_seg = dict(l2=l2_result, l1=l1_result, z0=z_here, seg=seg_idx)


def on_episode_end(ev, obs_t, proprio_pos, proprio_vel, locations, success_history):
    """Record the last pending segment (r_exec from final obs) and back-fill episode success."""
    if _resid.get_logger() is None:
        return
    prev = getattr(ev, "_zeus_seg", None)
    if prev is not None:
        z_final = _encode_obs(ev, obs_t, proprio_pos, proprio_vel, locations)
        _resid.record_segment(
            method=os.environ.get("ZEUS_METHOD", "hwm"),
            L=_level_to_L(getattr(ev, "prefix", None)),
            maze_id=getattr(ev, "_zeus_maze_ids", 0),
            env_idx=getattr(ev, "_zeus_env_base", 0),
            episode=0, segment=prev["seg"],
            l2_result=prev["l2"], l1_result=prev["l1"],
            z_achieved_obs=z_final, z_current_obs=prev["z0"],
            segment_succeeded=(_resid.r_exec_from_achieved(z_final, prev["l2"])
                               < getattr(ev, "_zeus_subgoal_eps", 1e9)).int().tolist(),
        )
    ev._zeus_seg = None
    # episode success: env reached goal at any step
    if success_history:
        import numpy as np
        ep_succ = (np.stack([s for s in success_history]).max(axis=0) > 0).astype(int)
        base = getattr(ev, "_zeus_env_base", 0)
        for j, s in enumerate(ep_succ):
            _resid.get_logger().set_episode_success(base + j, 0, int(s))


def on_episode_caa(ev, location_history, targets, success_history):
    import os, csv, traceback
    import numpy as np
    p = os.environ.get("ZEUS_CAA_CSV")
    if not p or not location_history:
        return
    try:
        locs = np.stack([np.asarray(x) for x in location_history])
        locs = locs.reshape(locs.shape[0], locs.shape[1], -1)[..., :2]
        tg = targets.detach().cpu().numpy() if hasattr(targets, "detach") else np.asarray(targets)
        tg = tg.reshape(tg.shape[0], -1)[:, :2]
        d = np.linalg.norm(locs - tg[None], axis=-1)
        caa = d.min(axis=0); final = d[-1]
        L = _level_to_L(getattr(ev, "prefix", None))
        base = getattr(ev, "_zeus_env_base", 0)
        cond = os.environ.get("ZEUS_REPAIR_CONDITION", "hwm") or "hwm"
        rows = [dict(condition=cond, L=L, env_idx=int(base + j), closest_approach=float(caa[j]), final_dist=float(final[j]), start_dist=float(d[0, j])) for j in range(caa.shape[0])]
        hdr = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if hdr: w.writeheader()
            w.writerows(rows)
        print(f"[zeus-caa] logged {len(rows)} envs (L={L}, base={base}) -> {p}", flush=True)
        # --- additive: per-step dist-to-true-goal trajectory (top-down goal-progress signal) ---
        pp = os.environ.get("ZEUS_PROGRESS_CSV")
        if pp:
            try:
                T = d.shape[0]
                prows = [dict(condition=cond, L=L, env_idx=int(base + j), step=int(t),
                              dist_to_goal=float(d[t, j]))
                         for j in range(d.shape[1]) for t in range(T)]
                phdr = not os.path.exists(pp)
                with open(pp, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
                    if phdr: w.writeheader()
                    w.writerows(prows)
                print(f"[zeus-progress] logged {d.shape[1]} envs x {T} steps -> {pp}", flush=True)
            except Exception as e2:
                print(f"[zeus-progress] WARNING progress log failed, run continues: {e2}", flush=True)
        # --- additive: the FULL per-step (x,y) rollout, not just its norm ---------------------
        # `locs` already holds every agent position; the scalar dist above is a projection of it.
        xp = os.environ.get("ZEUS_XY_CSV")
        if xp:
            try:
                T = locs.shape[0]
                xrows = [dict(condition=cond, L=L, env_idx=int(base + j), step=int(t),
                              x=float(locs[t, j, 0]), y=float(locs[t, j, 1]),
                              goal_x=float(tg[j, 0]), goal_y=float(tg[j, 1]))
                         for j in range(locs.shape[1]) for t in range(T)]
                xhdr = not os.path.exists(xp)
                with open(xp, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(xrows[0].keys()))
                    if xhdr: w.writeheader()
                    w.writerows(xrows)
                print(f"[zeus-xy] logged {locs.shape[1]} envs x {T} steps -> {xp}", flush=True)
            except Exception as e3:
                print(f"[zeus-xy] WARNING xy log failed, run continues: {e3}", flush=True)
        # --- additive: the maze layout per env, so the trajectory can be drawn against walls ---
        mp = os.environ.get("ZEUS_MAZE_NPZ")
        if mp:
            try:
                mazes = {}
                for j, env in enumerate(getattr(ev, "envs", []) or []):
                    m = getattr(env, "maze_map", None)
                    if m is None:
                        m = getattr(getattr(env, "unwrapped", None), "maze_map", None)
                    if m is not None:
                        mazes[f"env{int(base + j)}"] = np.asarray(m)
                if mazes:
                    prev = {}
                    if os.path.exists(mp):
                        with np.load(mp) as z:
                            prev = {k: z[k] for k in z.files}
                    prev.update(mazes)
                    np.savez_compressed(mp, **prev)
                    print(f"[zeus-maze] saved {len(mazes)} maze layouts -> {mp}", flush=True)
            except Exception as e4:
                print(f"[zeus-maze] WARNING maze dump failed, run continues: {e4}", flush=True)
    except Exception as e:
        print(f"[zeus-caa] WARNING caa failed, run continues: {e}", flush=True)
        traceback.print_exc()
