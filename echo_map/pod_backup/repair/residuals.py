"""Phase 2 — reachability residual instrumentation.

The residual is the low-level planner's OWN reachability estimate, which HWM computes and throws away.
We surface it. Two residuals per segment (a "segment" = one high-level replan interval = `replan_every`
primitive steps, over which the low level chases a single subgoal g):

  r_plan  (pre-execution, planner-derived):  d( ẑ_T(a*), g )
      ẑ_T(a*) = low-level predicted TERMINAL latent under its best action seq a* = l1_result.pred_obs[-1]
      g       = high-level's first predicted latent (the subgoal)             = l2_result.pred_obs[1]
  r_exec  (post-execution):                  d( z_achieved, g )
      z_achieved = encoder(obs after executing the segment in the env)

`d` is the SAME distance the low-level planner's cost uses (mppi_planner.RunningCost):
  d(a,b) = mean_over_features( (flatten(a) - flatten(b))**2 )   on the obs-component latent.
We deliberately do NOT invent a new metric (spec 2.1).

Design: instrumentation is a passive sink. `record_segment(...)` is a no-op unless a SegmentLogger is
installed via `set_logger`, and it only ever reads DETACHED tensors — zero effect on trajectories/RNG
(asserted bit-exact in tests). The upstream patch (see PATCHES.md) is 3 lines calling record_segment.
"""
from __future__ import annotations
import csv
import threading
from dataclasses import dataclass, asdict, fields
from typing import Optional
import torch

try:
    from pldm.models.utils import flatten_conv_output
except Exception:                                   # allow import off-repo (tests/tooling)
    def flatten_conv_output(x):
        return x.reshape(x.shape[0], -1) if x.dim() > 2 else x


# ------------------------------------------------------------ the distance (matches the planner cost)
def latent_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """d(a,b) exactly as mppi_planner.RunningCost: mean-squared error over flattened obs-component.
    a,b: (bs, ...) obs-component latents.  Returns (bs,) per-env distances."""
    a = flatten_conv_output(a.detach()).float()
    b = flatten_conv_output(b.detach()).float()
    return ((a - b) ** 2).mean(dim=-1)


def subgoal_from_l2(l2_result) -> torch.Tensor:
    """g = high-level's FIRST predicted latent (obs-component). pred_obs is (horizon+1, bs, ...)."""
    return l2_result.pred_obs[1]


def r_plan_from_results(l2_result, l1_result) -> torch.Tensor:
    """r_plan = d(ẑ_T(a*), g).  l1_result.pred_obs is (l2_step_skip+1, bs, ...); [-1] = predicted terminal."""
    g = subgoal_from_l2(l2_result)
    z_pred_terminal = l1_result.pred_obs[-1]
    return latent_distance(z_pred_terminal, g)


def r_exec_from_achieved(z_achieved_obs: torch.Tensor, l2_result) -> torch.Tensor:
    """r_exec = d(z_achieved, g).  z_achieved_obs = obs-component of encoder(achieved obs)."""
    return latent_distance(z_achieved_obs, subgoal_from_l2(l2_result))


# ------------------------------------------------------------ normalized residual (spec 2.4 caveat)
def r_plan_normalized(l2_result, l1_result, z_current_obs: torch.Tensor) -> torch.Tensor:
    """Dimensionless "fraction of the gap left unclosed": r_plan / d(z_t, g). Guards against latent-space
    region scale so a GLOBAL tau is meaningful. Used only if sanity-check-2 shows heavy overlap."""
    g = subgoal_from_l2(l2_result)
    gap = latent_distance(z_current_obs, g).clamp_min(1e-8)
    return r_plan_from_results(l2_result, l1_result) / gap


# ------------------------------------------------------------ per-segment record + logger
@dataclass
class SegmentRecord:
    method: str
    L: int                 # difficulty level (goal distance bucket)
    maze_id: int
    env_idx: int
    episode: int
    segment: int           # replan index within the episode
    r_plan: float
    r_exec: float
    r_plan_norm: float
    segment_succeeded: int  # subgoal reached within eps at segment end (r_exec small)
    episode_succeeded: int  # env info["success"] by episode end (filled in on flush)
    # Phase-3 fields (0 for pure Stage-1/2 baseline runs):
    r_plan_after_repair: float = -1.0
    n_repairs: int = 0
    n_escalations: int = 0
    fwd_passes_low: int = 0
    fwd_passes_high: int = 0


class SegmentLogger:
    """Accumulates SegmentRecords in memory, back-fills episode_succeeded, dumps one CSV per run."""
    def __init__(self, method: str):
        self.method = method
        self.records: list[SegmentRecord] = []
        self._lock = threading.Lock()

    def add(self, rec: SegmentRecord):
        with self._lock:
            self.records.append(rec)
            _n=len(self.records)
            if _n<=12 or _n%50==0:
                import sys; print(f"[echo-instrument] {_n} segments logged", file=sys.stderr, flush=True)

    def set_episode_success(self, env_idx: int, episode: int, success: int):
        for r in self.records:
            if r.env_idx == env_idx and r.episode == episode:
                r.episode_succeeded = int(success)

    def to_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(SegmentRecord)])
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))
        return path


# ------------------------------------------------------------ passive global hook (zero-effect when None)
_ACTIVE: Optional[SegmentLogger] = None


def set_logger(logger: Optional[SegmentLogger]):
    global _ACTIVE
    _ACTIVE = logger


def get_logger() -> Optional[SegmentLogger]:
    return _ACTIVE


def record_segment(*, method, L, maze_id, env_idx, episode, segment,
                   l2_result, l1_result, z_achieved_obs, z_current_obs,
                   segment_succeeded, **phase3):
    """Called from the patched _perform_mpc at each replan/segment end. No-op unless a logger is set.
    Reads detached tensors only. Per-env: emits one record per env in the batch."""
    if _ACTIVE is None:
        return
    rp = r_plan_from_results(l2_result, l1_result)
    re = r_exec_from_achieved(z_achieved_obs, l2_result)
    rn = r_plan_normalized(l2_result, l1_result, z_current_obs)
    bs = rp.shape[0]
    g = subgoal_from_l2(l2_result)
    gap = latent_distance(z_current_obs, g)
    for j in range(bs):
        _ACTIVE.add(SegmentRecord(
            method=method, L=int(L), maze_id=int(maze_id[j] if hasattr(maze_id, "__len__") else maze_id),
            env_idx=int(env_idx[j] if hasattr(env_idx, "__len__") else env_idx + j),
            episode=int(episode), segment=int(segment),
            r_plan=float(rp[j]), r_exec=float(re[j]), r_plan_norm=float(rn[j]),
            segment_succeeded=int(re[j] < gap[j]),  # reached = closer to subgoal than segment start
            episode_succeeded=-1,
            **{k: (int(v) if "n_" in k or "fwd" in k else float(v)) for k, v in phase3.items()},
        ))
