"""Receding-horizon MPPI with a HARD step budget H. The step budget is what makes
doorways bite -- it is not optional.

Changes vs the spec (same MPPI math, same capability):
  * Vectorized over samples: `step_batch(states[N,2], actions[N,2]) -> [N,2]` rolls
    all N candidate trajectories forward together, instead of a Python loop that called
    a scalar `step_fn` N*T*H times per hop (that made the probe take days).
  * Generic over ANY batched step fn -> the SAME executor drives the true env (probe)
    and the frozen learned model (rung 3). Pass `env.step_batch` for the probe.
  * Seeded RNG (spec used global np.random) so the paired bootstrap in the probe is
    reproducible.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class MPPIConfig:
    plan_horizon: int = 14       # lookahead T   (certified: strong per-hop search so GT
    n_samples: int = 384         # stays reliable to L=4 without raising the H budget that
    n_iters: int = 3             # would otherwise collapse the GT-ADVERSARIAL gap)
    lam: float = 0.1             # temperature
    noise_std: float = 0.08      # action-space exploration
    max_step: float = 0.14
    w_running: float = 1.0
    w_terminal: float = 6.0
    cost_mode: str = "running_terminal"   # "running_terminal" (probe default) | "min"
    #   "min" = min distance-to-target over the horizon ("did the rollout EVER reach it?").
    #   Robust to wall detours (a terminal/running cost presses into walls); use it when the
    #   target may require moving AWAY first (flat-to-goal, long hops). The grid planner needed it.


class BoundedMPPI:
    def __init__(self, step_batch, cfg: MPPIConfig, rng=None):
        """step_batch(states[N,2], actions[N,2]) -> next_states[N,2]  (true env or model)."""
        self.step_batch = step_batch
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def _clip(self, a):
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        return a * np.minimum(1.0, self.cfg.max_step / (n + 1e-9))

    def _step_one(self, state, action):
        return self.step_batch(state[None], action[None])[0]

    def _rollout_costs(self, state, samples, subgoal):
        """samples[N,T,2] -> costs[N]. All N trajectories rolled out together."""
        cfg = self.cfg
        N, T, _ = samples.shape
        states = np.repeat(state[None], N, axis=0)   # [N,2]
        running = np.zeros(N)
        mind = np.full(N, np.inf)
        for t in range(T):
            states = self.step_batch(states, samples[:, t, :])
            d = np.linalg.norm(states - subgoal, axis=-1)
            running += d
            mind = np.minimum(mind, d)
        term = np.linalg.norm(states - subgoal, axis=-1)
        if cfg.cost_mode == "min":
            return mind + 0.1 * term            # reach-ever, small terminal tiebreak
        return cfg.w_running * running + cfg.w_terminal * term

    def _plan_action(self, state, subgoal, nominal):
        cfg = self.cfg
        T = cfg.plan_horizon
        for _ in range(cfg.n_iters):
            noise = self.rng.standard_normal((cfg.n_samples, T, 2)) * cfg.noise_std
            samples = self._clip(nominal[None] + noise)           # [N,T,2]
            costs = self._rollout_costs(state, samples, subgoal)  # [N]
            beta = costs.min()
            w = np.exp(-(costs - beta) / cfg.lam)
            w /= (w.sum() + 1e-9)
            nominal = self._clip((w[:, None, None] * samples).sum(axis=0))
        return nominal[0], nominal

    def reach(self, state, subgoal, H, subgoal_eps):
        """Reach subgoal from state within H primitive steps.
        Returns (reached, final_state, steps_used, trajectory)."""
        s = state.copy()
        traj = [s.copy()]
        nominal = np.zeros((self.cfg.plan_horizon, 2))
        for step in range(H):
            a, nominal = self._plan_action(s, subgoal, nominal)
            s = self._step_one(s, a)
            traj.append(s.copy())
            nominal = np.roll(nominal, -1, axis=0); nominal[-1] = 0.0   # warm start
            if np.linalg.norm(s - subgoal) < subgoal_eps:
                return True, s, step + 1, np.array(traj)
        return False, s, H, np.array(traj)

    def execute_chain(self, start, subgoals, goal, H, subgoal_eps, goal_eps):
        """Reach each subgoal in order (<=H steps each), then the goal.
        Returns (success, total_steps, full_trajectory, per_hop_reached)."""
        s = np.asarray(start, float).copy()
        full = [s.copy()]
        hops = []
        targets = list(subgoals) + [goal]
        for g in targets:
            is_goal = g is targets[-1]
            eps = goal_eps if is_goal else subgoal_eps
            reached, s, used, traj = self.reach(s, np.asarray(g, float), H, eps)
            full.extend(traj[1:])
            hops.append(reached)
            if not reached:
                return False, len(full) - 1, np.array(full), hops
        return True, len(full) - 1, np.array(full), hops
