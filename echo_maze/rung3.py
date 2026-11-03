"""Rung-3 interface (spec only; build the bodies after the probe PASSes).

Left deliberately open: the surrogate-vs-REINFORCE choice and continuous-s vs
discrete-levels are live decisions; committing bodies now would front-run them. This
fixes the *interfaces* so the certified env (echo_maze) and frozen base plug in
cleanly, and names the shapes and losses.

Losses (rung-3a):
  Placement  = sum of predicted reachability residual along the subgoal chain
               (drives boundaries toward doorways).
  Minimality = rate cost on sum(gate)  (else the gate emits boundaries everywhere).
  Closure    = coarse-only rollout reaches goal: || CoarseModel chain end - goal ||.
  Freeze base AND low-level while B trains (collapses come from too many moving parts).

Measurements (vs HWM-hier baseline, rung-2 CI rigor): success vs L; junction lift =
boundary density at doorways / chance, scored against maze.path_doorways ground truth;
coarse-only closure success; boundary count vs L.
Decision gate: match baseline within CIs AND junction lift > 1.
"""
import torch.nn as nn


class BoundaryGate(nn.Module):
    """B_phi: proposes subgoals along the route + a Gumbel-Softmax gate for
    'is this a boundary'. Differentiable placement from step one.
    Input:  context [B, D]  (current state, goal, optional coarse-plan features)
    Output: subgoals [B, K, 2], gate_logits [B, K]   (K = max boundaries)
    """
    def forward(self, context, tau):
        raise NotImplementedError("rung-3a: build after probe PASS")


class CoarseModel(nn.Module):
    """P^2: predicts coarse-step transition subgoal_i -> subgoal_{i+1} for closure.
    Input [B, 2] (+goal), output [B, 2]."""
    def forward(self, g, goal):
        raise NotImplementedError("rung-3a: build after probe PASS")


def reachability_residual(frozen_base, executor_or_predictor, s, g):
    """CONTINUOUS residual = || low_level_end(s->g) - g ||, the differentiable placement
    signal. Rung-3a: a learned R_psi(s,g) trained to match the bounded executor over the
    frozen base (differentiable w.r.t. g, hence B_phi). Upgrade path (staged, not now):
    differentiable energy-descent pass for goal-relevance; REINFORCE-as-correction fallback."""
    raise NotImplementedError("rung-3a: build after probe PASS")
