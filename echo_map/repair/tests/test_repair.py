"""Unit tests (spec engineering ground rules). Run: python -m pytest repair/tests -q
Covers: (a) frozen params across a repair, (b) gradient reaches g and only g, (c) the residual metric
matches the planner's RunningCost mean-sq, (d) instrumentation is a no-op when no logger is set.
Tests (c-instrument-bit-exact) and compute-counter live in eval_harness once wired to the repo."""
import torch
import torch.nn as nn

from repair.repair_op import repair_subgoal, RepairConfig, frozen_snapshot, assert_frozen, _meansq
from repair.residuals import latent_distance, SegmentLogger, set_logger, record_segment


class TinyHigh(nn.Module):
    """Stand-in frozen high-level predictor: z' = z + W z (linear macro-step)."""
    def __init__(self, d=8):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)

    def forward(self, z):
        return z + self.W(z)


def test_frozen_params_and_only_g_grad():
    torch.manual_seed(0)
    d = 8; bs = 4
    high = TinyHigh(d)
    for p in high.parameters():
        p.requires_grad_(False)          # frozen model
    g0 = torch.randn(bs, d)
    z_T = torch.randn(bs, d)
    z_goal = torch.randn(bs, d)
    snap = frozen_snapshot(high)

    g_rep = repair_subgoal(g0, z_T, roll_high=lambda g: high(high(g)), z_goal=z_goal,
                           cfg=RepairConfig(n_inner=15, eta=5e-2), model=high)

    assert_frozen(high, snap)                       # (a) no model param changed
    assert g_rep.shape == g0.shape
    assert not torch.equal(g_rep, g0)               # g actually moved
    # (b) energy decreased
    E0 = (_meansq(z_T, g0) + _meansq(high(high(g0)), z_goal)).sum()
    E1 = (_meansq(z_T, g_rep) + _meansq(high(high(g_rep)), z_goal)).sum()
    assert E1 < E0


def test_gradient_reaches_only_g():
    d = 6; bs = 2
    high = TinyHigh(d)
    for p in high.parameters():
        p.requires_grad_(False)
    g = torch.randn(bs, d, requires_grad=True)
    z_T = torch.randn(bs, d); z_goal = torch.randn(bs, d)
    E = (_meansq(z_T, g) + _meansq(high(g), z_goal)).sum()
    E.backward()
    assert g.grad is not None and torch.count_nonzero(g.grad) > 0
    for p in high.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_metric_matches_runningcost():
    # RunningCost: diff = (state-target).pow(2); return diff.mean(-1)
    a = torch.randn(5, 32); b = torch.randn(5, 32)
    expected = (a - b).pow(2).mean(-1)
    assert torch.allclose(latent_distance(a, b), expected, atol=1e-6)


def test_instrumentation_is_noop_without_logger():
    set_logger(None)
    # record_segment must not raise and must not require real results when logger is None
    record_segment(method="hwm", L=2, maze_id=0, env_idx=0, episode=0, segment=0,
                   l2_result=None, l1_result=None, z_achieved_obs=None, z_current_obs=None,
                   segment_succeeded=1)
