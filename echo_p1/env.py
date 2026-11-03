"""Continuous 2D FourRoom with a distractor. State/step are numpy-batched; render is torch-batched."""
import numpy as np
import torch

SPEED = 0.025
WALL_HALF = 0.02      # wall thickness 0.04
DOOR_HALF = 0.06      # door gap width 0.12
MARGIN = 0.02
# doors 0,1 are on the vertical wall (x=.5) at y=.25/.75; doors 2,3 on the horizontal wall (y=.5) at x=.25/.75
DOOR_C = np.array([[.5, .25], [.5, .75], [.25, .5], [.75, .5]])
OU_THETA, OU_SIGMA, TELEPORT_MEAN = 0.15, 0.3, 35.0


def sample_doors(n, p_close, rng):
    """All doors open by default. With p_close>0, resample until >=3 doors open (rooms form a 4-cycle => connected)."""
    op = np.ones((n, 4), bool)
    if p_close > 0:
        op = rng.random((n, 4)) >= p_close
        bad = op.sum(1) < 3
        while bad.any():
            op[bad] = rng.random((bad.sum(), 4)) >= p_close
            bad = op.sum(1) < 3
    return op


def in_wall(xy, doors):
    """True where a point lies inside a (thick) wall outside any open door gap. xy [...,2], doors [...,4] bool."""
    x, y = xy[..., 0], xy[..., 1]
    gap_v = ((np.abs(y - .25) < DOOR_HALF) & doors[..., 0]) | ((np.abs(y - .75) < DOOR_HALF) & doors[..., 1])
    gap_h = ((np.abs(x - .25) < DOOR_HALF) & doors[..., 2]) | ((np.abs(x - .75) < DOOR_HALF) & doors[..., 3])
    return ((np.abs(x - .5) < WALL_HALF) & ~gap_v) | ((np.abs(y - .5) < WALL_HALF) & ~gap_h)


def room_of(xy):
    return (xy[..., 0] > .5).astype(int) + 2 * (xy[..., 1] > .5).astype(int)  # 0=LB 1=RB 2=LT 3=RT


def sample_free(n, rng, margin=0.05):
    """Uniform positions away from walls."""
    out = rng.uniform(MARGIN, 1 - MARGIN, (n, 2))
    bad = (np.abs(out[:, 0] - .5) < margin) | (np.abs(out[:, 1] - .5) < margin)
    while bad.any():
        out[bad] = rng.uniform(MARGIN, 1 - MARGIN, (bad.sum(), 2))
        bad = (np.abs(out[:, 0] - .5) < margin) | (np.abs(out[:, 1] - .5) < margin)
    return out


def reset(n, rng, doors=None, agent=None, p_close=0.0):
    if doors is None:
        doors = sample_doors(n, p_close, rng)
    return dict(
        agent=sample_free(n, rng) if agent is None else agent.copy(),
        dist=rng.uniform(MARGIN, 1 - MARGIN, (n, 2)),
        dvel=np.zeros((n, 2)),
        tp=rng.geometric(1 / TELEPORT_MEAN, n).astype(np.int64),
        doors=doors,
    )


def step(state, action, rng):
    """Batched step. Draws a fixed number of random numbers per call so trajectories are reproducible
    across planners given the same rng seed."""
    n = len(state["agent"])
    new = np.clip(state["agent"] + SPEED * np.clip(action, -1, 1), MARGIN, 1 - MARGIN)
    blocked = in_wall(new, state["doors"])
    agent = np.where(blocked[:, None], state["agent"], new)
    vel = state["dvel"] * (1 - OU_THETA) + OU_SIGMA * rng.standard_normal((n, 2))
    dist = np.clip(state["dist"] + SPEED * np.clip(vel, -1, 1), MARGIN, 1 - MARGIN)
    tele_pos = rng.uniform(MARGIN, 1 - MARGIN, (n, 2))
    tele_tp = rng.geometric(1 / TELEPORT_MEAN, n)
    tp = state["tp"] - 1
    hit = tp <= 0
    dist = np.where(hit[:, None], tele_pos, dist)
    tp = np.where(hit, tele_tp, tp)
    return dict(agent=agent, dist=dist, dvel=vel, tp=tp, doors=state["doors"])


# ---- room-graph routing for the expert ----
def _paths():
    edges = {0: (0, 1), 1: (2, 3), 2: (0, 2), 3: (1, 3)}  # door -> rooms it connects
    adj = {r: [] for r in range(4)}
    for dd, (a, b) in edges.items():
        adj[a].append((b, dd)); adj[b].append((a, dd))
    out = {}
    for a in range(4):
        for b in range(4):
            res = []

            def dfs(r, seen, doors):
                if r == b:
                    res.append(list(doors)); return
                for nr, dd in adj[r]:
                    if nr not in seen:
                        dfs(nr, seen | {nr}, doors + [dd])
            dfs(a, {a}, [])
            out[(a, b)] = res
    return out


PATHS = _paths()


def next_waypoint(pos, goal, doors):
    """Shortest open route over the room graph; waypoint = next door centre (or the goal in the same room)."""
    wp = goal.copy()
    best = np.full(len(pos), np.inf)
    ra, rb = room_of(pos), room_of(goal)
    for (a, b), paths in PATHS.items():
        if a == b:
            continue
        m = (ra == a) & (rb == b)
        if not m.any():
            continue
        for dl in paths:
            valid = m.copy()
            for dd in dl:
                valid &= doors[:, dd]
            pts = [pos] + [np.broadcast_to(DOOR_C[dd], pos.shape) for dd in dl] + [goal]
            length = sum(np.linalg.norm(p - q, axis=1) for p, q in zip(pts[:-1], pts[1:]))
            upd = valid & (length < best)
            wp[upd] = DOOR_C[dl[0]]
            best[upd] = length[upd]
    return wp


# ---- rendering ----
_GRID = {}


def _grid(res, device):
    k = (res, str(device))
    if k not in _GRID:
        c = (torch.arange(res, device=device, dtype=torch.float32) + .5) / res
        _GRID[k] = (c[None, :].expand(res, res), c[:, None].expand(res, res))  # X (cols), Y (rows)
    return _GRID[k]


def render(agent_xy, dist_xy, doors=None, res=64):
    """agent_xy, dist_xy: [B,2] tensors in [0,1]^2; doors: [B,4] bool or None (all open). -> uint8 [B,3,res,res]."""
    dev = agent_xy.device
    B = agent_xy.shape[0]
    X, Y = _grid(res, dev)
    X, Y = X[None], Y[None]
    if doors is None:
        doors = torch.ones(B, 4, dtype=torch.bool, device=dev)
    doors = doors.to(dev)
    d = lambda i: doors[:, i, None, None]
    gap_v = ((Y - .25).abs() < DOOR_HALF) & d(0) | ((Y - .75).abs() < DOOR_HALF) & d(1)
    gap_h = ((X - .25).abs() < DOOR_HALF) & d(2) | ((X - .75).abs() < DOOR_HALF) & d(3)
    wall = (((X - .5).abs() <= WALL_HALF) & ~gap_v) | (((Y - .5).abs() <= WALL_HALF) & ~gap_h)
    img = torch.ones(B, 3, res, res, device=dev)
    img = torch.where(wall[:, None], torch.full_like(img, 0.5), img)

    def disk(img, xy, color):
        dpx = torch.sqrt((X - xy[:, 0, None, None]) ** 2 + (Y - xy[:, 1, None, None]) ** 2) * res
        cov = (3.0 + 0.5 - dpx).clamp(0, 1)[:, None]  # anti-aliased radius-3px disk
        col = torch.tensor(color, device=dev).view(1, 3, 1, 1)
        return img * (1 - cov) + col * cov

    img = disk(img, dist_xy, (0.1, 0.2, 0.95))   # distractor: blue
    img = disk(img, agent_xy, (0.95, 0.1, 0.1))  # agent: red (drawn on top)
    return (img * 255).round().to(torch.uint8)
