"""Geometry, collision, ground truth, geodesic field, and the Gym-style env.

Changes vs the spec (capabilities identical; feasibility/correctness only):
  * Collision is vectorized: `_first_hit_t` intersects a batch of N rays against all
    M wall segments at once. `step_batch` moves N states in one call -> the MPPI
    executor rolls out 256 samples without a Python loop (the spec's per-sample loop
    made the probe take days). `step_fn` (single) is a thin wrapper for env.step.
  * `seg_blocked` is one clean boolean (spec called `collide` twice with a fragile
    `is not c1` identity check).
  * The geodesic field is LAZY — computed only when `geodesic()` is first called and
    cached per goal. The probe and base training never read it, so eager recompute on
    every reset (a 180x180 Dijkstra) was pure waste.
"""
import numpy as np
import heapq
from collections import deque
from typing import List, Tuple
from .config import MazeConfig


# ---------- geometry ----------

def _room_index(i, j, cfg):            # (col i, row j) -> flat index
    return j * cfg.n_rooms_x + i


def build_geometry(cfg: MazeConfig):
    """Return (walls, bottlenecks, room_graph, W, H).
    walls: list of (x1,y1,x2,y2) segments.
    bottlenecks: list of (center: np.array([x,y]), (roomA, roomB)).
    room_graph: adjacency dict room_idx -> list of (neighbor_idx, doorway_center).
    """
    rng = np.random.default_rng(cfg.seed)
    W = cfg.n_rooms_x * cfg.room_size
    H = cfg.n_rooms_y * cfg.room_size
    walls: List[Tuple[float, float, float, float]] = [
        (0, 0, W, 0), (0, H, W, H), (0, 0, 0, H), (W, 0, W, H)  # outer boundary
    ]
    bottlenecks = []
    dw = cfg.doorway_width

    def jitter(lo, hi, mid):
        if cfg.doorway_jitter <= 0:
            c = mid
        else:
            c = mid + rng.uniform(-1, 1) * cfg.doorway_jitter
        return float(np.clip(c, lo + dw / 2, hi - dw / 2))

    # internal vertical walls (between room i and i+1, each row j)
    for i in range(cfg.n_rooms_x - 1):
        x = (i + 1) * cfg.room_size
        for j in range(cfg.n_rooms_y):
            y0, y1 = j * cfg.room_size, (j + 1) * cfg.room_size
            gc = jitter(y0, y1, (y0 + y1) / 2)
            walls.append((x, y0, x, gc - dw / 2))
            walls.append((x, gc + dw / 2, x, y1))
            bottlenecks.append((np.array([x, gc]),
                                (_room_index(i, j, cfg), _room_index(i + 1, j, cfg))))

    # internal horizontal walls (between room j and j+1, each col i)
    for j in range(cfg.n_rooms_y - 1):
        y = (j + 1) * cfg.room_size
        for i in range(cfg.n_rooms_x):
            x0, x1 = i * cfg.room_size, (i + 1) * cfg.room_size
            gc = jitter(x0, x1, (x0 + x1) / 2)
            walls.append((x0, y, gc - dw / 2, y))
            walls.append((gc + dw / 2, y, x1, y))
            bottlenecks.append((np.array([gc, y]),
                                (_room_index(i, j, cfg), _room_index(i, j + 1, cfg))))

    room_graph = {r: [] for r in range(cfg.n_rooms_x * cfg.n_rooms_y)}
    for center, (a, b) in bottlenecks:
        room_graph[a].append((b, center))
        room_graph[b].append((a, center))
    return walls, bottlenecks, room_graph, W, H


# ---------- collision (vectorized) ----------

def walls_to_arrays(walls):
    """(list of (x1,y1,x2,y2)) -> (A[M,2], B[M,2]) segment endpoints."""
    w = np.asarray(walls, dtype=float)
    return w[:, :2].copy(), w[:, 2:].copy()


def _first_hit_t(P, P2, A, B):
    """For each ray P[n]->P2[n], the smallest t in [0,1] where it crosses any wall
    segment A[m]->B[m]; inf if it crosses none. Fully vectorized over N rays x M walls.
    P,P2: [N,2]  A,B: [M,2]  ->  best_t: [N]."""
    r = P2 - P                                  # [N,2]
    s = B - A                                    # [M,2]
    rxs = r[:, 0:1] * s[None, :, 1] - r[:, 1:2] * s[None, :, 0]   # [N,M]
    qp = A[None, :, :] - P[:, None, :]           # [N,M,2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / rxs   # [N,M]
        u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / rxs   # [N,M]
    valid = (np.abs(rxs) > 1e-12) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    t_masked = np.where(valid, t, np.inf)
    return t_masked.min(axis=1)                  # [N]


def _clip_step(a, max_step):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a * np.minimum(1.0, max_step / (n + 1e-9))


def collide_batch(P, P2, A, B):
    """Move each P[n]->P2[n], stopping just before the nearest wall crossing. [N,2]."""
    eps = 1e-3
    best_t = _first_hit_t(P, P2, A, B)           # [N]
    hit = np.isfinite(best_t)
    frac = np.where(hit, np.maximum(best_t - eps, 0.0), 1.0)[:, None]
    return P + frac * (P2 - P)


def seg_blocked(p, p2, A, B):
    """True iff the single segment p->p2 crosses any wall (for the geodesic field)."""
    return bool(np.isfinite(_first_hit_t(p[None], p2[None], A, B)[0]))


def collide(p, p_next, A, B):
    """Single-sample convenience wrapper around collide_batch."""
    return collide_batch(np.asarray(p, float)[None], np.asarray(p_next, float)[None], A, B)[0]


# ---------- geodesic distance field (true maze distance; lazy, optional) ----------

def geodesic_field(cfg, A, B, W, H, goal):
    """Grid Dijkstra from goal over free cells with 8-connectivity. An edge between
    adjacent cell centers is allowed iff the connecting segment crosses no wall.
    Returns (field[ny,nx], to_cell fn)."""
    nx = int(cfg.n_rooms_x * cfg.geodesic_res)
    ny = int(cfg.n_rooms_y * cfg.geodesic_res)
    cx, cy = W / nx, H / ny

    def center(ix, iy):
        return np.array([(ix + 0.5) * cx, (iy + 0.5) * cy])

    def to_cell(p):
        ix = int(np.clip(p[0] / cx, 0, nx - 1))
        iy = int(np.clip(p[1] / cy, 0, ny - 1))
        return ix, iy

    dist = np.full((ny, nx), np.inf)
    gx, gy = to_cell(goal)
    dist[gy, gx] = 0.0
    pq = [(0.0, gx, gy)]
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while pq:
        d, ix, iy = heapq.heappop(pq)
        if d > dist[iy, ix]:
            continue
        c0 = center(ix, iy)
        for ddx, ddy in neigh:
            jx, jy = ix + ddx, iy + ddy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            c1 = center(jx, jy)
            if seg_blocked(c0, c1, A, B):
                continue
            nd = d + np.linalg.norm(c1 - c0)
            if nd < dist[jy, jx]:
                dist[jy, jx] = nd
                heapq.heappush(pq, (nd, jx, jy))
    return dist, to_cell


# ---------- room-path routing (for L and ground-truth boundary sets) ----------

def room_path(room_graph, start_room, goal_room):
    """BFS shortest room path; (list_of_rooms, list_of_doorway_centers_on_path).
    len(doors) == L, len(rooms) == L + 1."""
    prev = {start_room: None}
    q = deque([start_room])
    while q:
        r = q.popleft()
        if r == goal_room:
            break
        for nb, center in room_graph[r]:
            if nb not in prev:
                prev[nb] = (r, center)
                q.append(nb)
    if goal_room not in prev:
        return None, None
    rooms, doors = [goal_room], []
    node = goal_room
    while prev[node] is not None:
        parent, center = prev[node]
        doors.append(center)
        rooms.append(parent)
        node = parent
    rooms.reverse(); doors.reverse()
    return rooms, doors


# ---------- environment ----------

class BottleneckMaze:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.walls, self.bottlenecks, self.room_graph, self.W, self.H = build_geometry(cfg)
        self.wA, self.wB = walls_to_arrays(self.walls)   # for vectorized collision
        self.state = None
        self.goal = None
        self._field = None
        self._to_cell = None
        self._field_goal = None      # goal the cached field was computed for

    # --- geometry helpers ---
    def which_room(self, p):
        i = int(np.clip(p[0] // self.cfg.room_size, 0, self.cfg.n_rooms_x - 1))
        j = int(np.clip(p[1] // self.cfg.room_size, 0, self.cfg.n_rooms_y - 1))
        return _room_index(i, j, self.cfg)

    def room_center(self, room):
        i = room % self.cfg.n_rooms_x
        j = room // self.cfg.n_rooms_x
        return np.array([(i + 0.5), (j + 0.5)]) * self.cfg.room_size

    def sample_free_point(self, rng, room=None):
        if room is None:
            room = int(rng.integers(self.cfg.n_rooms_x * self.cfg.n_rooms_y))
        i = room % self.cfg.n_rooms_x
        j = room // self.cfg.n_rooms_x
        pad = 0.12 * self.cfg.room_size
        x = rng.uniform(i * self.cfg.room_size + pad, (i + 1) * self.cfg.room_size - pad)
        y = rng.uniform(j * self.cfg.room_size + pad, (j + 1) * self.cfg.room_size - pad)
        return np.array([x, y])

    # --- ground truth for the probe / rung-3 measurement ---
    def path_doorways(self, start, goal):
        """(rooms, ordered doorway centers) on the shortest route: KNOWN-OPTIMAL boundaries."""
        sr, gr = self.which_room(start), self.which_room(goal)
        return room_path(self.room_graph, sr, gr)

    def difficulty_L(self, start, goal):
        _, doors = self.path_doorways(start, goal)
        return None if doors is None else len(doors)

    def geodesic(self, p):
        """True maze distance-to-goal at p. Field is computed lazily and cached per goal."""
        if self._field is None or self._field_goal is None or \
                not np.array_equal(self._field_goal, self.goal):
            self._field, self._to_cell = geodesic_field(
                self.cfg, self.wA, self.wB, self.W, self.H, self.goal)
            self._field_goal = self.goal.copy()
        ix, iy = self._to_cell(p)
        return float(self._field[iy, ix])

    # --- episode API ---
    def reset(self, start=None, goal=None, L=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng(self.cfg.seed)
        if L is not None:
            start, goal = self._sample_pair_for_L(L, rng)
        elif start is None or goal is None:
            start = self.sample_free_point(rng)
            goal = self.sample_free_point(rng)
        self.state = np.asarray(start, float).copy()
        self.goal = np.asarray(goal, float).copy()
        self._field = None           # invalidate lazy geodesic cache
        self._field_goal = None
        return self._obs()

    def _sample_pair_for_L(self, L, rng, tries=200):
        n = self.cfg.n_rooms_x * self.cfg.n_rooms_y
        for _ in range(tries):
            sr, gr = int(rng.integers(n)), int(rng.integers(n))
            if sr == gr:
                continue
            rooms, doors = room_path(self.room_graph, sr, gr)
            if doors is not None and len(doors) == L:
                return self.sample_free_point(rng, sr), self.sample_free_point(rng, gr)
        raise RuntimeError(f"could not sample a start/goal with L={L}")

    def step(self, action):
        """action: displacement (kinematic). double_integrator reserved for later rungs."""
        cfg = self.cfg
        if cfg.dynamics != "kinematic":
            raise NotImplementedError("double_integrator: wire up when needed; kinematic is the rung-3 default")
        self.state = self.step_fn(self.state, action)
        done = np.linalg.norm(self.state - self.goal) < cfg.goal_eps
        return self._obs(), float(done), done, {}

    def _obs(self):
        return np.concatenate([self.state, self.goal])   # full observability for v0

    # --- pure true-dynamics steps for the executor (no side effects on self.state) ---
    def step_fn(self, state, action):
        return self.step_batch(np.asarray(state, float)[None],
                               np.asarray(action, float)[None])[0]

    def step_batch(self, states, actions):
        """Vectorized kinematic step. states[N,2], actions[N,2] -> next[N,2].
        This is what BoundedMPPI calls to roll out all samples at once."""
        a = _clip_step(np.asarray(actions, float), self.cfg.max_step)
        P = np.asarray(states, float)
        return collide_batch(P, P + a, self.wA, self.wB)
