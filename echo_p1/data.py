"""Data collection (states + actions only) and on-the-fly clip sampling/rendering."""
import os
import numpy as np
import torch

import env
from config import Config, get_device, parse


def collect(cfg: Config, seed=None):
    rng = np.random.default_rng(cfg.data_seed if seed is None else seed)
    N, T, fs = cfg.n_episodes, cfg.ep_len, cfg.frameskip
    doors = env.sample_doors(N, cfg.door_close_prob, rng)
    st = env.reset(N, rng, doors=doors)
    expert = np.zeros(N, bool); expert[rng.permutation(N)[: N // 2]] = True
    goal = env.sample_free(N, rng)
    ou = np.zeros((N, 2))
    A = np.zeros((N, T, 2), np.float32)
    ag = np.zeros((N, T // fs + 1, 2), np.float32); ds = np.zeros_like(ag)
    for t in range(T + 1):
        if t % fs == 0:
            ag[:, t // fs], ds[:, t // fs] = st["agent"], st["dist"]
        if t == T:
            break
        wp = env.next_waypoint(st["agent"], goal, doors)
        vec = wp - st["agent"]
        unit = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
        a_exp = np.clip(unit + 0.5 * rng.standard_normal((N, 2)), -1, 1)
        ou = ou * (1 - 0.1) + 0.4 * rng.standard_normal((N, 2))
        a = np.where(expert[:, None], a_exp, np.clip(ou, -1, 1))
        A[:, t] = a
        st = env.step(st, a, rng)
        reached = np.linalg.norm(goal - st["agent"], axis=1) < 0.03
        goal = np.where(reached[:, None], env.sample_free(N, rng), goal)
    return dict(agent=ag, dist=ds, act=A.reshape(N, T // fs, 2 * fs), doors=doors, expert=expert)


class Data:
    """Holds all episodes on-device; samples clips of n1 level-1 frames and renders them on the fly."""

    def __init__(self, cfg: Config, device, n1=None):
        if not os.path.exists(cfg.data_path):
            os.makedirs(os.path.dirname(cfg.data_path), exist_ok=True)
            d = collect(cfg)
            np.savez_compressed(cfg.data_path, **d)
        d = np.load(cfg.data_path)
        self.cfg, self.dev = cfg, device
        self.n1 = n1 or cfg.n1
        self.agent = torch.tensor(d["agent"], device=device)
        self.dist = torch.tensor(d["dist"], device=device)
        self.act = torch.tensor(d["act"], device=device)
        self.doors = torch.tensor(d["doors"], device=device)
        N = self.agent.shape[0]
        self.val_ids = torch.arange(N - cfg.n_val, N, device=device)
        self.train_ids = torch.arange(0, N - cfg.n_val, device=device)
        self.L = self.agent.shape[1]  # level-1 frames per episode (41)

    def ids(self, split):
        return self.train_ids if split == "train" else self.val_ids

    def render(self, ag, ds, doors=None):
        return env.render(ag.reshape(-1, 2), ds.reshape(-1, 2), doors, self.cfg.res).float().div_(255.0)

    def sample(self, B, split="train", gen=None):
        ids = self.ids(split)
        ep = ids[torch.randint(len(ids), (B,), device=self.dev, generator=gen)]
        k0 = torch.randint(0, self.L - self.n1 + 1, (B,), device=self.dev, generator=gen)
        t = k0[:, None] + torch.arange(self.n1, device=self.dev)[None]  # [B,n1]
        ag, ds = self.agent[ep[:, None], t], self.dist[ep[:, None], t]
        act = self.act[ep[:, None], t[:, :-1]]
        doors = self.doors[ep][:, None].expand(-1, self.n1, -1).reshape(-1, 4)
        fr = self.render(ag, ds, doors).view(B, self.n1, 3, self.cfg.res, self.cfg.res)
        return fr, act, ag, ds

    def all_frames(self, split="val"):
        """Every level-1 frame of the split's episodes (no striding): frames [E*L,3,H,W], agent, dist, ep index."""
        ids = self.ids(split)
        ag, ds = self.agent[ids], self.dist[ids]
        doors = self.doors[ids][:, None].expand(-1, self.L, -1).reshape(-1, 4)
        fr = self.render(ag, ds, doors)
        ep = ids[:, None].expand(-1, self.L).reshape(-1)
        return fr, ag.reshape(-1, 2), ds.reshape(-1, 2), ep


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = parse(description="collect data + coverage plot")
    dev = get_device(cfg.device or "cpu")
    data = Data(cfg, dev)
    ag = data.agent.cpu().numpy().reshape(-1, 2)
    print(f"episodes {data.agent.shape[0]}  transitions {data.agent.shape[0] * cfg.ep_len}  agent range {ag.min(0)} {ag.max(0)}")
    room = env.room_of(ag); print("fraction per room:", np.bincount(room, minlength=4) / len(room))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (m, ttl) in zip(ax[:2], [(data_e := np.load(cfg.data_path)["expert"], "expert"), (None, "")]):
        pass
    ex = np.load(cfg.data_path)["expert"]
    for a, sel, ttl in [(ax[0], ex, "noisy expert"), (ax[1], ~ex, "smooth random")]:
        p = data.agent.cpu().numpy()[sel].reshape(-1, 2)
        a.hist2d(p[:, 0], p[:, 1], bins=50, range=[[0, 1], [0, 1]]); a.set_title(ttl); a.set_aspect("equal"); a.invert_yaxis()
    p = data.dist.cpu().numpy().reshape(-1, 2)
    ax[2].hist2d(p[:, 0], p[:, 1], bins=50, range=[[0, 1], [0, 1]]); ax[2].set_title("distractor"); ax[2].set_aspect("equal"); ax[2].invert_yaxis()
    plt.tight_layout(); plt.savefig("data/coverage.png", dpi=80)
    fr, act, a, d = data.sample(4); print("sample shapes", fr.shape, act.shape, a.shape, d.shape, fr.min().item(), fr.max().item())
