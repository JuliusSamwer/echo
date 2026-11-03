"""Milestone 5: MLP probes (z1/z2 -> agent_xy / distractor_xy) + cost-geometry heatmaps."""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import env
from config import get_device, parse
from data import Data
from model import load_model


@torch.no_grad()
def encode_all(model, fr, bs=1024):
    z1 = torch.cat([model.E1(fr[i:i + bs].float() / 255.0) for i in range(0, len(fr), bs)])
    z2 = model.e2(z1) if model.cfg.levels >= 2 else None
    return z1, z2


def fit_probe(Xtr, Ytr, Xte, Yte, seed, steps=3000, dev="cpu"):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Ytr.shape[1])).to(dev)
    # "2-layer MLP (hidden 256, GELU)" = one hidden layer; use exactly that:
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, Ytr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(len(Xtr), (512,))
        loss = ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        mse = ((net(Xte) - Yte) ** 2).mean(0)
    return (mse / Yte.var(0)).mean().item()  # NMSE averaged over dims


def probe_run(model, cfg, data, seed=0):
    fr, ag, ds, ep = data.all_frames("val")
    z1, z2 = encode_all(model, fr)
    z1, z2, ag, ds = z1.cpu(), (z2.cpu() if z2 is not None else None), ag.cpu(), ds.cpu()
    ep = ep.cpu()
    ueps = ep.unique(); g = torch.Generator().manual_seed(seed)
    perm = ueps[torch.randperm(len(ueps), generator=g)]
    te = torch.isin(ep, perm[: int(0.2 * len(perm))])  # split by episode, 80/20
    res = {}
    for lvl, z in (("z1", z1), ("z2", z2)):
        if z is None:
            continue
        for tgt, y in (("agent", ag), ("distractor", ds)):
            res[f"{lvl}_{tgt}"] = fit_probe(z[~te], y[~te], z[te], y[te], seed)
    return res


@torch.no_grad()
def heatmaps(model, cfg, path, anchors=((0.2, 0.2), (0.8, 0.8)), n=48, dist_xy=(0.35, 0.8)):
    dev = next(model.parameters()).device
    g = (torch.arange(n, dtype=torch.float32) + .5) / n
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gx.flatten(), gy.flatten()], 1).to(dev)
    D = torch.tensor(dist_xy, dtype=torch.float32, device=dev).expand(len(pts), 2)
    z1 = model.E1(env.render(pts, D, None, cfg.res).float() / 255.0)
    z2 = model.e2(z1) if cfg.levels >= 2 else None
    wall = env.in_wall(pts.cpu().numpy(), np.ones((len(pts), 4), bool)).reshape(n, n)
    levels = [("level 1 (z1)", z1)] + ([("level 2 (z2)", z2)] if z2 is not None else [])
    fig, ax = plt.subplots(len(anchors), len(levels), figsize=(4.2 * len(levels), 4 * len(anchors)), squeeze=False)
    for i, (ax_, ay_) in enumerate(anchors):
        a = torch.tensor([[ax_, ay_]], dtype=torch.float32, device=dev)
        za1 = model.E1(env.render(a, D[:1], None, cfg.res).float() / 255.0)
        for j, (ttl, z) in enumerate(levels):
            za = za1 if j == 0 else model.e2(za1)
            c = ((z - za) ** 2).sum(1).cpu().numpy().reshape(n, n)
            c = c / np.median(c)
            c = np.ma.masked_where(wall, c)
            im = ax[i, j].imshow(c, extent=[0, 1, 1, 0], cmap="viridis")
            ax[i, j].plot(ax_, ay_, "r*", ms=12); ax[i, j].set_title(f"{ttl}  anchor=({ax_},{ay_})"); plt.colorbar(im, ax=ax[i, j], fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


if __name__ == "__main__":
    cfg0 = parse(description="probe a trained run (--name/--seed select runs/<name>/seed<k>)")
    dev = get_device(cfg0.device)
    model, cfg, _ = load_model(cfg0.run_dir, dev)
    data = Data(cfg, dev)
    res = probe_run(model, cfg, data, seed=cfg0.seed)
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{cfg0.run_dir}/probe.json", "w"), indent=1)
    heatmaps(model, cfg, f"{cfg0.run_dir}/cost_heatmaps.png")
