import json
import torch
import torch.nn as nn

from config import Config
from sigreg import sigreg


class ResPredictor(nn.Module):
    """z_{t+1} = z_t + MLP([z_t, a_t])"""

    def __init__(self, d, da, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + da, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))


class Encoder1(nn.Module):
    def __init__(self, d):
        super().__init__()
        c = [3, 32, 64, 128, 128]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 4, 2, 1), nn.GELU()]
        self.conv = nn.Sequential(*layers)  # 64 -> 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.GELU(), nn.Linear(256, d), nn.BatchNorm1d(d))

    def forward(self, x):
        return self.head(self.conv(x))


class HJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.levels in (1, 2), "levels=3 not implemented"
        self.cfg, d = cfg, cfg.d
        self.E1 = Encoder1(d)
        self.A1 = nn.Sequential(nn.Linear(10, 64), nn.GELU(), nn.Linear(64, 32))
        self.F1 = ResPredictor(d, 32)
        if cfg.levels >= 2:
            self.E2 = nn.Identity() if cfg.hwm else nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d), nn.BatchNorm1d(d))
            a2 = [nn.Linear(64, 64), nn.GELU(), nn.Linear(64, cfg.a2_dim)]
            if cfg.a2_ln:
                a2.append(nn.LayerNorm(cfg.a2_dim))
            self.A2 = nn.Sequential(*a2)
            self.F2 = ResPredictor(d, cfg.a2_dim)

    def l1_params(self):
        return [p for m in (self.E1, self.A1, self.F1) for p in m.parameters()]

    def l2_params(self):
        return [p for m in (self.E2, self.A2, self.F2) for p in m.parameters()] if self.cfg.levels >= 2 else []

    def e2(self, z1):
        """Apply E2 to [..., d] (BatchNorm needs a 2D input)."""
        s = z1.shape
        return self.E2(z1.reshape(-1, s[-1])).reshape(s)

    def a2(self, a1):
        """a1: [B, 2k, 32] embeddings -> pooled macro-actions [B, k, a2_dim] (a2_t pools a1_{2t}, a1_{2t+1})."""
        B, n, e = a1.shape
        return self.A2(a1.reshape(B, n // 2, 2 * e))

    def _pred(self, pred, target):
        sq = ((pred - target) ** 2)
        return sq.sum(-1).mean() if self.cfg.pred_norm == "sum" else sq.mean()

    def encode_clip(self, frames):
        B, n = frames.shape[:2]
        return self.E1(frames.flatten(0, 1)).view(B, n, -1)

    def losses(self, frames, act):
        """frames [B,7,3,H,W], act [B,6,10]. Returns dict of scalars (+ a2, z1, z2 for logging/buffer)."""
        cfg = self.cfg
        z1 = self.encode_clip(frames)
        a1 = self.A1(act)
        l1 = self._pred(self.F1(z1[:, :-1], a1), z1[:, 1:])
        out = dict(l1=l1, sig1=sigreg(z1.flatten(0, 1)), z1=z1)
        loss = l1 + cfg.lam * out["sig1"]
        if cfg.levels >= 2:
            s = cfg.stride
            z2 = self.e2(z1[:, ::s])                        # z2_t = E2(z1_{2t}), t=0..3
            a2 = self.a2(a1)                                # [B,3,a2_dim]
            l2 = self._pred(self.F2(z2[:, :-1], a2), z2[:, 1:])
            out.update(l2=l2, z2=z2, a2=a2)
            loss = loss + l2
            if not cfg.hwm:
                out["sig2"] = sigreg(z2.flatten(0, 1))
                loss = loss + cfg.lam * out["sig2"]
            else:
                out["sig2"] = torch.zeros((), device=l1.device)
        out["loss"] = loss
        return out


def load_model(run_dir, device, ckpt="final.pt"):
    ck = torch.load(f"{run_dir}/{ckpt}", map_location="cpu")
    cfg = Config(**ck["cfg"])
    m = HJEPA(cfg)
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg, ck["a2_buffer"].to(device) if ck.get("a2_buffer") is not None else None
