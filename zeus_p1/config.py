import argparse
import dataclasses
from dataclasses import dataclass, fields

import torch


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Config:
    name: str = "hjepa"
    seed: int = 0
    data_seed: int = 0
    device: str = ""
    # env
    door_close_prob: float = 0.5   # paper: each door closed w.p. 0.5 at reset (>=3 kept open so rooms stay connected)
    res: int = 64
    # data
    n_episodes: int = 3750
    n_val: int = 150
    ep_len: int = 400
    frameskip: int = 5
    data_path: str = "data/fourroom_p05.npz"
    # model
    d: int = 64
    levels: int = 2          # 1 = flat LeWM, 2 = H-JEPA (3 not implemented)
    hwm: bool = False        # E2 = identity, no SIGReg on level 2
    stride: int = 2
    a2_dim: int = 4
    a2_ln: bool = False
    pred_norm: str = "sum"   # paper Eq.2: squared L2 norm ("mean" = per-dim MSE)
    # train
    batch: int = 128
    steps: int = 20000
    lr: float = 3e-4
    l2_lr_mult: float = 0.2
    wd: float = 1e-3
    lam: float = 0.18        # paper FourRoom: 0.18 for H-JEPA/HWM, 0.72 for flat LeWM (pass --lam 0.72)
    clip: float = 1.0
    log_every: int = 200
    ckpt_every: int = 5000
    a2_buf: int = 4000
    out_dir: str = "runs"

    @property
    def n1(self):
        return (4 - 1) * self.stride + 1  # level-1 frames per clip (2 levels)

    @property
    def run_dir(self):
        return f"{self.out_dir}/{self.name}/seed{self.seed}"


def _bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def parse(argv=None, extra=None, description=""):
    """Every Config field becomes a --flag. `extra` = {flag: (type, default)} for script-specific args."""
    p = argparse.ArgumentParser(description=description)
    for f in fields(Config):
        if f.type is bool:
            p.add_argument(f"--{f.name}", type=_bool, nargs="?", const=True, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=f.type, default=f.default)
    for k, (t, dflt) in (extra or {}).items():
        if t is bool:
            p.add_argument(f"--{k}", type=_bool, nargs="?", const=True, default=dflt)
        else:
            p.add_argument(f"--{k}", type=t, default=dflt)
    ns = p.parse_args(argv)
    cfg = Config(**{f.name: getattr(ns, f.name) for f in fields(Config)})
    rest = {k: getattr(ns, k) for k in (extra or {})}
    return (cfg, rest) if extra else cfg


def cfg_to_dict(cfg):
    return dataclasses.asdict(cfg)
