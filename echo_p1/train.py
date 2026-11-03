import csv, math, os, time, json
import numpy as np
import torch

from config import cfg_to_dict, get_device, parse
from data import Data
from model import HJEPA


def eff_rank(Z):
    Z = Z.detach().float().cpu()
    Z = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()


@torch.no_grad()
def evaluate(model, data, cfg, gen):
    model.eval()
    fr, act, _, _ = data.sample(256, "val", gen)
    o = model.losses(fr, act)
    r = dict(val_l1=o["l1"].item(), val_sig1=o["sig1"].item(), rank1=eff_rank(o["z1"].flatten(0, 1)))
    if cfg.levels >= 2:
        r.update(val_l2=o["l2"].item(), val_sig2=o["sig2"].item(), rank2=eff_rank(o["z2"].flatten(0, 1)))
    model.train()
    return r


def main():
    cfg = parse(description="train flat LeWM (--levels 1), H-JEPA (--levels 2) or HWM (--levels 2 --hwm)")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = get_device(cfg.device)
    os.makedirs(cfg.run_dir, exist_ok=True)
    json.dump(cfg_to_dict(cfg), open(f"{cfg.run_dir}/config.json", "w"), indent=1)
    data = Data(cfg, dev)
    model = HJEPA(cfg).to(dev)
    groups = [dict(params=model.l1_params(), lr=cfg.lr, base=cfg.lr)]
    if cfg.levels >= 2:
        groups.append(dict(params=model.l2_params(), lr=cfg.lr * cfg.l2_lr_mult, base=cfg.lr * cfg.l2_lr_mult))
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    warm = max(1, int(0.01 * cfg.steps))
    gen_val = torch.Generator(device=dev); gen_val.manual_seed(1234)
    gen_tr = torch.Generator(device=dev); gen_tr.manual_seed(cfg.seed)
    buf = torch.zeros(0, cfg.a2_dim, device=dev)
    log = open(f"{cfg.run_dir}/log.csv", "w", newline=""); w = None
    t0 = time.time()

    def save(name):
        torch.save(dict(cfg=cfg_to_dict(cfg), model=model.state_dict(), a2_buffer=buf.cpu() if cfg.levels >= 2 else None),
                   f"{cfg.run_dir}/{name}")

    model.train()
    for step in range(1, cfg.steps + 1):
        mult = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, cfg.steps - warm)))
        for g in opt.param_groups:
            g["lr"] = g["base"] * mult
        fr, act, _, _ = data.sample(cfg.batch, "train", gen_tr)
        o = model.losses(fr, act)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        if cfg.levels >= 2:
            buf = torch.cat([buf, o["a2"].detach().reshape(-1, cfg.a2_dim)])[-cfg.a2_buf:]
        if step % cfg.log_every == 0 or step == 1:
            row = dict(step=step, time=round(time.time() - t0, 1), loss=o["loss"].item(), l1=o["l1"].item(), sig1=o["sig1"].item())
            if cfg.levels >= 2:
                row.update(l2=o["l2"].item(), sig2=o["sig2"].item())
            row.update(evaluate(model, data, cfg, gen_val))
            if w is None:
                w = csv.DictWriter(log, fieldnames=list(row)); w.writeheader()
            w.writerow(row); log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if row["rank1"] < 3:
                print("  WARNING: z1 effective rank < 3 (collapse?) -> consider raising --lam", flush=True)
        if step % cfg.ckpt_every == 0:
            save("ckpt.pt")
    save("final.pt")
    print("done", cfg.run_dir, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
