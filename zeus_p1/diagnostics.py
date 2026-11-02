"""Milestone 7: reachability residuals (from plan.py replan logs) + offline cross-scale consistency gap."""
import json, os
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_device, parse
from data import Data
from model import load_model


@torch.no_grad()
def level2_step_norm(model, data):
    """Median ||z2_{t+1} - z2_t||^2 over held-out data (stride-2 consecutive level-1 frames, E2 applied pointwise)."""
    fr, _, _, _ = data.all_frames("val")
    z1 = torch.cat([model.E1(fr[i:i + 1024].float() / 255.0) for i in range(0, len(fr), 1024)])
    z2 = model.e2(z1).view(len(data.val_ids), data.L, -1)
    s = model.cfg.stride
    return ((z2[:, s:] - z2[:, :-s]) ** 2).sum(-1).median().item()


@torch.no_grad()
def consistency_gap(model, data, norm, n_clips=2000, seed=0):
    """Open-loop L1 rollout (2k steps, true actions) lifted through E2 vs open-loop L2 rollout (k steps, pooled macro-actions)
    and vs the encoded z2_k. Returns dict of arrays over k=1..3 (normalised by `norm`)."""
    g = torch.Generator(device=data.dev); g.manual_seed(seed)
    fr, act, _, _ = data.sample(n_clips, "val", g)
    model.eval()
    z1 = model.encode_clip(fr); a1 = model.A1(act)
    z = z1[:, 0]; z1hat = [z]
    for j in range(a1.shape[1]):
        z = model.F1(z, a1[:, j]); z1hat.append(z)
    z1hat = torch.stack(z1hat, 1)                      # [B,7,d]
    a2 = model.a2(a1)                                  # [B,3,4]
    z2 = model.e2(z1[:, ::2])                          # true encoded z2_0..3
    zz = z2[:, 0]; z2hat = [zz]
    for k in range(a2.shape[1]):
        zz = model.F2(zz, a2[:, k]); z2hat.append(zz)
    z2hat = torch.stack(z2hat, 1)                      # [B,4,d]
    lifted = model.e2(z1hat[:, ::2])                   # E2(z1hat_{2k})
    sq = lambda a, b: ((a - b) ** 2).sum(-1).mean(0)[1:].cpu().numpy() / norm
    return dict(k=np.arange(1, 4), gap=sq(lifted, z2hat), l1_drift=sq(lifted, z2), l2_drift=sq(z2hat, z2),
                l1_drift_pure=sq(z1hat[:, ::2], z1[:, ::2]) / 1.0)


def reach_stats(out_dir, S, norm, mode="hier"):
    rp = pd.read_csv(f"{out_dir}/replans_{mode}_S{S}.csv"); ep = pd.read_csv(f"{out_dir}/episodes_{mode}_S{S}.csv")
    rp = rp.merge(ep[["task", "success_any", "success_final"]], on="task")
    rp["r_plan"] = rp.r_plan_raw / norm; rp["r_real"] = rp.r_real_raw / norm
    sub = rp[rp.H2 > 1]  # replans that actually used a level-2 subgoal
    res = dict(S=S, n_replans=len(sub), success_rate=float(ep.success_any.mean()), median_r_plan=float(sub.r_plan.median()),
               median_r_real=float(sub.r_real.median()), clip_rate_mean=float(sub.clip_rate.mean()))
    for name, m in (("all", sub.success_any >= 0), ("success", sub.success_any == 1), ("fail", sub.success_any == 0)):
        s = sub[m]
        res[f"unreach>0.5_{name}"] = float((s.r_real > 0.5).mean()) if len(s) else np.nan
        res[f"unreach>1.0_{name}"] = float((s.r_real > 1.0).mean()) if len(s) else np.nan
        res[f"clip_rate_{name}"] = float(s.clip_rate.mean()) if len(s) else np.nan
    # per-episode unreachable rate vs failure
    pe = sub.groupby("task").agg(u=("r_real", lambda x: (x > 0.5).mean()), ok=("success_any", "first"))
    res["spearman_unreach_vs_success"] = float(pe.u.corr(pe.ok, method="spearman")) if pe.ok.nunique() > 1 else np.nan
    res["plan_vs_real_corr"] = float(sub.r_plan.corr(sub.r_real, method="spearman"))
    return res, sub


def main():
    cfg0, ex = parse(extra=dict(S=(str, "64"), tag=(str, ""), mode=(str, "hier")), description="diagnostics for a trained run")
    dev = get_device(cfg0.device or "cpu")
    model, cfg, _ = load_model(cfg0.run_dir, dev)
    assert cfg.levels >= 2
    data = Data(cfg, dev)
    out = f"{cfg0.run_dir}/diag"; os.makedirs(out, exist_ok=True)
    norm = level2_step_norm(model, data)
    print(f"normaliser (median one-step level-2 move ||dz2||^2) = {norm:.4f}")
    cg = consistency_gap(model, data, norm)
    pd.DataFrame(cg).to_csv(f"{out}/consistency_gap.csv", index=False)
    print(pd.DataFrame(cg).round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for key, lab in (("gap", "L1-lifted vs L2 rollout (consistency gap)"), ("l1_drift", "L1-lifted vs true z2 (L1 drift)"), ("l2_drift", "L2 rollout vs true z2 (L2 drift)")):
        ax.plot(cg["k"], cg[key], "o-", label=lab)
    ax.set_xlabel("k (level-2 steps = 2k level-1 steps)"); ax.set_ylabel("normalised sq. error"); ax.set_xticks([1, 2, 3]); ax.legend(fontsize=6); ax.set_title("open-loop consistency")
    plt.tight_layout(); plt.savefig(f"{out}/consistency_gap.png", dpi=100); plt.close()

    allres = []
    for S in [int(s) for s in ex["S"].split(",")]:
        pdir = f"{cfg0.run_dir}/plan{ex['tag']}"
        if not os.path.exists(f"{pdir}/replans_{ex['mode']}_S{S}.csv"):
            continue
        res, sub = reach_stats(pdir, S, norm, ex["mode"]); allres.append(res)
        print(json.dumps(res, indent=1))
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        bins = np.linspace(0, max(2.5, sub.r_real.quantile(.99)), 40)
        for lab, m in (("success", sub.success_any == 1), ("fail", sub.success_any == 0)):
            if m.any():
                ax[0].hist(sub.r_real[m], bins=bins, alpha=.55, density=True, label=f"{lab} (n={m.sum()})")
        ax[0].axvline(.5, c="k", ls=":"); ax[0].axvline(1., c="k", ls="--"); ax[0].set_xlabel("normalised r_real"); ax[0].legend(); ax[0].set_title(f"realised subgoal residual (S={S})")
        ax[1].scatter(sub.r_plan, sub.r_real, s=4, alpha=.3); ax[1].set_xlabel("normalised r_plan"); ax[1].set_ylabel("normalised r_real"); ax[1].set_title("planned vs realised")
        plt.tight_layout(); plt.savefig(f"{out}/reachability_S{S}.png", dpi=100); plt.close()
    if allres:
        pd.DataFrame(allres).to_csv(f"{out}/reachability_summary.csv", index=False)
    json.dump(dict(norm=norm, consistency=cg, reach=allres), open(f"{out}/diag.json", "w"), indent=1, default=lambda x: x.tolist())


if __name__ == "__main__":
    main()
