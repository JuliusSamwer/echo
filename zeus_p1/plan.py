"""Milestone 6/7: goal-conditioned planning (flat / flat+projected / hierarchical) on FourRoom + reachability logging.

All planners run batched over tasks: candidate tensors are [T, S, H, dim]; AdamW is elementwise so tasks stay independent.
Compute proxy = predictor-step evaluations (forward + backward counted separately), per task.
"""
import csv, json, os, time
import numpy as np
import torch

import env
from config import get_device, parse
from model import load_model

TASK_SEED = 1234


# ------------------------------------------------------------------ tasks
def make_tasks(n=50, seed=TASK_SEED, T_goal=75, p_close=0.5):
    rng = np.random.default_rng(seed)
    r0 = rng.integers(0, 4, n); r1 = 3 - r0  # opposite (diagonal) room
    lo = lambda r: np.stack([(r % 2) * .5 + .08, (r // 2) * .5 + .08], 1)
    start = lo(r0) + rng.uniform(0, .5 - .16 - .04, (n, 2)) * 1.0
    goal = lo(r1) + rng.uniform(0, .5 - .16 - .04, (n, 2)) * 1.0
    doors = env.sample_doors(n, p_close, rng)
    st = env.reset(n, rng, doors=doors, agent=start)
    start_state = {k: v.copy() for k, v in st.items()}
    for _ in range(T_goal):  # noiseless expert; stays put once at the goal
        wp = env.next_waypoint(st["agent"], goal, doors)
        vec = wp - st["agent"]; nrm = np.linalg.norm(vec, axis=1, keepdims=True)
        a = np.where(np.linalg.norm(goal - st["agent"], axis=1, keepdims=True) < 0.03, 0, vec / (nrm + 1e-8))
        st = env.step(st, a, rng)
    return dict(start_state=start_state, goal_agent=st["agent"].copy(), goal_dist=st["dist"].copy(), start_agent=start)


def load_tasks(path="runs/tasks_p05.npz", n=50):
    if not os.path.exists(path):
        t = make_tasks(n)
        flat = {f"s_{k}": v for k, v in t["start_state"].items()}
        np.savez(path, goal_agent=t["goal_agent"], goal_dist=t["goal_dist"], start_agent=t["start_agent"], **flat)
    d = np.load(path)
    return dict(start_state={k[2:]: d[k] for k in d.files if k.startswith("s_")}, goal_agent=d["goal_agent"], goal_dist=d["goal_dist"])


# ------------------------------------------------------------------ planner
class Planner:
    def __init__(self, model, a2buf, dev, lr=0.1, iters=60, patience=8):
        self.m, self.dev, self.lr, self.iters, self.patience = model, dev, lr, iters, patience
        self.d = model.cfg.d
        self.hier = model.cfg.levels >= 2
        if self.hier:
            self.mu, self.sd = a2buf.mean(0), a2buf.std(0)
            self.lo, self.hi = torch.quantile(a2buf, .02, dim=0), torch.quantile(a2buf, .98, dim=0)
        self.calls = 0  # predictor-step evaluations (fwd+bwd)

    # rollouts -> states [T,S,H+1,d]
    def roll1(self, z0, a):
        a1 = self.m.A1(a); z = z0[:, None].expand(-1, a.shape[1], -1); zs = [z]
        for h in range(a.shape[2]):
            z = self.m.F1(z, a1[:, :, h]); zs.append(z)
        return torch.stack(zs, 2)

    def roll2(self, z0, a2):
        z = z0[:, None].expand(-1, a2.shape[1], -1); zs = [z]
        for h in range(a2.shape[2]):
            z = self.m.F2(z, a2[:, :, h]); zs.append(z)
        return torch.stack(zs, 2)

    def optimize(self, init, cost_fn, H, project=None):
        T, S = init.shape[:2]
        p = init.clone().requires_grad_(True)
        opt = torch.optim.AdamW([p], lr=self.lr, weight_decay=0.0)
        best, stall = float("inf"), 0
        for _ in range(self.iters):
            c = cost_fn(p)
            opt.zero_grad(); c.sum().backward(); opt.step()
            self.calls += 2 * T * S * H
            if project is not None:
                with torch.no_grad():
                    project(p)
            cur = c.detach().min(1).values.mean().item()
            if cur < best * (1 - 1e-3):
                best, stall = cur, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
        with torch.no_grad():
            c = cost_fn(p)
        idx = c.argmin(1)
        return p.detach(), c.detach(), idx

    # ---- flat planners
    def plan_flat(self, z1, g1, g2, mode, S, H):
        T = len(z1)
        u0 = torch.randn(T, S, H, 10, device=self.dev)
        tgt = g1 if mode == "flat" else g2

        def cost(u):
            last = self.roll1(z1, torch.tanh(u))[:, :, -2:]  # score last 2 states
            if mode == "proj":
                last = self.m.e2(last)
            return ((last - tgt[:, None, None]) ** 2).sum(-1).sum(-1)
        u, c, idx = self.optimize(u0, cost, H)
        ar = torch.arange(T, device=self.dev)
        return torch.tanh(u[ar, idx]), {}

    # ---- hierarchical planner
    def plan_hier(self, z1, g2, H2, S1, S2):
        T, ar = len(z1), torch.arange(len(z1), device=self.dev)
        z2 = self.m.e2(z1)
        info = dict(H2=H2)
        if H2 > 1:
            init = self.mu + self.sd * torch.randn(T, S2, H2, self.m.cfg.a2_dim, device=self.dev)
            init[:, 0] = self.mu
            proj = lambda p: p.data.copy_(torch.max(torch.min(p.data, self.hi), self.lo))
            proj(init)
            cost2 = lambda p: ((self.roll2(z2, p)[:, :, -1] - g2[:, None]) ** 2).sum(-1)
            p, c2, idx2 = self.optimize(init, cost2, H2, proj)
            pb = p[ar, idx2]                                        # [T,H2,a2]
            with torch.no_grad():
                sub = self.roll2(z2, pb[:, None])[:, 0, 1]          # z2_hat_1*
            eps = 1e-6
            info["clip_rate"] = ((pb <= self.lo + eps) | (pb >= self.hi - eps)).float().mean((1, 2)).cpu().numpy()
            info["cost2"] = c2[ar, idx2].cpu().numpy()
            target = sub
        else:
            target = g2
            info["clip_rate"] = np.full(T, np.nan); info["cost2"] = np.full(T, np.nan)
        u0 = torch.randn(T, S1, 2, 10, device=self.dev)               # H1 = K*s + w - 1 = 2 blocks
        cost1 = lambda u: ((self.m.e2(self.roll1(z1, torch.tanh(u))[:, :, 2]) - target[:, None]) ** 2).sum(-1)
        u, c1, idx1 = self.optimize(u0, cost1, 2)
        info["r_plan"] = c1[ar, idx1].cpu().numpy()
        info["subgoal"] = target if H2 > 1 else None
        return torch.tanh(u[ar, idx1]), info


# ------------------------------------------------------------------ closed-loop evaluation
@torch.no_grad()
def encode_obs(model, st, dev, res=64):
    fr = env.render(torch.tensor(st["agent"], dtype=torch.float32, device=dev), torch.tensor(st["dist"], dtype=torch.float32, device=dev),
                    torch.tensor(st["doors"], device=dev), res).float() / 255.0
    return model.E1(fr)


def run_eval(model, a2buf, tasks, mode, S, dev, seed=0, budget=210, flat_H=16, h2_start=12, S2=None, exec_blocks=2, log=None):
    planner = Planner(model, a2buf, dev)
    S1, S2 = S, (S2 or 8 * S)
    T = len(tasks["goal_agent"])
    st = {k: v.copy() for k, v in tasks["start_state"].items()}
    rng = np.random.default_rng(10_000 + seed)
    torch.manual_seed(seed)
    gst = dict(agent=tasks["goal_agent"], dist=tasks["goal_dist"], doors=st["doors"])
    g1 = encode_obs(model, gst, dev); g2 = model.e2(g1) if model.cfg.levels >= 2 else None
    goal = tasks["goal_agent"]
    mind = np.linalg.norm(st["agent"] - goal, axis=1); t_env = 0
    n_replans = -(-budget // (exec_blocks * 5))
    prev_sub, prev_H2, records = None, None, []
    for r in range(n_replans):
        z1 = encode_obs(model, st, dev)
        if prev_sub is not None:  # realized reachability residual of the previous subgoal
            r_real = ((model.e2(z1) - prev_sub) ** 2).sum(-1).cpu().numpy()
            for i in range(T):
                records[-T + i]["r_real_raw"] = float(r_real[i])
        if mode == "hier":
            H2 = max(1, h2_start - r)
            acts, info = planner.plan_hier(z1, g2, H2, S1, S2)
            prev_sub = info["subgoal"]
            for i in range(T):
                records.append(dict(task=i, replan=r, H2=H2, r_plan_raw=float(info["r_plan"][i]), r_real_raw=np.nan,
                                    clip_rate=float(info["clip_rate"][i]), cost2=float(info["cost2"][i]),
                                    dist_to_goal=float(np.linalg.norm(st["agent"][i] - goal[i]))))
        else:
            acts, _ = planner.plan_flat(z1, g1, g2, mode, S1, flat_H)
        a = acts[:, :exec_blocks].cpu().numpy().reshape(T, exec_blocks * 5, 2)
        for k in range(exec_blocks * 5):
            st = env.step(st, a[:, k], rng); t_env += 1
            mind = np.minimum(mind, np.linalg.norm(st["agent"] - goal, axis=1))
    if mode == "hier" and prev_sub is not None:  # r_real for the last replan
        r_real = ((model.e2(encode_obs(model, st, dev)) - prev_sub) ** 2).sum(-1).cpu().numpy()
        for i in range(T):
            records[-T + i]["r_real_raw"] = float(r_real[i])
    final = np.linalg.norm(st["agent"] - goal, axis=1)
    return dict(success_any=(mind <= 0.05), success_final=(final <= 0.05), min_dist=mind, final_dist=final,
                compute_per_task=planner.calls / T, records=records)


def main():
    cfg0, ex = parse(extra=dict(planners=(str, "flat,proj,hier"), S=(str, "100"), S2=(int, 800), n_tasks=(int, 50), plan_device=(str, "cpu"),
                                budget=(int, 210), flat_H=(int, 16), tag=(str, "")), description="evaluate planners on a trained run")
    dev = get_device(ex["plan_device"])
    model, cfg, buf = load_model(cfg0.run_dir, dev)
    tasks = load_tasks(n=ex["n_tasks"])
    out = f"{cfg0.run_dir}/plan{ex['tag']}"; os.makedirs(out, exist_ok=True)
    rows = []
    for S in [int(s) for s in ex["S"].split(",")]:
        for mode in ex["planners"].split(","):
            if mode == "hier" and cfg.levels < 2:
                continue
            if mode == "proj" and (cfg.levels < 2 or cfg.hwm):
                continue  # identical to flat for HWM (E2 = id), undefined for 1 level
            t0 = time.time()
            res = run_eval(model, buf, tasks, mode, S, dev, seed=cfg0.seed, budget=ex["budget"], flat_H=ex["flat_H"], S2=ex["S2"] if mode == "hier" else None)
            row = dict(run=cfg0.run_dir, planner=mode, S=S, success_any=res["success_any"].mean(), success_final=res["success_final"].mean(),
                       mean_min_dist=res["min_dist"].mean(), compute_per_task=res["compute_per_task"], sec=time.time() - t0)
            rows.append(row); print(row, flush=True)
            with open(f"{out}/episodes_{mode}_S{S}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["task", "success_any", "success_final", "min_dist", "final_dist"])
                for i in range(len(res["min_dist"])):
                    w.writerow([i, int(res["success_any"][i]), int(res["success_final"][i]), res["min_dist"][i], res["final_dist"][i]])
            if res["records"]:
                with open(f"{out}/replans_{mode}_S{S}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(res["records"][0])); w.writeheader(); w.writerows(res["records"])
    with open(f"{out}/summary.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); 
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
