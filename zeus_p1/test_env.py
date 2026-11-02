"""Milestone 1: env + renderer checks."""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import env

rng = np.random.default_rng(0)
# walls block
doors = np.ones((1, 4), bool)
def move(p, a):
    s = dict(agent=np.array([p]), dist=np.array([[.1, .1]]), dvel=np.zeros((1, 2)), tp=np.array([1000]), doors=doors)
    for _ in range(60): s = env.step(s, np.array([a]), rng)
    return s["agent"][0]
print("into wall (y=.40 -> right):", move([.40, .40], [1, 0]), "(should stop x<0.48)")
print("through door (y=.25 -> right):", move([.40, .25], [1, 0]), "(should pass x>0.5)")
print("into horiz wall (x=.15 up):", move([.15, .40], [0, 1]), "(should stop y<0.48)")
print("through horiz door (x=.25 up):", move([.25, .40], [0, 1]), "(should pass y>0.5)")
# distractor teleports
s = env.reset(1, rng); pos = []
for _ in range(400): s = env.step(s, np.zeros((1, 2)), rng); pos.append(s["dist"][0].copy())
pos = np.array(pos); jumps = (np.linalg.norm(np.diff(pos, axis=0), axis=1) > 0.05).sum()
print("distractor teleports in 400 steps:", jumps, "(expect ~11)")
# grid of random states
s = env.reset(16, rng)
imgs = env.render(torch.tensor(s["agent"], dtype=torch.float32), torch.tensor(s["dist"], dtype=torch.float32)).permute(0, 2, 3, 1).numpy()
fig, ax = plt.subplots(4, 4, figsize=(8, 8))
for a, im in zip(ax.flat, imgs): a.imshow(im); a.axis("off")
plt.tight_layout(); plt.savefig("data/env_grid.png", dpi=80)
# closed-door sampler
d = env.sample_doors(1000, 0.5, rng); print("closed-door min open doors:", d.sum(1).min())
