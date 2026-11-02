"""Self-contained statistics for the eval harness (spec 1.2 / 3.3): Wilson CI, paired bootstrap on the
success indicator, McNemar. No scipy dependency."""
from __future__ import annotations
import numpy as np


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """Mean(a-b) with 95% CI via paired bootstrap over the shared tuples (a,b are 0/1 indicators)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed); n = len(a); diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return float(a.mean() - b.mean()), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def mcnemar(a: np.ndarray, b: np.ndarray):
    """McNemar exact-ish on paired binary outcomes. Returns (b01, b10, two-sided p via binomial)."""
    a = np.asarray(a, int); b = np.asarray(b, int)
    b01 = int(np.sum((a == 0) & (b == 1)))   # a fail, b succeed
    b10 = int(np.sum((a == 1) & (b == 0)))   # a succeed, b fail
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    # two-sided exact binomial p at 0.5
    from math import comb
    p = 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b01, b10, min(1.0, p)
