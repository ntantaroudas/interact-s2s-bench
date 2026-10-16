"""Statistics over meetings (the unit of analysis, n = 16 AMI eval meetings).

* Friedman test across >= 3 related conditions (policies) per metric.
* Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
* Percentile bootstrap 95 % CIs of the mean across meetings.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np


def bootstrap_ci(x: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def friedman(table: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """``table``: condition -> per-meeting values (same meeting order)."""
    from scipy.stats import friedmanchisquare

    keys = list(table)
    if len(keys) < 3:
        return {"friedman_chi2": float("nan"), "friedman_p": float("nan")}
    stat, p = friedmanchisquare(*[np.asarray(table[k], dtype=float) for k in keys])
    return {"friedman_chi2": float(stat), "friedman_p": float(p)}


def pairwise_wilcoxon(table: Dict[str, Sequence[float]]) -> List[dict]:
    from scipy.stats import wilcoxon

    rows = []
    for a, b in combinations(list(table), 2):
        x, y = np.asarray(table[a], float), np.asarray(table[b], float)
        d = x - y
        if np.allclose(d, 0):
            p = 1.0
            stat = 0.0
        else:
            stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        rows.append({"a": a, "b": b, "median_diff": float(np.median(d)), "W": float(stat), "p": float(p)})
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
    return rows
