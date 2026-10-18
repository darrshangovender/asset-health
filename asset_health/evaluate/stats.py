"""Small statistics helpers (no scipy dependency beyond the standard library)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_sd(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray([v for v in values if v == v], dtype=float)
    if len(a) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": len(a),
    }


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 4000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling seeds, the unit of independence)."""
    a = np.asarray([v for v in values if v == v], dtype=float)
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def quartiles(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    if len(a) == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    q25, med, q75 = np.quantile(a, [0.25, 0.5, 0.75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75)}
