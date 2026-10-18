"""Degradation-trend extrapolation: where is the health index heading, and when does it arrive at
the level at which assets of this kind have failed?

The slope is a Theil-Sen estimate over daily samples (robust to the odd bad reading that slipped
past the data-quality gate) with its own confidence interval. The failure level is itself
uncertain (a hard-running machine fails at a different index value than a gentle one), so both are
sampled and the RUL quantiles are read from the draws.

Everything here is causal: ``fit_trend(hi, t)`` reads only ``hi`` at or before ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_health.config import RULConfig

_AGE_FLOOR = 1.0  # index value regarded as "healthy" when measuring how long degradation has run
_HOUR_NS = 3_600_000_000_000
_DAY_NS = 24 * _HOUR_NS


@dataclass(frozen=True)
class TrendFit:
    ok: bool
    reason: str
    hi_now: float = float("nan")
    slope: float = float("nan")  # health-index units per day
    slope_lo: float = float("nan")
    slope_hi: float = float("nan")
    n_points: int = 0
    age_days: float = 0.0  # how long the index has been above the healthy floor


def theil_sen(x: np.ndarray, y: np.ndarray, level: float) -> tuple[float, float, float]:
    """Theil-Sen slope with Sen's (1968) distribution-free confidence interval.

    Written out in numpy: ``scipy.stats.theilslopes`` costs tens of milliseconds per call, which
    dominates when RUL is re-estimated at hundreds of checkpoints.
    """
    n = len(x)
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    slopes = np.sort((y[j] - y[i])[ok] / dx[ok])
    m = len(slopes)
    if m == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(slopes))
    sigma = np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    c = _z_two_sided(level) * sigma
    lo_i = int(np.clip(np.floor((m - c) / 2.0), 0, m - 1))
    hi_i = int(np.clip(np.ceil((m + c) / 2.0), 0, m - 1))
    return med, float(slopes[lo_i]), float(slopes[hi_i])


def _z_two_sided(level: float) -> float:
    from scipy.special import ndtri

    return float(ndtri(0.5 + level / 2.0))


def fit_trend(hi: pd.Series, t: pd.Timestamp, cfg: RULConfig) -> TrendFit:
    """Trend of the health index over the trailing window ending at ``t`` (causal)."""
    ts = hi.index.as_unit("ns").asi8
    vals = hi.to_numpy(dtype=float)
    t_ns = int(pd.Timestamp(t).as_unit("ns").value)
    end = int(np.searchsorted(ts, t_ns, side="right"))
    lo = int(np.searchsorted(ts, t_ns - cfg.max_staleness_h * _HOUR_NS, side="left"))
    recent = vals[lo:end]
    recent = recent[~np.isnan(recent)]
    if len(recent) < 3:
        return TrendFit(False, "no recent trustworthy health-index values")
    hi_now = float(np.median(recent[-12:]))

    xs: list[float] = []
    ys: list[float] = []
    for j in range(cfg.trend_window_days, -1, -1):
        e_ns = t_ns - j * _DAY_NS
        a = int(np.searchsorted(ts, e_ns - 6 * _HOUR_NS, side="left"))
        b = int(np.searchsorted(ts, e_ns, side="right"))
        w = vals[a:b]
        w = w[~np.isnan(w)]
        if len(w) >= 2:
            xs.append(-float(j))
            ys.append(float(np.median(w)))
    if len(xs) < cfg.min_points:
        return TrendFit(False, f"only {len(xs)} daily samples", hi_now=hi_now, n_points=len(xs))
    slope, s_lo, s_hi = theil_sen(np.asarray(xs), np.asarray(ys), cfg.slope_ci)

    past = vals[:end]
    healthy = np.flatnonzero(past < _AGE_FLOOR)
    if len(healthy):
        start = ts[healthy[-1]]
    else:
        valid = np.flatnonzero(~np.isnan(past))
        start = ts[valid[0]] if len(valid) else t_ns
    age = max(0.0, (t_ns - int(start)) / _DAY_NS)
    return TrendFit(True, "", hi_now, float(slope), float(s_lo), float(s_hi), len(xs), age)


_Z = np.random.default_rng(20261014).standard_normal((4096, 2))
_Z90 = 1.2815515655446004


def trend_rul(
    fit: TrendFit, level_mean: float, level_sd: float, cfg: RULConfig
) -> tuple[float, float, float] | None:
    """(p10, p50, p90) days until the index reaches the failure level, or None when the index is
    not rising convincingly."""
    if not fit.ok or fit.hi_now < cfg.hi_min or fit.slope < cfg.min_slope_per_day:
        return None
    z = _Z[: cfg.mc_draws]
    sd_slope = max((fit.slope_hi - fit.slope_lo) / (2.0 * _Z90), 0.05 * fit.slope, 1e-3)
    slope = fit.slope + sd_slope * z[:, 0]
    level = level_mean + level_sd * z[:, 1]
    gap = np.clip(level - fit.hi_now, 0.0, None)
    ok = slope >= cfg.min_slope_per_day
    rul = np.where(ok, gap / np.where(ok, slope, 1.0), cfg.horizon_days)
    rul = np.clip(rul, 0.0, cfg.horizon_days)
    p10, p50, p90 = np.quantile(rul, [0.1, 0.5, 0.9])
    return float(p10), float(p50), float(p90)
