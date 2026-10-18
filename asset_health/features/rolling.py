"""Causal rolling statistics (every window looks backwards only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_median(s: pd.Series | pd.DataFrame, window: int, min_periods: int):
    return s.rolling(window, min_periods=min_periods).median()


def rolling_slope_per_day(
    s: pd.Series, window: int, min_periods: int, dt_hours: float = 1.0
) -> pd.Series:
    """OLS slope (units per day) of ``s`` against time over a trailing window.

    Computed from rolling moments of the *same* NaN-aligned pairs, so gaps do not bias it.
    """
    t = pd.Series(np.arange(len(s), dtype=float) * dt_hours / 24.0, index=s.index)
    t = t.where(s.notna())
    r = lambda x: x.rolling(window, min_periods=min_periods).mean()  # noqa: E731
    et, ey, ety, et2 = r(t), r(s), r(t * s), r(t * t)
    var = et2 - et * et
    cov = ety - et * ey
    return (cov / var.where(var > 1e-12)).rename(s.name)


def run_hours_true(mask: pd.Series) -> pd.Series:
    """Length (in samples) of the current run of True values, 0 when False. Causal."""
    arr = mask.to_numpy(dtype=bool)
    out = np.zeros(len(arr), dtype=int)
    c = 0
    for i, v in enumerate(arr):
        c = c + 1 if v else 0
        out[i] = c
    return pd.Series(out, index=mask.index)
