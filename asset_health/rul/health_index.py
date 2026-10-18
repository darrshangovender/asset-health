"""Health index: one scalar that grows as the asset degrades, whatever the failure mode.

``HI(t)`` is the mean of the ``k`` largest *harmful-direction* regime residuals (24 h medians, in
robust sigma units). It is deliberately mode-agnostic: a bearing spall lights up the race-frequency
and high-frequency bands, impeller wear lights up pressure and vane-pass energy, overheating lights
up temperature; in every case the top few residuals rise. Healthy operation sits near 0 to 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_health.catalog import CHANNEL_DIRECTION


def harmful_residuals(z: pd.DataFrame) -> pd.DataFrame:
    """Residuals oriented so that larger always means worse (>= 0)."""
    out = z.copy()
    for c in out.columns:
        if CHANNEL_DIRECTION.get(c, "up") == "up":
            out[c] = out[c].clip(lower=0.0)
        else:
            out[c] = out[c].abs()
    return out


def health_index(z24: pd.DataFrame, top_k: int = 3, cap: float = 40.0) -> pd.Series:
    """Causal health index on the grid of ``z24``; NaN where fewer than ``top_k`` channels have a
    trustworthy residual (no judgement is better than a judgement from one channel)."""
    if z24.shape[1] == 0 or len(z24) == 0:
        return pd.Series(np.nan, index=z24.index, name="hi")
    adj = harmful_residuals(z24).clip(upper=cap).to_numpy(dtype=float)
    valid = ~np.isnan(adj)
    filled = np.where(valid, adj, -1.0)
    top = -np.sort(-filled, axis=1)[:, :top_k]
    hi = top.mean(axis=1)
    hi[valid.sum(axis=1) < top_k] = np.nan
    return pd.Series(hi, index=z24.index, name="hi")


def top_contributors(z24_row: pd.Series, n: int = 5) -> list[tuple[str, float]]:
    """Channels driving the health index at one instant, largest first."""
    adj = harmful_residuals(z24_row.to_frame().T).iloc[0].dropna()
    adj = adj.sort_values(ascending=False)
    return [(str(k), float(v)) for k, v in adj.head(n).items()]
