"""CUSUM on smoothed regime residuals (one-sided upward, two-sided for channels where a drop is
also a fault, e.g. discharge pressure or power)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_health.catalog import CHANNEL_DIRECTION


def cusum_matrix(
    z: np.ndarray,
    k: float,
    h: float,
    two_sided: np.ndarray,
    reset: np.ndarray | None = None,
    cap: float = 3.0,
) -> np.ndarray:
    """CUSUM of every column of ``z`` (T x C) at once. Returns ``S / h`` (T x C).

    S+ accumulates ``z - k`` and S- accumulates ``-z - k`` (only for ``two_sided`` columns). A NaN
    holds the state: "no trustworthy reading" is neither evidence of a fault nor of health. The
    state is capped at ``cap * h`` (anti-windup) so a long excursion decays in bounded time once
    the signal returns to normal. ``reset`` zeroes the state at flagged samples (planned service).
    """
    t, c = z.shape
    out = np.zeros((t, c))
    s_pos = np.zeros(c)
    s_neg = np.zeros(c)
    top = cap * h
    two = np.asarray(two_sided, dtype=bool)
    for i in range(t):
        if reset is not None and reset[i]:
            s_pos[:] = 0.0
            s_neg[:] = 0.0
        v = z[i]
        ok = v == v  # not NaN
        if ok.any():
            vv = np.where(ok, v, 0.0)
            s_pos = np.where(ok, np.minimum(top, np.maximum(0.0, s_pos + vv - k)), s_pos)
            s_neg = np.where(ok & two, np.minimum(top, np.maximum(0.0, s_neg - vv - k)), s_neg)
        out[i] = np.maximum(s_pos, s_neg) / h
    return out


def cusum_arrays(
    z: np.ndarray,
    k: float,
    h: float,
    two_sided: bool,
    reset: np.ndarray | None = None,
    cap: float = 3.0,
) -> np.ndarray:
    """Single-series CUSUM, ``S / h`` per sample (see :func:`cusum_matrix`)."""
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    return cusum_matrix(z, k, h, np.array([two_sided]), reset, cap)[:, 0]


def cusum_frame(
    z6: pd.DataFrame, k: float, h: float, reset: np.ndarray | None = None
) -> pd.DataFrame:
    """Per-channel normalised CUSUM score (1.0 = decision interval reached)."""
    two = np.array([CHANNEL_DIRECTION.get(name, "up") == "both" for name in z6.columns])
    out = cusum_matrix(z6.to_numpy(dtype=float), k, h, two, reset)
    return pd.DataFrame(out, index=z6.index, columns=z6.columns)
