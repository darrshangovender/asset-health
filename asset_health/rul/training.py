"""Build RUL training and calibration sets from run-to-failure episodes and fit the model.

The seeds that go in here must never be evaluation seeds; ``benchmarks/run.py`` asserts that the
training, calibration and evaluation seed sets are pairwise disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_health.catalog import PROFILES
from asset_health.config import RULConfig
from asset_health.models import AssetType
from asset_health.rul.model import (
    FEATURES,
    RULModel,
    conformal_margin,
    feature_vector,
    fit_quantile_model,
)
from asset_health.rul.trend import fit_trend, trend_rul


@dataclass
class RunToFailure:
    """One failing asset: its causal health-index series and the (ground-truth) failure time."""

    seed: int
    asset_id: str
    hi: pd.Series
    failure: pd.Timestamp
    asset_type: AssetType = AssetType.SLURRY_PUMP


def failure_level(ep: RunToFailure) -> float | None:
    w = ep.hi.loc[ep.failure - pd.Timedelta(hours=36) : ep.failure - pd.Timedelta(hours=6)]
    w = w.dropna()
    return float(w.median()) if len(w) >= 3 else None


def level_stats(episodes: list[RunToFailure]) -> tuple[float, float]:
    levels = [v for v in (failure_level(e) for e in episodes) if v is not None]
    if len(levels) < 3:
        raise ValueError("need at least 3 run-to-failure episodes to learn a failure level")
    return float(np.mean(levels)), float(max(np.std(levels, ddof=1), 0.5))


def collect_rows(
    ep: RunToFailure,
    cfg: RULConfig,
    level_mean: float,
    level_sd: float,
    step_days: float = 1.0,
    offset_days: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Features and log1p(true RUL) at daily checkpoints counted back from the failure."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    first = ep.hi.dropna().index[0] if ep.hi.notna().any() else ep.failure
    prior = PROFILES[ep.asset_type].typical_degradation_days
    j = 0
    while True:
        t = ep.failure - pd.Timedelta(days=offset_days + j * step_days)
        if t < first + pd.Timedelta(days=cfg.trend_window_days) or j > cfg.horizon_days:
            break
        fit = fit_trend(ep.hi, t, cfg)
        tr = trend_rul(fit, level_mean, level_sd, cfg)
        if tr is not None:
            xs.append(feature_vector(fit, tr[1], level_mean, prior))
            ys.append(float(np.log1p((ep.failure - t).total_seconds() / 86400.0)))
        j += 1
    if not xs:
        return np.empty((0, len(FEATURES))), np.empty(0)
    return np.vstack(xs), np.asarray(ys)


def _stack(
    episodes: list[RunToFailure], cfg: RULConfig, lm: float, ls: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xs, ys, ws = [], [], []
    used = 0
    for ep in episodes:
        x, y = collect_rows(ep, cfg, lm, ls)
        if len(y) == 0:
            continue
        used += 1
        xs.append(x)
        ys.append(y)
        ws.append(np.full(len(y), 1.0 / len(y)))
    if not xs:
        raise ValueError("no usable RUL rows")
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws), used


def fit_rul_model(
    train: list[RunToFailure],
    calibration: list[RunToFailure],
    cfg: RULConfig | None = None,
) -> RULModel:
    """Fit on ``train`` episodes, then set the conformal margin on ``calibration`` episodes."""
    cfg = cfg or RULConfig()
    train_seeds = sorted({e.seed for e in train})
    cal_seeds = sorted({e.seed for e in calibration})
    if set(train_seeds) & set(cal_seeds):
        raise ValueError("training and calibration seeds must be disjoint")
    lm, ls = level_stats(train)
    x, y, w, n_ep = _stack(train, cfg, lm, ls)
    model = fit_quantile_model(x, y, w, cfg, lm, ls)
    xc, yc, wc, n_cal = _stack(calibration, cfg, lm, ls)
    margin = conformal_margin(model, xc, yc, wc, cfg.target_coverage)
    return model.model_copy(
        update={
            "conformal_margin": margin,
            "train_seeds": train_seeds,
            "cal_seeds": cal_seeds,
            "n_train_events": n_ep,
            "n_train_rows": len(y),
            "n_cal_events": n_cal,
        }
    )
