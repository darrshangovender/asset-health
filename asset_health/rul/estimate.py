"""Remaining useful life with an uncertainty interval. Never a bare point estimate."""

from __future__ import annotations

import pandas as pd

from asset_health.catalog import PROFILES
from asset_health.config import RULConfig
from asset_health.models import AssetType, RULEstimate, RULStatus
from asset_health.rul.model import DEFAULT_PRIOR_DAYS, RULModel, feature_vector
from asset_health.rul.trend import TrendFit, fit_trend, trend_rul

# Used only when no trained model is available (trend extrapolation alone, uncalibrated).
DEFAULT_LEVEL_MEAN = 9.0
DEFAULT_LEVEL_SD = 3.0


def estimate_from_fit(
    asset_id: str,
    t: pd.Timestamp,
    fit: TrendFit,
    model: RULModel | None,
    cfg: RULConfig,
    prior_days: float = DEFAULT_PRIOR_DAYS,
) -> RULEstimate:
    base = {"asset_id": asset_id, "as_of": t.to_pydatetime()}
    if not fit.ok:
        return RULEstimate(**base, status=RULStatus.INSUFFICIENT_TREND, note=fit.reason)
    level_mean = model.level_mean if model else DEFAULT_LEVEL_MEAN
    level_sd = model.level_sd if model else DEFAULT_LEVEL_SD
    tr = trend_rul(fit, level_mean, level_sd, cfg)
    if tr is None:
        why = (
            f"health index {fit.hi_now:.1f} is below the degradation threshold {cfg.hi_min:.1f}"
            if fit.hi_now < cfg.hi_min
            else f"health index is not rising convincingly (slope {fit.slope:+.2f}/day)"
        )
        return RULEstimate(
            **base,
            status=RULStatus.NOT_DEGRADING,
            health_index=fit.hi_now,
            failure_threshold=level_mean,
            note=why,
        )
    if model is None:
        p10, p50, p90 = tr
        return RULEstimate(
            **base,
            status=RULStatus.ESTIMATED,
            p10_days=p10,
            p50_days=p50,
            p90_days=p90,
            health_index=fit.hi_now,
            failure_threshold=level_mean,
            method_trend_p50=tr[1],
            note="uncalibrated: trend extrapolation only",
        )
    x = feature_vector(fit, tr[1], model.level_mean, prior_days)
    q = model.predict_days(x, cfg.horizon_days)[0]
    return RULEstimate(
        **base,
        status=RULStatus.ESTIMATED,
        p10_days=float(q[0]),
        p50_days=float(q[1]),
        p90_days=float(q[2]),
        health_index=fit.hi_now,
        failure_threshold=level_mean,
        method_trend_p50=tr[1],
        method_quantile_p50=float(q[1]),
        note=f"conformal margin {model.conformal_margin:.2f} (log scale)",
    )


def estimate_rul(
    asset_id: str,
    hi: pd.Series,
    t: pd.Timestamp,
    model: RULModel | None = None,
    cfg: RULConfig | None = None,
    asset_type: AssetType | None = None,
) -> RULEstimate:
    """RUL of one asset at time ``t`` from its health-index history (rows after ``t`` ignored)."""
    cfg = cfg or RULConfig()
    prior = (
        PROFILES[asset_type].typical_degradation_days
        if asset_type is not None
        else DEFAULT_PRIOR_DAYS
    )
    return estimate_from_fit(asset_id, t, fit_trend(hi, t, cfg), model, cfg, prior)
