"""Quantile-regression RUL model with split-conformal widening, serialisable to JSON.

Stage 1 (``trend``) extrapolates the health index to the failure level. Stage 2 (this module)
learns, from run-to-failure episodes on *training* seeds, how the trend-extrapolated RUL, the
current index, its slope and the age of the degradation map onto the quantiles of the true RUL
(log1p scale). Stage 3 widens (or narrows) the interval by a margin chosen on *separate
calibration seeds* so that its empirical coverage hits the target (split-conformal quantile
regression). Evaluation seeds are used for neither.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import QuantileRegressor

import asset_health
from asset_health.config import RULConfig
from asset_health.rul.trend import TrendFit

FEATURES = (
    "log1p_trend_p50",
    "hi_now",
    "slope",
    "log1p_age",
    "log1p_gap_to_level",
    "log1p_prior_remaining",
    "log_prior_days",
)

# Used when the asset class is unknown: roughly the fleet-average P-F interval, in days.
DEFAULT_PRIOR_DAYS = 40.0


def feature_vector(
    fit: TrendFit, trend_p50: float, level_mean: float, prior_days: float = DEFAULT_PRIOR_DAYS
) -> np.ndarray:
    """Model inputs at one checkpoint. ``prior_days`` is the asset class's typical P-F interval
    (what a reliability study records); it anchors the estimate when the trend is still short."""
    age = max(fit.age_days, 0.0)
    return np.array(
        [
            np.log1p(trend_p50),
            fit.hi_now,
            fit.slope,
            np.log1p(age),
            np.log1p(max(level_mean - fit.hi_now, 0.0)),
            np.log1p(max(prior_days - age, 1.0)),
            np.log(prior_days),
        ],
        dtype=float,
    )


class RULModel(BaseModel):
    """Learned RUL quantile model (SYNTHETIC training data)."""

    model_config = ConfigDict(extra="forbid")

    synthetic: bool = True
    version: str = asset_health.__version__
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURES))
    mean: list[float]
    scale: list[float]
    quantiles: list[float]
    coef: list[list[float]]
    intercept: list[float]
    conformal_margin: float = 0.0
    level_mean: float
    level_sd: float
    train_seeds: list[int] = Field(default_factory=list)
    cal_seeds: list[int] = Field(default_factory=list)
    n_train_events: int = 0
    n_train_rows: int = 0
    n_cal_events: int = 0
    target_coverage: float = 0.8

    def predict_log(self, x: np.ndarray) -> np.ndarray:
        """Raw quantiles on the log1p(days) scale, shape (n, 3), sorted and margin-adjusted."""
        x = np.atleast_2d(x)
        z = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        q = z @ np.asarray(self.coef).T + np.asarray(self.intercept)
        q = np.sort(q, axis=1)
        q[:, 0] -= self.conformal_margin
        q[:, 2] += self.conformal_margin
        q[:, 0] = np.minimum(q[:, 0], q[:, 1])
        q[:, 2] = np.maximum(q[:, 2], q[:, 1])
        return q

    def predict_days(self, x: np.ndarray, horizon: float) -> np.ndarray:
        days = np.expm1(self.predict_log(x))
        return np.clip(days, 0.0, horizon)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RULModel:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def fit_quantile_model(
    x: np.ndarray,
    y_log: np.ndarray,
    weights: np.ndarray,
    cfg: RULConfig,
    level_mean: float,
    level_sd: float,
) -> RULModel:
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    z = (x - mean) / scale
    coefs, icpts = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # highs prints convergence notes on tiny problems
        for q in cfg.quantiles:
            reg = QuantileRegressor(quantile=q, alpha=1e-3, solver="highs")
            reg.fit(z, y_log, sample_weight=weights)
            coefs.append([float(c) for c in reg.coef_])
            icpts.append(float(reg.intercept_))
    return RULModel(
        mean=[float(m) for m in mean],
        scale=[float(s) for s in scale],
        quantiles=list(cfg.quantiles),
        coef=coefs,
        intercept=icpts,
        level_mean=level_mean,
        level_sd=level_sd,
        target_coverage=cfg.target_coverage,
    )


def conformal_margin(
    model: RULModel, x: np.ndarray, y_log: np.ndarray, weights: np.ndarray, coverage: float
) -> float:
    """Weighted split-conformal margin (log scale) so ~``coverage`` of calibration points fall
    inside ``[q_lo - m, q_hi + m]``. Weights give each episode equal say."""
    base = model.model_copy(update={"conformal_margin": 0.0}).predict_log(x)
    score = np.maximum(base[:, 0] - y_log, y_log - base[:, 2])
    order = np.argsort(score)
    w = weights[order] / weights.sum()
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, min(coverage + 1.0 / max(len(score), 1), 1.0)))
    return float(score[order][min(idx, len(score) - 1)])
