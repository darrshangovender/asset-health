"""Regime-normalised residuals: a per-asset healthy-reference model.

A busy shift or a hot afternoon changes vibration, temperature, power and pressure without
anything being wrong. For each channel we fit, on the asset's own healthy baseline window, a
small ridge regression of the channel on the operating regime (load, lagged load, speed,
ambient). The residual (observed - expected) divided by its out-of-fold robust sigma is the
quantity every detector looks at.

This is a data-driven reference model of normal behaviour, not a physics-based digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LINEAR_CHANNELS = {"temp_c"}


def regime_design(frame: pd.DataFrame, nominal_rpm: float) -> pd.DataFrame:
    """Operating-regime regressors: load, lagged load, speed and their low-order products.

    Ambient temperature is deliberately *not* a regressor: a baseline learned in one season
    would have to extrapolate in another. Its physical effect on temperature is handled instead
    by modelling temperature rise above ambient (see ``features.build``).
    """
    load = frame["load_pct"] / 100.0
    speed = frame["speed_rpm"] / nominal_rpm
    load_ew = load.ewm(span=3, ignore_na=True).mean()
    return pd.DataFrame(
        {
            "load": load,
            "load_ew": load_ew,
            "load2": load**2,
            "speed": speed,
            "speed2": speed**2,
            "load_speed": load * speed,
            "load2_speed": load**2 * speed,
        },
        index=frame.index,
    )


def operating_envelope(
    x: pd.DataFrame, train_mask: pd.Series, margin: float = 0.25
) -> tuple[pd.Series, pd.Series]:
    """Per-regressor (lo, hi) the healthy baseline covered, widened by ``margin`` of its span.

    A regression is only trusted inside the regime it was fitted on; outside it the residual is
    undefined (NaN) rather than an extrapolated guess that could raise or hide an alarm.
    """
    base = x.loc[train_mask, ["load", "speed"]].dropna()
    lo = base.quantile(0.01)
    hi = base.quantile(0.99)
    span = (hi - lo).clip(lower=1e-3)
    return lo - margin * span, hi + margin * span


def inside_envelope(x: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> pd.Series:
    ok = pd.Series(True, index=x.index)
    for col in lo.index:
        ok &= (x[col] >= lo[col]) & (x[col] <= hi[col])
    return ok


def to_model_space(name: str, y: pd.Series) -> pd.Series:
    if name in LINEAR_CHANNELS:
        return y
    return np.log(y.where(y > 1e-9))


def from_model_space(name: str, y: pd.Series | np.ndarray):
    return y if name in LINEAR_CHANNELS else np.exp(np.clip(y, -50.0, 50.0))


@dataclass
class ChannelModel:
    name: str
    scaler: StandardScaler
    ridge: Ridge
    sigma: float
    n_train: int

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        ok = x.notna().all(axis=1).to_numpy()
        out = np.full(len(x), np.nan)
        if ok.any():
            out[ok] = self.ridge.predict(self.scaler.transform(x.to_numpy()[ok]))
        return out


@dataclass
class HealthyReferenceModel:
    models: dict[str, ChannelModel] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def fit(
        x: pd.DataFrame,
        targets: dict[str, pd.Series],
        train_mask: pd.Series,
        sigma_floor: dict[str, float],
        min_rows: int = 120,
        alpha: float = 1.0,
        sigma_inflate: float = 1.0,
    ) -> HealthyReferenceModel:
        out = HealthyReferenceModel()
        for name, y in targets.items():
            ok = train_mask & x.notna().all(axis=1) & y.notna()
            if int(ok.sum()) < min_rows:
                out.skipped[name] = f"only {int(ok.sum())} usable baseline rows"
                continue
            xs, ys = x.loc[ok].to_numpy(), y.loc[ok].to_numpy()
            scaler = StandardScaler().fit(xs)
            ridge = Ridge(alpha=alpha).fit(scaler.transform(xs), ys)
            # out-of-fold residuals (two contiguous blocks) give an honest sigma
            half = len(ys) // 2
            res = np.empty(len(ys))
            for tr, te in (
                (slice(0, half), slice(half, None)),
                (slice(half, None), slice(0, half)),
            ):
                sc = StandardScaler().fit(xs[tr])
                rg = Ridge(alpha=alpha).fit(sc.transform(xs[tr]), ys[tr])
                res[te] = ys[te] - rg.predict(sc.transform(xs[te]))
            # robust spread of held-out residuals, inflated: a 3-week window cannot show the
            # slower (seasonal, campaign) variation the asset will have later
            sigma = sigma_inflate * 1.4826 * float(np.median(np.abs(res - np.median(res))))
            sigma = max(sigma, sigma_floor.get(name, 0.0), 1e-6)
            out.models[name] = ChannelModel(name, scaler, ridge, sigma, int(ok.sum()))
        return out

    def residual_z(self, x: pd.DataFrame, name: str, y: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Return (z, expected) for one channel in model space."""
        m = self.models[name]
        expected = pd.Series(m.predict(x), index=x.index)
        return (y - expected) / m.sigma, expected
