"""Isolation Forest on the multivariate residual vector, trained on the asset's own baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from asset_health.catalog import CHANNEL_DIRECTION


def one_sided(z: pd.DataFrame) -> pd.DataFrame:
    """Keep only the harmful direction of each residual.

    A vibration level that falls after a lubrication round is not an anomaly worth scoring, so
    "up" channels are floored at zero; "both" channels (power, pressure, flow ...) are unchanged.
    """
    out = z.copy()
    for c in out.columns:
        if CHANNEL_DIRECTION.get(c, "up") == "up":
            out[c] = out[c].clip(lower=0.0)
    return out


def _forest(n: int, trees: int, seed: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=trees,
        max_samples=min(256, max(16, n)),
        contamination="auto",
        random_state=seed,
    )


@dataclass
class ForestScorer:
    model: IsolationForest
    centre: float
    threshold: float
    columns: list[str]

    def score(self, z6: pd.DataFrame) -> pd.Series:
        """Normalised score: 0 at the baseline median, 1 at the calibrated threshold."""
        x = z6[self.columns]
        ok = x.notna().sum(axis=1) >= max(2, len(self.columns) // 3)
        raw = pd.Series(np.nan, index=z6.index)
        if ok.any():
            filled = x.loc[ok].fillna(0.0).to_numpy()
            raw.loc[ok] = -self.model.score_samples(filled)
        norm = (raw - self.centre) / max(self.threshold - self.centre, 1e-6)
        return norm.clip(lower=0.0)


def fit_forest(
    z6_train: pd.DataFrame, trees: int, quantile: float, margin: float, seed: int
) -> ForestScorer | None:
    """Fit on baseline rows; calibrate the alarm threshold on 2-fold out-of-fold scores so the
    threshold is not set from the very points the trees were grown on."""
    cols = [c for c in z6_train.columns if z6_train[c].notna().mean() > 0.5]
    x = z6_train[cols]
    x = x.loc[x.notna().sum(axis=1) >= max(2, len(cols) // 3)].fillna(0.0)
    if len(cols) < 2 or len(x) < 60:
        return None
    arr = x.to_numpy()
    folds = np.array_split(np.arange(len(arr)), 2)
    oof = np.empty(len(arr))
    for f in folds:
        tr = np.setdiff1d(np.arange(len(arr)), f)
        m = _forest(len(tr), trees, seed).fit(arr[tr])
        oof[f] = -m.score_samples(arr[f])
    centre = float(np.median(oof))
    threshold = centre + margin * (float(np.quantile(oof, quantile)) - centre)
    model = _forest(len(arr), trees, seed).fit(arr)
    return ForestScorer(model, centre, threshold, cols)
