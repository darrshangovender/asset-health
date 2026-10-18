"""Combine CUSUM, Isolation Forest and the rule layer into one explainable decision stream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_health.config import DetectConfig, FeatureConfig
from asset_health.detect.cusum import cusum_frame
from asset_health.detect.iforest import ForestScorer, fit_forest, one_sided
from asset_health.detect.rules import best_hypothesis, rule_strengths
from asset_health.features.build import AssetFeatures
from asset_health.models import FaultHypothesis


@dataclass
class Detection:
    frame: pd.DataFrame  # cusum, iforest, rule, fused, n_firing, decision_valid, hold, hypothesis
    cusum_channels: pd.DataFrame
    rule_strengths: pd.DataFrame
    forest: ForestScorer | None
    ready: bool


def hold_mask(
    index: pd.DatetimeIndex, service_times: list[pd.Timestamp], hours: int
) -> tuple[np.ndarray, np.ndarray]:
    """(hold, reset): hold is True for ``hours`` after a planned service; reset marks the
    service sample itself so CUSUM state restarts."""
    hold = np.zeros(len(index), dtype=bool)
    reset = np.zeros(len(index), dtype=bool)
    for t in service_times:
        i = int(index.searchsorted(t))
        if i < len(index):
            reset[i] = True
            hold[i : i + hours + 1] = True
    return hold, reset


def detect_asset(
    feats: AssetFeatures,
    service_times: list[pd.Timestamp] | None = None,
    cfg: DetectConfig | None = None,
    fcfg: FeatureConfig | None = None,
) -> Detection:
    cfg = cfg or DetectConfig()
    fcfg = fcfg or FeatureConfig()
    idx = feats.index
    empty = pd.DataFrame(
        {
            "cusum": 0.0,
            "iforest": 0.0,
            "rule": 0.0,
            "fused": np.nan,
            "n_firing": 0,
            "decision_valid": False,
            "hold": False,
            "hypothesis": FaultHypothesis.UNEXPLAINED.value,
            "rule_name": "",
        },
        index=idx,
    )
    if not feats.ready:
        return Detection(empty, pd.DataFrame(index=idx), pd.DataFrame(index=idx), None, False)

    hold, reset = hold_mask(idx, service_times or [], fcfg.hold_hours_after_service)
    z6 = feats.z6.copy()
    z6.loc[hold] = np.nan  # service run-in transients are not evidence
    cus_ch = cusum_frame(z6, cfg.cusum_k, cfg.cusum_h, reset)
    cusum = cus_ch.max(axis=1)

    train = one_sided(feats.z6_all).loc[
        (idx >= feats.baseline_start) & (idx < feats.baseline_end) & feats.running.to_numpy()
    ]
    forest = fit_forest(train, cfg.if_trees, cfg.if_quantile, cfg.if_margin, cfg.random_state)
    if forest is not None:
        iso = forest.score(one_sided(z6)).rolling(6, min_periods=3).median().fillna(0.0)
    else:
        iso = pd.Series(0.0, index=idx)

    z24 = feats.z24.copy()
    z24.loc[hold] = np.nan
    strengths = rule_strengths(z24, feats.spec.asset_type)
    rule, hyp, rule_name = best_hypothesis(strengths)

    n_ch = max(1, feats.n_channels)
    valid = (
        (idx >= feats.baseline_end)
        & ~hold
        & (feats.n_valid.to_numpy() >= cfg.min_valid_fraction * n_ch)
    )
    c = cfg.clip
    fused = (
        cfg.weight_cusum * cusum.clip(upper=c)
        + cfg.weight_if * iso.clip(upper=c)
        + cfg.weight_rule * rule.clip(upper=c)
    )
    n_firing = (cusum >= 1).astype(int) + (iso >= 1).astype(int) + (rule >= 1).astype(int)
    frame = pd.DataFrame(
        {
            "cusum": cusum,
            "iforest": iso,
            "rule": rule,
            "fused": fused.where(valid),
            "n_firing": n_firing.where(valid, 0),
            "decision_valid": valid,
            "hold": hold,
            "hypothesis": hyp.where(rule > 0, FaultHypothesis.UNEXPLAINED.value),
            "rule_name": rule_name,
        },
        index=idx,
    )
    pre = idx < feats.baseline_end
    frame.loc[pre, ["cusum", "iforest", "rule"]] = 0.0
    return Detection(frame, cus_ch, strengths, forest, True)
