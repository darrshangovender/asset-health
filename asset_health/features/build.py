"""Assemble the per-asset feature set from a cleaned frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asset_health.catalog import PROFILES, SHARE_FEATURES
from asset_health.config import FeatureConfig
from asset_health.features.regime import (
    LINEAR_CHANNELS,
    HealthyReferenceModel,
    from_model_space,
    inside_envelope,
    operating_envelope,
    regime_design,
    to_model_space,
)
from asset_health.features.rolling import rolling_median, rolling_slope_per_day
from asset_health.ingest.checks import CleanAsset
from asset_health.models import AssetSpec


@dataclass
class AssetFeatures:
    spec: AssetSpec
    index: pd.DatetimeIndex
    running: pd.Series
    z: pd.DataFrame
    z6: pd.DataFrame
    z6_all: pd.DataFrame  # smoothed z including the baseline window
    z24: pd.DataFrame
    slope: pd.DataFrame  # z24 change per day, over the trailing slope window
    obs24: pd.DataFrame  # trailing-24h median of observed value, physical units
    exp24: pd.DataFrame  # trailing-24h median of regime-expected value, physical units
    sigma: dict[str, float]
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    segment_start: pd.Timestamp
    n_valid: pd.Series  # monitored channels with a trustworthy z6 at each row
    model: HealthyReferenceModel | None
    ready: bool
    skipped: dict[str, str] = field(default_factory=dict)
    channels: list[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def feature_channels(spec: AssetSpec) -> list[str]:
    monitored = list(PROFILES[spec.asset_type].monitored)
    shares = [SHARE_FEATURES[b] for b in monitored if b in SHARE_FEATURES]
    return [*monitored, *shares]


def running_mask(frame: pd.DataFrame, spec: AssetSpec, cfg: FeatureConfig) -> pd.Series:
    ok = (frame["speed_rpm"] >= cfg.running_speed_frac * spec.nominal_rpm) & (
        frame["load_pct"] >= cfg.min_load_pct
    )
    return ok.fillna(False)


def build_features(
    clean: CleanAsset,
    seg_start: pd.Timestamp | None = None,
    seg_end: pd.Timestamp | None = None,
    cfg: FeatureConfig | None = None,
) -> AssetFeatures:
    cfg = cfg or FeatureConfig()
    spec = clean.spec
    frame = clean.frame
    if seg_start is None:
        # an empty (blocked) frame has no start; any anchor will do because nothing is scored
        seg_start = frame.index[0] if len(frame) else pd.Timestamp("1970-01-01")
    sel = frame.index >= seg_start
    if seg_end is not None:
        sel &= frame.index < seg_end
    frame = frame.loc[sel]
    index = frame.index
    b_start = seg_start + pd.Timedelta(hours=cfg.settle_hours)
    b_end = b_start + pd.Timedelta(days=cfg.baseline_days)
    channels = feature_channels(spec)
    empty = pd.DataFrame(np.nan, index=index, columns=channels)

    def not_ready(reason: str) -> AssetFeatures:
        return AssetFeatures(
            spec=spec,
            index=index,
            running=pd.Series(False, index=index),
            z=empty.copy(),
            z6=empty.copy(),
            z6_all=empty.copy(),
            z24=empty.copy(),
            slope=empty.copy(),
            obs24=empty.copy(),
            exp24=empty.copy(),
            sigma={},
            baseline_start=b_start,
            baseline_end=b_end,
            segment_start=seg_start,
            n_valid=pd.Series(0, index=index),
            model=None,
            ready=False,
            skipped={"*": reason},
            channels=channels,
        )

    if len(index) == 0 or clean.blocked:
        return not_ready("asset blocked by data-quality gate or empty segment")
    if index[-1] < b_end:
        return not_ready("segment shorter than the baseline window")

    running = running_mask(frame, spec, cfg)
    x = regime_design(frame, spec.nominal_rpm)
    if not cfg.regime_normalise:  # ablation: a constant design => residual vs baseline level
        x = x.where(x.isna(), 0.0)
    x = x.copy()
    x.loc[~running] = np.nan
    train_mask = pd.Series((index >= b_start) & (index < b_end), index=index) & running

    # Temperature is modelled as rise above ambient (x coupling), so a baseline learned in winter
    # does not have to extrapolate to a summer afternoon.
    coupling = PROFILES[spec.asset_type].ambient_coupling
    amb = frame["ambient_c"]
    if cfg.regime_normalise:
        lo, hi = operating_envelope(x, train_mask, cfg.envelope_margin)
        x.loc[~inside_envelope(x, lo, hi)] = np.nan
    else:
        coupling = 0.0

    targets: dict[str, pd.Series] = {}
    phys: dict[str, pd.Series] = {}
    for name in channels:
        if name in frame.columns:
            y = frame[name].where(running)
            phys[name] = y
            targets[name] = to_model_space(name, y - coupling * amb if name == "temp_c" else y)
    for band, share in SHARE_FEATURES.items():
        if share in channels and band in frame.columns and "vib_rms" in frame.columns:
            ratio = (frame[band] / (frame["vib_rms"] ** 2)).where(running)
            phys[share] = ratio
            targets[share] = np.log(ratio.where(ratio > 1e-12))
    floor = {
        n: (cfg.sigma_floor_temp_c if n in LINEAR_CHANNELS else cfg.sigma_floor_log)
        for n in targets
    }
    model = HealthyReferenceModel.fit(
        x, targets, train_mask, floor, cfg.min_baseline_rows, cfg.ridge_alpha, cfg.sigma_inflate
    )

    z = pd.DataFrame(np.nan, index=index, columns=channels)
    exp_phys = pd.DataFrame(np.nan, index=index, columns=channels)
    for name in model.models:
        zs, expected = model.residual_z(x, name, targets[name])
        z[name] = zs
        exp_phys[name] = from_model_space(name, expected)
        if name == "temp_c":
            exp_phys[name] = exp_phys[name] + coupling * amb
    z_all = z.copy()  # includes the baseline window (used to train/calibrate the forest)
    learning = index < b_end
    z.loc[learning] = np.nan
    z6_all = rolling_median(z_all, cfg.smooth_h, max(2, cfg.smooth_h // 2))

    obs = pd.DataFrame({n: phys[n] for n in phys}, index=index).reindex(columns=channels)
    z6 = rolling_median(z, cfg.smooth_h, max(2, cfg.smooth_h // 2))
    z24 = rolling_median(z, cfg.med_h, max(4, cfg.med_h // 3))
    slope = pd.DataFrame(
        {n: rolling_slope_per_day(z24[n], cfg.slope_h, cfg.slope_h // 3) for n in channels}
    )
    obs24 = rolling_median(obs, cfg.med_h, max(4, cfg.med_h // 3))
    exp_run = exp_phys.copy()
    exp_run.loc[~running] = np.nan
    exp24 = rolling_median(exp_run, cfg.med_h, max(4, cfg.med_h // 3))
    n_valid = z6.notna().sum(axis=1)
    return AssetFeatures(
        spec=spec,
        index=index,
        running=running,
        z=z,
        z6=z6,
        z6_all=z6_all,
        z24=z24,
        slope=slope,
        obs24=obs24,
        exp24=exp24,
        sigma={n: m.sigma for n, m in model.models.items()},
        baseline_start=b_start,
        baseline_end=b_end,
        segment_start=seg_start,
        n_valid=n_valid,
        model=model,
        ready=bool(model.models),
        skipped=dict(model.skipped),
        channels=channels,
    )
