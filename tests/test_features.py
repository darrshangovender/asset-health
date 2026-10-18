from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_health.catalog import PROFILES, spec_from_profile
from asset_health.config import FeatureConfig, IngestConfig
from asset_health.features import HealthyReferenceModel, build_features, feature_channels
from asset_health.features.regime import (
    inside_envelope,
    operating_envelope,
    regime_design,
    to_model_space,
)
from asset_health.features.rolling import rolling_slope_per_day, run_hours_true
from asset_health.ingest import ingest_asset
from asset_health.models import AssetType

PUMP = spec_from_profile(PROFILES[AssetType.SLURRY_PUMP], "PMP-F", "Feature Pump")
ENGINE = spec_from_profile(PROFILES[AssetType.HAUL_TRUCK_ENGINE], "ENG-F", "Feature Engine")
N = 24 * 70


def synth(spec, n=N, seed=0, drift=None, season=0.0, noise=0.01) -> pd.DataFrame:
    """A healthy asset whose channels follow a known regime law (so z should be ~N(0,1))."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="3600s")
    load = np.clip(0.65 + 0.2 * np.sin(np.arange(n) / 17.0) + rng.normal(0, 0.05, n), 0.15, 1.0)
    speed = 0.9 + 0.1 * load + rng.normal(0, 0.002, n)
    ambient = (
        22
        + season * np.arange(n) / n
        + 6 * np.sin(2 * np.pi * np.arange(n) / 24)
        + rng.normal(0, 0.3, n)
    )
    cols = {
        "timestamp": ts,
        "load_pct": load * 100,
        "speed_rpm": spec.nominal_rpm * speed,
        "ambient_c": ambient,
    }
    vib = 2.0 * (0.4 + 0.6 * load) ** 1.5 * (1 + rng.normal(0, noise, n))
    for c in spec.channels:
        if c == "vib_rms":
            cols[c] = vib
        elif c.startswith("vib_band"):
            cols[c] = 0.1 * vib**2 * (1 + rng.normal(0, 2 * noise, n))
        elif c == "temp_c":
            cols[c] = ambient + 12 + 15 * load**1.5 + rng.normal(0, 0.4, n)
        elif c == "power_kw":
            cols[c] = 200 * load * speed * (1 + rng.normal(0, noise, n))
        elif c == "current_a":
            cols[c] = 250 * load * (1 + rng.normal(0, noise, n))
        elif c == "pressure_kpa":
            cols[c] = 450 * speed**2 * (1 + rng.normal(0, noise, n))
        elif c == "flow_m3h":
            cols[c] = 800 * load * speed * (1 + rng.normal(0, noise, n))
    df = pd.DataFrame(cols)
    if drift:
        for ch, f in drift.items():
            df[ch] = df[ch] * f
    return df


def clean(spec, **kw):
    return ingest_asset(synth(spec, **kw), spec, IngestConfig())


def test_regime_design_has_no_ambient_regressor():
    x = regime_design(synth(PUMP), PUMP.nominal_rpm)
    assert "ambient" not in x.columns
    assert {"load", "speed", "load_speed"} <= set(x.columns)


def test_feature_channels_add_share_features():
    ch = feature_channels(PUMP)
    assert "bpfo_share" in ch and "hf_share" in ch and "vib_rms" in ch


def test_healthy_residuals_are_standardised():
    f = build_features(clean(PUMP))
    assert f.ready
    z = f.z.stack().dropna()
    assert abs(z.median()) < 0.6
    assert 0.2 < z.std() < 3.0


def test_a_busy_shift_is_not_a_fault():
    """Loading the machine harder moves vibration and power; the residual must stay small."""
    df = synth(PUMP)
    df.loc[df.index[-300:], "load_pct"] = np.clip(df["load_pct"].iloc[-300:] + 15, 0, 100)
    load = df["load_pct"] / 100
    speed = df["speed_rpm"] / PUMP.nominal_rpm
    vib = 2.0 * (0.4 + 0.6 * load) ** 1.5
    df["vib_rms"] = vib * (1 + np.random.default_rng(1).normal(0, 0.01, len(df)))
    df["power_kw"] = 200 * load * speed
    f = build_features(ingest_asset(df, PUMP))
    tail = f.z6[["vib_rms", "power_kw"]].iloc[-250:].abs().median()
    assert (tail < 2.5).all()


def test_a_real_degradation_is_flagged():
    df = synth(PUMP)
    n = len(df)
    ramp = np.linspace(0, 1, 400)
    df.loc[df.index[-400:], "vib_rms"] = df["vib_rms"].iloc[-400:] * (1 + 1.5 * ramp**2)
    f = build_features(ingest_asset(df, PUMP))
    assert f.z24["vib_rms"].iloc[-1] > 8
    assert f.z24["vib_rms"].iloc[: n - 400].abs().max() < 6


def test_temperature_is_modelled_as_rise_above_ambient():
    """A seasonal ambient drift must not look like a thermal fault."""
    f = build_features(clean(PUMP, season=12.0))
    tail = f.z6["temp_c"].iloc[-200:]
    assert abs(tail.median()) < 2.5


def test_engine_temperature_is_not_coupled_to_ambient():
    df = synth(ENGINE)
    df["temp_c"] = (
        88 + 5 * (df["load_pct"] / 100) + np.random.default_rng(0).normal(0, 0.4, len(df))
    )
    f = build_features(ingest_asset(df, ENGINE))
    assert abs(f.z6["temp_c"].dropna().median()) < 2.0


def test_baseline_window_is_never_scored():
    f = build_features(clean(PUMP))
    assert f.z.loc[f.index < f.baseline_end].isna().all().all()
    assert f.n_valid.loc[f.index < f.baseline_end].eq(0).all()


def test_segment_shorter_than_baseline_is_not_ready():
    df = synth(PUMP, n=24 * 15)
    c = ingest_asset(df, PUMP)
    f = build_features(c)
    assert not f.ready and "baseline" in f.skipped["*"]
    assert f.z.isna().all().all()


def test_blocked_asset_yields_not_ready_features():
    df = synth(PUMP).drop(columns=["pressure_kpa"])
    f = build_features(ingest_asset(df, PUMP))
    assert not f.ready


def test_features_are_causal():
    df = synth(PUMP)
    df.loc[df.index[-300:], "vib_rms"] *= 3
    full = build_features(ingest_asset(df, PUMP))
    cut = build_features(ingest_asset(df.iloc[:1400], PUMP))
    a = full.z24.iloc[:1380]
    b = cut.z24.iloc[:1380]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, atol=1e-9)


def test_regime_normalisation_ablation_increases_residual_spread():
    c = clean(PUMP)
    on = build_features(c, cfg=FeatureConfig())
    off = build_features(c, cfg=FeatureConfig(regime_normalise=False))
    assert off.sigma["vib_rms"] > on.sigma["vib_rms"]


def test_operating_envelope_marks_unseen_regimes():
    df = synth(PUMP)
    x = regime_design(df.assign(timestamp=df["timestamp"]).set_index("timestamp"), PUMP.nominal_rpm)
    mask = pd.Series(np.arange(len(x)) < 500, index=x.index)
    lo, hi = operating_envelope(x, mask, 0.1)
    inside = inside_envelope(x, lo, hi)
    assert inside.iloc[:500].mean() > 0.99
    x2 = x.copy()
    x2["load"] = 5.0  # far outside anything the baseline saw
    assert not inside_envelope(x2, lo, hi).any()


def test_reference_model_recovers_a_linear_law():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600)})
    y = 3 * x["a"] - 2 * x["b"] + rng.normal(0, 0.1, 600)
    m = HealthyReferenceModel.fit(x, {"y": y}, pd.Series(True, index=x.index), {"y": 0.01})
    z, expected = m.residual_z(x, "y", y)
    assert np.corrcoef(expected, y)[0, 1] > 0.99
    assert z.std() < 1.5


def test_reference_model_skips_channels_with_too_little_baseline():
    x = pd.DataFrame({"a": np.arange(50.0)})
    m = HealthyReferenceModel.fit(
        x, {"y": x["a"]}, pd.Series(True, index=x.index), {}, min_rows=120
    )
    assert "y" in m.skipped and not m.models


def test_to_model_space_logs_positive_channels_only():
    s = pd.Series([1.0, np.e, -3.0, 0.0])
    out = to_model_space("vib_rms", s)
    assert out.iloc[1] == pytest.approx(1.0)
    assert np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])
    assert to_model_space("temp_c", s).equals(s)


def test_rolling_slope_matches_numpy_polyfit():
    s = pd.Series(
        2.5 * np.arange(200) / 24.0 + 3, index=pd.date_range("2025-01-01", periods=200, freq="h")
    )
    sl = rolling_slope_per_day(s, 72, 24)
    assert sl.iloc[-1] == pytest.approx(2.5, rel=1e-6)


def test_rolling_slope_tolerates_gaps():
    s = pd.Series(
        1.0 * np.arange(300) / 24.0, index=pd.date_range("2025-01-01", periods=300, freq="h")
    )
    s.iloc[100:130] = np.nan
    sl = rolling_slope_per_day(s, 120, 40)
    assert sl.iloc[-1] == pytest.approx(1.0, rel=1e-6)


def test_run_hours_true_counts_consecutive_trues():
    m = pd.Series([True, True, False, True, True, True])
    assert run_hours_true(m).tolist() == [1, 2, 0, 1, 2, 3]
