from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_health.catalog import PROFILES, REGIME_CHANNELS, spec_from_profile
from asset_health.config import IngestConfig
from asset_health.ingest import SchemaError, ingest_asset
from asset_health.models import AssetType, DQAction, DQStatus, MaintenanceEvent

SPEC = spec_from_profile(PROFILES[AssetType.SLURRY_PUMP], "PMP-T", "Test Pump")
N = 24 * 30
T0 = pd.Timestamp("2025-03-01")


def make_raw(n=N, seed=0, interval_s=3600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(T0, periods=n, freq=f"{interval_s}s")
    cols = {"timestamp": ts}
    cols["load_pct"] = 60 + rng.normal(0, 5, n)
    cols["speed_rpm"] = SPEC.nominal_rpm * (0.95 + rng.normal(0, 0.01, n))
    cols["ambient_c"] = 25 + rng.normal(0, 2, n)
    for c in SPEC.channels:
        base = {
            "temp_c": 55.0,
            "current_a": 200.0,
            "power_kw": 180.0,
            "pressure_kpa": 400.0,
            "flow_m3h": 700.0,
        }.get(c, 3.0)
        cols[c] = base * (1 + rng.normal(0, 0.03, n))
    return pd.DataFrame(cols)


def test_clean_data_passes_with_no_masking():
    c = ingest_asset(make_raw(), SPEC)
    assert c.report.overall == DQStatus.PASS
    assert not c.blocked
    assert c.valid.to_numpy().all()
    assert c.report.usable_fraction == pytest.approx(1.0)


def test_grid_is_regular_and_ns_resolution():
    c = ingest_asset(make_raw(), SPEC)
    assert (c.frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert str(c.frame.index.dtype) == "datetime64[ns]"


def test_sentinel_values_are_masked_never_used():
    raw = make_raw()
    raw.loc[100, "vib_rms"] = 9999.0
    raw.loc[101, "temp_c"] = -999.0
    raw.loc[102, "pressure_kpa"] = 65535.0
    c = ingest_asset(raw, SPEC)
    assert np.isnan(c.frame["vib_rms"].iloc[100])
    assert np.isnan(c.frame["temp_c"].iloc[101])
    assert np.isnan(c.frame["pressure_kpa"].iloc[102])
    assert not c.valid["vib_rms"].iloc[100]
    assert c.report.channels["vib_rms"].n_out_of_range == 1
    check = next(k for k in c.report.checks if k.name == "out_of_range")
    assert check.action == DQAction.MASKED and check.status == DQStatus.WARN


def test_stuck_sensor_is_masked_after_the_minimum_run():
    raw = make_raw()
    raw.loc[200:260, "flow_m3h"] = 712.3
    c = ingest_asset(raw, SPEC, IngestConfig(stuck_min_run=8))
    masked = c.frame["flow_m3h"].iloc[200:261].isna().sum()
    assert masked >= 50  # all but the first few samples of the run
    assert c.report.channels["flow_m3h"].n_stuck >= 50
    assert any("flow_m3h" in a and "stuck" in a for a in c.report.sensor_advisories)


def test_short_repeat_is_not_stuck():
    raw = make_raw()
    raw.loc[300:303, "flow_m3h"] = 712.3
    c = ingest_asset(raw, SPEC, IngestConfig(stuck_min_run=8))
    assert c.report.channels["flow_m3h"].n_stuck == 0


def test_frozen_zero_while_stopped_is_not_a_stuck_sensor():
    raw = make_raw()
    raw.loc[400:430, ["load_pct", "speed_rpm"]] = 0.0
    raw.loc[400:430, ["vib_rms", "power_kw", "current_a"]] = 0.0
    c = ingest_asset(raw, SPEC)
    assert c.report.channels["power_kw"].n_stuck == 0


def test_isolated_spike_is_masked():
    raw = make_raw()
    raw.loc[500, "vib_rms"] = raw["vib_rms"].iloc[480:500].median() * 5
    c = ingest_asset(raw, SPEC)
    assert np.isnan(c.frame["vib_rms"].iloc[500])
    assert c.report.channels["vib_rms"].n_spikes >= 1


def test_sustained_step_is_not_a_spike():
    raw = make_raw()
    raw.loc[500:560, "vib_rms"] *= 2.0  # a real change, not a glitch
    c = ingest_asset(raw, SPEC)
    assert c.frame["vib_rms"].iloc[520:560].notna().mean() > 0.9


def test_dropout_becomes_missing_and_is_reported():
    raw = make_raw()
    raw.loc[600:700, "temp_c"] = np.nan
    c = ingest_asset(raw, SPEC)
    assert c.frame["temp_c"].iloc[600:700].isna().all()
    assert any("temp_c" in a and "no data" in a for a in c.report.sensor_advisories)
    assert c.report.channels["temp_c"].dropout_segments


def test_missing_rows_leave_gaps_not_interpolated_values():
    raw = make_raw().drop(index=range(300, 330)).reset_index(drop=True)
    c = ingest_asset(raw, SPEC)
    assert c.report.n_gap_samples == 30
    assert c.report.longest_gap_hours == pytest.approx(30)
    assert not c.present.iloc[300:330].any()
    assert c.frame.iloc[300:330].isna().all().all()


def test_duplicate_rows_are_dropped():
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[10, 11, 12]]]).sort_index(kind="stable").reset_index(drop=True)
    c = ingest_asset(raw, SPEC)
    assert c.report.n_duplicates == 3
    assert len(c.frame) == N


def test_clock_drift_is_estimated_and_the_grid_recovered():
    raw = make_raw()
    drift_s_per_h = 5  # 120 s/day, fast logger
    raw["timestamp"] = raw["timestamp"] + pd.to_timedelta(np.arange(N) * drift_s_per_h, unit="s")
    c = ingest_asset(raw, SPEC)
    assert c.report.clock_drift_s_per_day == pytest.approx(120, abs=6)
    assert len(c.frame) == N  # no phantom gaps from the drifting clock
    assert c.report.n_gap_samples == 0
    assert next(k for k in c.report.checks if k.name == "clock_skew").status == DQStatus.WARN


def test_large_backwards_clock_step_is_survived():
    raw = make_raw()
    raw.loc[400:, "timestamp"] = raw.loc[400:, "timestamp"] - pd.Timedelta(minutes=95)
    c = ingest_asset(raw, SPEC)
    assert c.report.n_backwards >= 0
    assert len(c.frame) >= N - 2
    assert not c.blocked


def test_missing_channel_blocks_the_asset():
    raw = make_raw().drop(columns=["pressure_kpa"])
    c = ingest_asset(raw, SPEC)
    assert c.blocked and c.report.overall == DQStatus.FAIL
    assert c.report.checks[0].action == DQAction.ASSET_BLOCKED
    assert len(c.frame) == 0


def test_too_few_rows_blocks_the_asset():
    c = ingest_asset(make_raw(n=1), SPEC)
    assert c.blocked


def test_no_timestamp_column_is_a_schema_error():
    with pytest.raises(SchemaError):
        ingest_asset(make_raw().drop(columns=["timestamp"]), SPEC)


def test_dead_regime_channel_blocks_the_asset():
    raw = make_raw()
    raw["load_pct"] = np.nan
    c = ingest_asset(raw, SPEC)
    assert c.blocked


def test_mostly_missing_asset_is_blocked_by_the_gap_gate():
    raw = make_raw()
    raw = raw.iloc[np.sort(np.random.default_rng(1).choice(N, size=int(N * 0.5), replace=False))]
    c = ingest_asset(raw.reset_index(drop=True), SPEC)
    assert c.blocked


def test_masks_are_causal_truncating_history_reproduces_them():
    raw = make_raw()
    raw.loc[150:200, "flow_m3h"] = 700.0
    raw.loc[250, "vib_rms"] = 9999.0
    full = ingest_asset(raw, SPEC)
    cut = ingest_asset(raw.iloc[:400], SPEC)
    a = full.valid.iloc[:400].to_numpy()
    b = cut.valid.to_numpy()
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def test_gating_off_lets_sentinels_through_as_an_ablation():
    raw = make_raw()
    raw.loc[100, "vib_rms"] = 9999.0
    c = ingest_asset(raw, SPEC, IngestConfig(gating=False))
    assert c.frame["vib_rms"].iloc[100] == 9999.0


def test_maintenance_events_are_filtered_to_the_asset():
    ev = [
        MaintenanceEvent(asset_id="PMP-T", timestamp=T0.to_pydatetime(), kind="planned_service"),
        MaintenanceEvent(asset_id="OTHER", timestamp=T0.to_pydatetime(), kind="repair"),
    ]
    c = ingest_asset(make_raw(), SPEC, maintenance=ev)
    assert [m.asset_id for m in c.maintenance] == ["PMP-T"]


def test_regime_channels_are_never_spike_masked():
    raw = make_raw()
    raw.loc[500, "load_pct"] = 118.0  # in range, large jump in a regime channel
    c = ingest_asset(raw, SPEC)
    assert c.report.channels["load_pct"].n_spikes == 0
    assert set(REGIME_CHANNELS) <= set(c.report.channels)


def test_report_is_json_serialisable():
    c = ingest_asset(make_raw(), SPEC)
    assert "PMP-T" in c.report.model_dump_json()
