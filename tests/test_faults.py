from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_health.config import DataFaultConfig
from asset_health.simulate.faults import SENTINELS, apply_data_faults

N = 24 * 60


def clean_frame(n=N) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="3600s")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "load_pct": 60 + rng.normal(0, 3, n),
            "speed_rpm": 1400 + rng.normal(0, 5, n),
            "vib_rms": 3 + rng.normal(0, 0.1, n),
            "temp_c": 60 + rng.normal(0, 0.5, n),
        },
        index=idx,
    )


def quiet() -> DataFaultConfig:
    return DataFaultConfig(
        dropout_per_channel_day=0,
        stuck_per_channel_day=0,
        sentinel_prob=0,
        glitch_prob=0,
        missing_row_prob=0,
        comm_outage_per_day=0,
        clock_drift_asset_fraction=0,
    )


def test_zero_rates_leave_the_values_untouched():
    f = clean_frame()
    out, events = apply_data_faults(np.random.default_rng(0), f, "A", quiet(), 3600)
    out = out.drop_duplicates("timestamp")  # historian replays are always on (0.05% of rows)
    np.testing.assert_allclose(out["vib_rms"].to_numpy(), f["vib_rms"].to_numpy())
    assert [e for e in events if e.kind != "duplicate_row"] == []


def test_output_has_a_timestamp_column_in_ns():
    out, _ = apply_data_faults(np.random.default_rng(0), clean_frame(), "A", quiet(), 3600)
    assert out.columns[0] == "timestamp"
    assert str(out["timestamp"].dtype) == "datetime64[ns]"


def test_sentinels_use_known_codes_and_are_logged():
    cfg = quiet().model_copy(update={"sentinel_prob": 0.02})
    out, events = apply_data_faults(np.random.default_rng(1), clean_frame(), "A", cfg, 3600)
    sent = [e for e in events if e.kind == "sentinel"]
    assert len(sent) > 5
    assert set(np.unique(out.to_numpy()[np.isin(out.to_numpy(), SENTINELS)])) <= set(SENTINELS)


def test_dropouts_become_nan_and_match_the_log():
    cfg = quiet().model_copy(update={"dropout_per_channel_day": 0.2})
    out, events = apply_data_faults(np.random.default_rng(2), clean_frame(), "A", cfg, 3600)
    drops = [e for e in events if e.kind == "dropout"]
    assert drops
    assert out["vib_rms"].isna().sum() + out["temp_c"].isna().sum() > 0
    for e in drops:
        assert e.channel in {"load_pct", "speed_rpm", "vib_rms", "temp_c"}


def test_stuck_segments_repeat_one_value():
    cfg = quiet().model_copy(update={"stuck_per_channel_day": 0.2, "stuck_hours": (12.0, 20.0)})
    f = clean_frame()
    out, events = apply_data_faults(np.random.default_rng(3), f, "A", cfg, 3600)
    stuck = [e for e in events if e.kind == "stuck"]
    assert stuck
    e = stuck[0]
    # reported timestamps may shift by clock noise; locate the segment through the clean index
    pos = f.index.get_loc(pd.Timestamp(e.start))
    seg = out[e.channel].iloc[pos : pos + 8]
    assert seg.nunique() == 1


def test_comm_outage_removes_rows():
    cfg = quiet().model_copy(update={"comm_outage_per_day": 0.2, "comm_outage_hours": (5.0, 8.0)})
    out, events = apply_data_faults(np.random.default_rng(4), clean_frame(), "A", cfg, 3600)
    assert any(e.kind == "comm_outage" for e in events)
    assert len(out) < N


def test_clock_drift_shifts_reported_time_progressively():
    cfg = quiet().model_copy(
        update={
            "clock_drift_asset_fraction": 1.0,
            "clock_drift_s_per_day": (100.0, 100.0),
            "clock_steps_max": 0,
        }
    )
    f = clean_frame()
    out, events = apply_data_faults(np.random.default_rng(5), f, "A", cfg, 3600)
    out = out.drop_duplicates("timestamp").reset_index(drop=True)
    assert any(e.kind == "clock_drift" for e in events)
    err = (
        (out["timestamp"].to_numpy() - f.index.to_numpy()[: len(out)])
        .astype("timedelta64[s]")
        .astype(float)
    )
    assert abs(err[-1]) > 50 * 24  # ~100 s/day over ~60 days is thousands of seconds
    assert abs(err[0]) < 10


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fault_injection_is_deterministic(seed):
    cfg = DataFaultConfig()
    a, ea = apply_data_faults(np.random.default_rng(seed), clean_frame(), "A", cfg, 3600)
    b, eb = apply_data_faults(np.random.default_rng(seed), clean_frame(), "A", cfg, 3600)
    pd.testing.assert_frame_equal(a, b)
    assert [e.model_dump() for e in ea] == [e.model_dump() for e in eb]
