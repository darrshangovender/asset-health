from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from asset_health.ingest import (
    CsvSource,
    FrameSource,
    ParquetSource,
    TelemetrySource,
    open_source,
)
from asset_health.simulate import write_dataset


@pytest.fixture(scope="module")
def csv_dir(tiny_fleet, tmp_path_factory):
    return write_dataset(tiny_fleet, tmp_path_factory.mktemp("csv") / "ds", "csv")


@pytest.fixture(scope="module")
def parquet_dir(tiny_fleet, tmp_path_factory):
    pytest.importorskip("pyarrow")
    return write_dataset(tiny_fleet, tmp_path_factory.mktemp("pq") / "ds", "parquet")


def test_csv_source_satisfies_the_protocol(csv_dir):
    assert isinstance(CsvSource(csv_dir), TelemetrySource)


def test_frame_source_satisfies_the_protocol(tiny_fleet):
    assert isinstance(FrameSource(tiny_fleet.specs, tiny_fleet.raw), TelemetrySource)


def test_csv_round_trip_preserves_assets_and_columns(csv_dir, tiny_fleet):
    src = CsvSource(csv_dir)
    assert [s.asset_id for s in src.assets()] == [s.asset_id for s in tiny_fleet.specs]
    spec = tiny_fleet.specs[0]
    df = src.read(spec.asset_id)
    assert {"timestamp", "vib_rms", "load_pct"} <= set(df.columns)
    assert len(df) == len(tiny_fleet.raw[spec.asset_id])


def test_csv_values_survive_the_round_trip(csv_dir, tiny_fleet):
    spec = tiny_fleet.specs[0]
    got = CsvSource(csv_dir).read(spec.asset_id)["vib_rms"].to_numpy()
    want = tiny_fleet.raw[spec.asset_id]["vib_rms"].to_numpy()
    assert (
        pd.Series(got).equals(pd.Series(want)) or abs(pd.Series(got) - pd.Series(want)).max() < 1e-9
    )


def test_parquet_round_trip(parquet_dir, tiny_fleet):
    src = ParquetSource(parquet_dir)
    spec = tiny_fleet.specs[0]
    df = src.read(spec.asset_id)
    assert len(df) == len(tiny_fleet.raw[spec.asset_id])


def test_open_source_picks_the_format_on_disk(csv_dir, parquet_dir):
    assert isinstance(open_source(csv_dir), CsvSource)
    assert isinstance(open_source(parquet_dir), ParquetSource)


def test_read_slices_by_time(csv_dir, tiny_fleet):
    spec = tiny_fleet.specs[0]
    start = datetime(2025, 1, 10)
    end = datetime(2025, 1, 20)
    df = CsvSource(csv_dir).read(spec.asset_id, start, end)
    ts = pd.to_datetime(df["timestamp"])
    assert ts.min() >= pd.Timestamp(start) and ts.max() < pd.Timestamp(end)
    assert 0 < len(df) <= 10 * 24 + 5


def test_unknown_asset_raises_keyerror(csv_dir):
    with pytest.raises(KeyError):
        CsvSource(csv_dir).read("NOPE")


def test_non_dataset_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CsvSource(tmp_path)


def test_maintenance_events_filter_by_asset(csv_dir, tiny_fleet):
    src = CsvSource(csv_dir)
    all_events = src.maintenance_events()
    spec = tiny_fleet.specs[0]
    mine = src.maintenance_events(spec.asset_id)
    assert all(e.asset_id == spec.asset_id for e in mine)
    assert len(mine) <= len(all_events)


def test_frame_source_filters_and_slices(tiny_fleet):
    src = FrameSource(tiny_fleet.specs, tiny_fleet.raw, tiny_fleet.maintenance)
    spec = tiny_fleet.specs[0]
    df = src.read(spec.asset_id, end=datetime(2025, 1, 5))
    assert pd.to_datetime(df["timestamp"]).max() < pd.Timestamp("2025-01-05")
    assert all(e.asset_id == spec.asset_id for e in src.maintenance_events(spec.asset_id))
