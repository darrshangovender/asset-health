"""Telemetry source interface and the file adapters that are actually implemented.

``TelemetrySource`` is the seam between this package and a plant. Implemented here:

* ``CsvSource`` / ``ParquetSource``: a directory written by ``asset-health simulate`` or any
  export with the same layout (``manifest.json``, ``maintenance.json``, ``assets/<id>.csv``).
* ``FrameSource``: in-memory frames (used by the pipeline and the tests).

NOT implemented: OPC UA, MQTT, or a vendor historian (PI, Ignition, Wonderware). See
``docs/ADAPTERS.md`` for how one would implement this Protocol; no such adapter ships here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from asset_health.models import AssetSpec, FleetManifest, MaintenanceEvent
from asset_health.timeutil import to_utc_naive, to_utc_naive_series


@runtime_checkable
class TelemetrySource(Protocol):
    """Read-only access to asset registers, telemetry and the maintenance log."""

    def assets(self) -> list[AssetSpec]: ...

    def read(
        self, asset_id: str, start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        """Return a frame with a ``timestamp`` column (UTC) and one column per channel."""
        ...

    def maintenance_events(self, asset_id: str | None = None) -> list[MaintenanceEvent]: ...


def _slice(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    """Rows in ``[start, end)``. Timestamps are compared as naive UTC (offsets converted)."""
    if start is None and end is None:
        return df.reset_index(drop=True)
    ts = to_utc_naive_series(df["timestamp"])
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= ts >= to_utc_naive(start)
    if end is not None:
        mask &= ts < to_utc_naive(end)
    return df.loc[mask].reset_index(drop=True)


class _DirectorySource:
    suffix = ""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest = self.path / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} not found; is this an asset-health dataset?")
        self.manifest = FleetManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
        mpath = self.path / "maintenance.json"
        self._maintenance = (
            [MaintenanceEvent.model_validate(m) for m in json.loads(mpath.read_text("utf-8"))]
            if mpath.exists()
            else []
        )

    def assets(self) -> list[AssetSpec]:
        return list(self.manifest.assets)

    def maintenance_events(self, asset_id: str | None = None) -> list[MaintenanceEvent]:
        return [m for m in self._maintenance if asset_id is None or m.asset_id == asset_id]

    def _load(self, file: Path) -> pd.DataFrame:  # pragma: no cover - overridden
        raise NotImplementedError

    def read(
        self, asset_id: str, start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        file = self.path / "assets" / f"{asset_id}{self.suffix}"
        if not file.exists():
            raise KeyError(f"no telemetry file for asset {asset_id!r}: {file}")
        return _slice(self._load(file), start, end)


class CsvSource(_DirectorySource):
    suffix = ".csv"

    def _load(self, file: Path) -> pd.DataFrame:
        df = pd.read_csv(file)
        df["timestamp"] = to_utc_naive_series(df["timestamp"])
        return df


class ParquetSource(_DirectorySource):
    """Needs the optional ``pyarrow`` (``pip install asset-health[parquet]``)."""

    suffix = ".parquet"

    def _load(self, file: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(file)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Parquet support needs: pip install 'asset-health[parquet]'") from exc


class FrameSource:
    """In-memory source, e.g. wrapping a ``SyntheticFleet`` or a dataframe you already hold."""

    def __init__(
        self,
        specs: list[AssetSpec],
        frames: dict[str, pd.DataFrame],
        maintenance: list[MaintenanceEvent] | None = None,
    ) -> None:
        self._specs = list(specs)
        self._frames = frames
        self._maintenance = list(maintenance or [])

    def assets(self) -> list[AssetSpec]:
        return list(self._specs)

    def read(
        self, asset_id: str, start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        return _slice(self._frames[asset_id], start, end)

    def maintenance_events(self, asset_id: str | None = None) -> list[MaintenanceEvent]:
        return [m for m in self._maintenance if asset_id is None or m.asset_id == asset_id]


def open_source(path: str | Path) -> TelemetrySource:
    """Open a dataset directory, choosing CSV or Parquet from what is on disk."""
    p = Path(path)
    assets = p / "assets"
    if assets.exists() and any(assets.glob("*.parquet")) and not any(assets.glob("*.csv")):
        return ParquetSource(p)
    return CsvSource(p)
