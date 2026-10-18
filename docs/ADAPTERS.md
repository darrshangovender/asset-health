# Adapters: getting plant data in (and work orders out)

**Implemented in this repo:** `CsvSource`, `ParquetSource` and `FrameSource` in
`asset_health/ingest/adapters.py`.

**Not implemented in this repo:** OPC UA, MQTT, PI Web API, Ignition, Wonderware or any other
historian or CMMS connector. This page describes how one would plug in. Nothing below has been run
against a real plant system.

## The seam: `TelemetrySource`

```python
class TelemetrySource(Protocol):
    def assets(self) -> list[AssetSpec]: ...
    def read(self, asset_id: str, start=None, end=None) -> pd.DataFrame: ...
    def maintenance_events(self, asset_id: str | None = None) -> list[MaintenanceEvent]: ...
```

`read` returns one row per sample with a `timestamp` column (UTC) and one column per channel
(`load_pct`, `speed_rpm`, `ambient_c`, `vib_rms`, `vib_band_*`, `temp_c`, `current_a`, `power_kw`,
`pressure_kpa`, `flow_m3h`, whichever the asset type has). Everything downstream, including the
data-quality gate, works from that contract. Anything that satisfies the Protocol can be passed to
`asset_health.analysis.analyse_source`.

## What a historian or OPC UA adapter has to get right

| Concern | What the adapter should do |
|---|---|
| Tag mapping | A small YAML file mapping each channel of each asset to a tag path, with engineering units. The asset register (`AssetSpec`) comes from the CMMS functional-location hierarchy, not from the tags. |
| Quality codes | Map OPC UA `StatusCode` (Good / Uncertain / Bad) and historian quality flags to `NaN` for Bad and Uncertain. Do not forward a Bad value as a number: the gate would have to guess. |
| Timestamps | Prefer the **source** timestamp over the server timestamp. Send UTC, or ISO 8601 with an offset: the package converts every timestamp (telemetry, maintenance records, `--as-of`) to naive UTC, and treats a naive value as UTC. Naive *local* wall-clock time with daylight saving cannot be repaired downstream (the autumn hour is ambiguous); convert it in the adapter. Keep the logger's own clock as delivered; the ingest stage estimates drift and rebuilds the sampling grid. |
| Row order and sources | One source per asset, rows in time order. A cleanly newest-first file is reversed and flagged; shuffled rows, two collectors interleaved (rows sharing a timestamp with different values) or a series whose re-timed grid would run past its last timestamp **block the asset** with a message, before any grid is built. |
| Sampling | Pull raw values or a fixed aggregate at the configured cadence (1 h by default: time-weighted average for process channels, 1 h RMS and band energies from the edge analyser for vibration). The gate infers the cadence (median step) and blocks an asset whose cadence differs from `IngestConfig.sample_interval_s` by more than 10%; `IngestConfig(resample=True)` (CLI `--resample`) averages a finer export onto the grid instead, without clock-drift correction. Do not interpolate gaps on the way in: gaps are information. |
| Vibration | Band energies in orders (multiples of shaft speed) need an edge analyser that order-tracks against a tachometer. Fixed-frequency bands on a variable-speed machine smear defect lines; the simulator here assumes order-tracked data. |
| Back-fill and replays | Historian replays produce duplicate rows. The gate drops exact duplicates; the adapter should still request by time range with a watermark. |
| Maintenance log | `MaintenanceEvent(kind="planned_service" | "repair")` from CMMS work-order completions. A `repair` starts a new baseline (the component changed), so it matters that these are real replacement events, not inspections. |
| Credentials | Read-only account, least privilege, secrets from the environment or a vault. This package never stores credentials. |

## Sketch (illustrative, untested)

```python
class OpcUaHistorianSource:                      # NOT shipped; shape only
    def __init__(self, endpoint, tag_map, registry): ...
    def assets(self): return self.registry.load_assets()
    def read(self, asset_id, start=None, end=None):
        # 1. history-read each mapped tag for [start, end)
        # 2. drop Bad/Uncertain-quality samples (NaN), keep source timestamps (UTC)
        # 3. outer-join on timestamp into one frame with the channel names above
        ...
    def maintenance_events(self, asset_id=None):
        return self.cmms.completed_work_orders(asset_id)   # replacements and services
```

## Work orders out

`WorkOrder` (pydantic) is machine-readable first: asset, severity, failure-mode hypothesis,
evidence features with observed and expected values, recommended action, latest safe date and the
RUL interval. A CMMS integration (SAP PM, IBM Maximo or another system) would map those fields onto
a notification or work-request record and attach the narrative as the long text. The mapping, the
approval workflow and the authority to release work belong to the plant: this package recommends
and never releases anything. No CMMS connector is included.

## Edge or cloud, and where the data lives

The pipeline is a pure function of the data it is given and needs no network, so the same code can
run on an edge gateway (per-asset detection, small memory footprint) or on a server in the plant's
own region. Data stays wherever the adapter reads it from; nothing is sent anywhere by default. The
optional LLM narrative (`asset-health[llm]`) is the only component that would call an external
service, only if you configure it, and only with the structured work order as input.
