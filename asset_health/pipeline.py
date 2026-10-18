"""Orchestration: raw telemetry -> data-quality gate -> features -> detection (-> alerts, RUL).

The per-asset timeline is cut into *component life segments* at CMMS repair records (a replaced
bearing or liner has a new healthy signature), and the healthy baseline is re-learned at the start
of every segment. Planned services do not start a new segment; they only hold the detectors for a
run-in window (see ``detect.ensemble.hold_mask``).

Scores are causal: a score at time t only uses rows up to t, so a run on a truncated history
reproduces the scores of the full run for the overlapping rows (tested). The asset-level gate
summarises the history it is given; ``as_of`` truncates that history first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asset_health.config import PipelineConfig
from asset_health.detect.ensemble import Detection, detect_asset
from asset_health.features.build import AssetFeatures, build_features
from asset_health.ingest.adapters import TelemetrySource
from asset_health.ingest.checks import CleanAsset, ingest_asset
from asset_health.models import AssetSpec, MaintenanceEvent


@dataclass
class SegmentRun:
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive
    features: AssetFeatures
    detection: Detection


@dataclass
class AssetRun:
    """Everything the pipeline knows about one asset. ``frame`` is on the full hourly grid."""

    spec: AssetSpec
    clean: CleanAsset
    segments: list[SegmentRun]
    frame: pd.DataFrame
    z24: pd.DataFrame
    obs24: pd.DataFrame
    exp24: pd.DataFrame
    repairs: list[pd.Timestamp] = field(default_factory=list)
    services: list[pd.Timestamp] = field(default_factory=list)

    @property
    def asset_id(self) -> str:
        return self.spec.asset_id

    @property
    def blocked(self) -> bool:
        return self.clean.blocked


def _ts(m: MaintenanceEvent) -> pd.Timestamp:
    return pd.Timestamp(m.timestamp).as_unit("ns")


def segment_bounds(
    index: pd.DatetimeIndex, repairs: list[pd.Timestamp]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Cut the grid at repair times: ``[(start, end_exclusive), ...]``."""
    if len(index) == 0:
        return []
    cuts = sorted({t for t in repairs if index[0] < t <= index[-1]})
    starts = [index[0], *cuts]
    ends = [*cuts, index[-1] + pd.Timedelta(seconds=1)]
    return list(zip(starts, ends, strict=True))


def run_asset(
    raw: pd.DataFrame,
    spec: AssetSpec,
    maintenance: list[MaintenanceEvent] | None = None,
    cfg: PipelineConfig | None = None,
    as_of: pd.Timestamp | None = None,
) -> AssetRun:
    """Run one asset. ``as_of`` drops every row and maintenance record after that instant before
    the data-quality gate runs, so a replay uses nothing later (the block decision included)."""
    cfg = cfg or PipelineConfig()
    maintenance = [m for m in (maintenance or []) if m.asset_id == spec.asset_id]
    if as_of is not None:
        maintenance = [m for m in maintenance if _ts(m) <= as_of]
    clean = ingest_asset(raw, spec, cfg.ingest, maintenance, as_of=as_of)
    repairs = [_ts(m) for m in maintenance if m.kind == "repair"]
    services = [_ts(m) for m in maintenance if m.kind == "planned_service"]
    index = clean.frame.index
    if clean.blocked or len(index) == 0:
        empty = pd.DataFrame(index=index)
        return AssetRun(
            spec, clean, [], _blank_frame(index), empty, empty, empty, repairs, services
        )

    segments: list[SegmentRun] = []
    pieces: list[pd.DataFrame] = []
    z24s: list[pd.DataFrame] = []
    obs: list[pd.DataFrame] = []
    exp: list[pd.DataFrame] = []
    for k, (s, e) in enumerate(segment_bounds(index, repairs)):
        feats = build_features(clean, s, e, cfg.features)
        seg_services = [t for t in services if s <= t < e]
        det = detect_asset(feats, seg_services, cfg.detect, cfg.features)
        segments.append(SegmentRun(s, e, feats, det))
        f = det.frame.copy()
        f["segment"] = k
        f["running"] = feats.running.to_numpy()
        f["n_valid"] = feats.n_valid.to_numpy()
        pieces.append(f)
        z24s.append(feats.z24)
        obs.append(feats.obs24)
        exp.append(feats.exp24)
    frame = pd.concat(pieces)
    return AssetRun(
        spec,
        clean,
        segments,
        frame,
        pd.concat(z24s),
        pd.concat(obs),
        pd.concat(exp),
        repairs,
        services,
    )


def _blank_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    from asset_health.models import FaultHypothesis

    return pd.DataFrame(
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
            "segment": 0,
            "running": False,
            "n_valid": 0,
        },
        index=index,
    )


def _slim_job(args: tuple) -> AssetRun:
    """Process-pool worker: run one asset and drop the per-segment working frames, which are
    large and not needed once the concatenated frames exist."""
    raw, spec, maintenance, cfg, as_of = args
    run = run_asset(raw, spec, maintenance, cfg, as_of)
    run.segments = []
    return run


def run_source(
    source: TelemetrySource,
    cfg: PipelineConfig | None = None,
    asset_ids: list[str] | None = None,
    n_jobs: int = 1,
    as_of: pd.Timestamp | None = None,
) -> dict[str, AssetRun]:
    """Run the pipeline over every asset a ``TelemetrySource`` exposes.

    ``n_jobs > 1`` spreads assets over processes (the per-segment working frames are then not
    returned; ``AssetRun.segments`` is empty). ``as_of`` replays the moment: nothing stamped
    later is read into any stage.
    """
    cfg = cfg or PipelineConfig()
    maintenance = source.maintenance_events()
    specs = [s for s in source.assets() if asset_ids is None or s.asset_id in asset_ids]
    if n_jobs <= 1 or len(specs) < 2:
        return {
            s.asset_id: run_asset(source.read(s.asset_id), s, maintenance, cfg, as_of)
            for s in specs
        }
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(source.read(s.asset_id), s, maintenance, cfg, as_of) for s in specs]
    with ProcessPoolExecutor(min(n_jobs, len(jobs))) as pool:
        runs = list(pool.map(_slim_job, jobs))
    return {r.spec.asset_id: r for r in runs}
