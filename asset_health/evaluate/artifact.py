"""Per-seed pipeline artefacts, computed once and cached on disk.

The expensive stage (simulate, ingest, features, detection) is a pure function of the simulator
config and the pipeline config. Policy tuning, RUL fitting and every metric then work from the
artefact, so a policy sweep does not re-run the detectors. The cache key hashes both configs and
the package version; a stale cache can never be read for a changed configuration.
"""

from __future__ import annotations

import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import asset_health
from asset_health.config import PipelineConfig, SimConfig
from asset_health.models import AssetSpec, GroundTruth
from asset_health.timeutil import to_utc_naive_series


@dataclass
class AssetArtifact:
    spec: AssetSpec
    frame: pd.DataFrame  # detection frame on the hourly grid (fused, n_firing, hypothesis ...)
    z24: pd.DataFrame
    repairs: list[pd.Timestamp]
    services: list[pd.Timestamp]
    blocked: bool
    dq_overall: str
    dq_usable_fraction: float
    naive: (
        pd.DataFrame
    )  # raw, ungated vib_rms / temp_c on the hourly grid (fixed-threshold baseline)
    seg_starts: list[pd.Timestamp] = field(default_factory=list)
    baseline_hours: int = 0


@dataclass
class SeedArtifact:
    seed: int
    sim: SimConfig
    truth: GroundTruth
    assets: dict[str, AssetArtifact]

    @property
    def months_per_asset(self) -> float:
        return self.sim.days / 30.4375


_ARTIFACT_SOURCES = (
    "config.py",
    "catalog.py",
    "models.py",
    "timeutil.py",
    "pipeline.py",
    "simulate",
    "ingest",
    "features",
    "detect",
)


def _source_fingerprint() -> str:
    """Hash of every source file the cached stage runs, so editing the simulator, the gate, the
    features or the detectors can never read an artefact built by the old code."""
    root = Path(asset_health.__file__).resolve().parent
    h = hashlib.sha1()
    for rel in _ARTIFACT_SOURCES:
        path = root / rel
        files = sorted(path.glob("*.py")) if path.is_dir() else [path]
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def config_key(sim: SimConfig, pipe: PipelineConfig) -> str:
    blob = (
        sim.model_dump_json()
        + pipe.model_dump_json()
        + asset_health.__version__
        + _source_fingerprint()
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _naive_series(raw: pd.DataFrame, start: pd.Timestamp, n: int, freq_s: int) -> pd.DataFrame:
    """What a static alarm limit would see: values at reported timestamps, nothing masked.
    ``temp_rise`` is temperature minus the logged ambient (the seasonal-proof variant)."""
    df = raw[["timestamp", "vib_rms", "temp_c", "ambient_c"]].copy()
    df["timestamp"] = to_utc_naive_series(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    df["temp_rise"] = df["temp_c"] - df["ambient_c"]
    grid = pd.date_range(start, periods=n, freq=f"{freq_s}s").as_unit("ns")
    out = df.reindex(grid, method="nearest", tolerance=pd.Timedelta(seconds=freq_s // 2))
    return out[["vib_rms", "temp_c", "temp_rise"]]


def _asset_job(args):  # module level so it can be pickled by the process pool
    from asset_health.pipeline import run_asset

    raw, spec, maint, pipe, sim = args
    run = run_asset(raw, spec, maint, pipe)
    naive = _naive_series(raw, pd.Timestamp(sim.start), sim.n_samples, sim.sample_interval_s)
    return spec.asset_id, AssetArtifact(
        spec=spec,
        frame=run.frame,
        z24=run.z24,
        repairs=run.repairs,
        services=run.services,
        blocked=run.blocked,
        dq_overall=run.clean.report.overall.value,
        dq_usable_fraction=run.clean.report.usable_fraction,
        naive=naive,
        seg_starts=[s.start for s in run.segments],
        baseline_hours=pipe.features.baseline_days * 24 + pipe.features.settle_hours,
    )


def build_seed_artifact(
    seed: int,
    sim: SimConfig | None = None,
    pipe: PipelineConfig | None = None,
    pool: ProcessPoolExecutor | None = None,
    cache_dir: Path | None = None,
) -> SeedArtifact:
    from asset_health.simulate import generate_fleet

    sim = (sim or SimConfig()).model_copy(update={"seed": seed})
    pipe = pipe or PipelineConfig()
    path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"seed_{seed}_{config_key(sim, pipe)}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)
    fleet = generate_fleet(sim, keep_latent=False)
    jobs = [(fleet.raw[s.asset_id], s, fleet.maintenance, pipe, sim) for s in fleet.specs]
    results = pool.map(_asset_job, jobs) if pool is not None else map(_asset_job, jobs)
    art = SeedArtifact(seed, sim, fleet.truth, dict(results))
    if path is not None:
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(art, fh)
        tmp.replace(path)
    return art


def healthy_hours(
    art: AssetArtifact, truth: GroundTruth, monitored: np.ndarray | None = None
) -> np.ndarray:
    """Boolean mask of monitored hours that are not inside this asset's failure window: the
    denominator of the false-alarm rate. ``monitored`` defaults to the system's own valid-decision
    hours; a baseline passes the hours *it* was watching."""
    if monitored is None:
        valid = art.frame["decision_valid"].to_numpy(dtype=bool).copy()
        index = art.frame.index
    else:
        valid = np.asarray(monitored, dtype=bool).copy()
        index = art.naive.index
    for f in truth.failures:
        if f.asset_id == art.spec.asset_id:
            on, fa = pd.Timestamp(f.onset), pd.Timestamp(f.failure)
            valid &= ~((index >= on) & (index <= fa))
    return valid
