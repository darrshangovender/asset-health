"""End-to-end analysis of a fleet: pipeline -> alerts -> RUL -> work orders.

This is the object the CLI, the HTML report and the examples use. The evaluation harness works on
the same building blocks but caches the expensive pipeline stage per seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from asset_health.alerting.cost import CostModel
from asset_health.alerting.narrative import NarrativeGenerator
from asset_health.alerting.policy import (
    EpisodeTrace,
    episode_hypothesis,
    find_episodes,
    threshold_for,
    trace_episode,
)
from asset_health.alerting.workorder import build_work_order
from asset_health.config import PipelineConfig, PolicyConfig, RULConfig
from asset_health.ingest.adapters import TelemetrySource
from asset_health.ingest.checks import report_as_of
from asset_health.models import GroundTruth, Severity, WorkOrder
from asset_health.pipeline import AssetRun, run_source
from asset_health.rul.health_index import health_index
from asset_health.rul.model import RULModel
from asset_health.timeutil import to_utc_naive


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    rul: RULConfig = Field(default_factory=RULConfig)
    costs: CostModel = Field(default_factory=CostModel)


def load_default_rul_model() -> RULModel | None:
    """The RUL model shipped with the package (trained on SYNTHETIC development seeds)."""
    try:
        path = resources.files("asset_health").joinpath("data/rul_model.json")
        return RULModel.model_validate_json(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError):
        return None


@dataclass
class AssetAnalysis:
    run: AssetRun
    hi: pd.Series
    threshold: float
    traces: list[EpisodeTrace] = field(default_factory=list)
    work_orders: list[WorkOrder] = field(default_factory=list)
    severity_now: Severity = Severity.NORMAL
    watch: bool = False
    blocked: bool = False  # the data-quality gate as it stood at the analysis instant

    @property
    def asset_id(self) -> str:
        return self.run.asset_id

    @property
    def open_trace(self) -> EpisodeTrace | None:
        for tr in self.traces:
            if tr.episode.closed_at is None:
                return tr
        return None


@dataclass
class FleetAnalysis:
    assets: dict[str, AssetAnalysis]
    as_of: pd.Timestamp
    config: AnalysisConfig
    rul_model: RULModel | None
    truth: GroundTruth | None = None

    def all_work_orders(self) -> list[WorkOrder]:
        wos = [wo for a in self.assets.values() for wo in a.work_orders]
        return sorted(wos, key=lambda w: (-w.severity.rank, w.asset_id))


def analyse_run(
    run: AssetRun,
    cfg: AnalysisConfig,
    model: RULModel | None,
    narrator: NarrativeGenerator | None = None,
    as_of: pd.Timestamp | None = None,
) -> AssetAnalysis:
    spec = run.spec
    thr = threshold_for(spec.asset_type, cfg.policy)
    report = report_as_of(run.clean, as_of)
    if run.blocked and as_of is not None and not report.asset_blocked:
        raise ValueError(
            f"{spec.asset_id} was blocked on data after {as_of}; the run cannot replay that "
            "moment. Re-run with run_source(..., as_of=as_of) or analyse_source(..., as_of=as_of)."
        )
    if report.asset_blocked or run.frame["fused"].notna().sum() == 0:
        return AssetAnalysis(run, pd.Series(dtype=float), thr, blocked=report.asset_blocked)
    hi = health_index(run.z24, cfg.rul.top_k)
    cut = run.frame.index[-1] if as_of is None else as_of
    frame = run.frame.loc[:cut]
    episodes = find_episodes(frame, spec.asset_id, thr, cfg.policy, run.repairs)
    traces = [
        trace_episode(
            ep, hi.loc[:cut], frame, spec.asset_type, model, cfg.costs, cfg.policy, cfg.rul
        )
        for ep in episodes
    ]
    out = AssetAnalysis(run, hi, thr, traces)
    open_tr = out.open_trace
    if open_tr is not None:
        _, alts = episode_hypothesis(frame, cut, cfg.policy.hypothesis_window_h)
        wo = build_work_order(
            spec, open_tr, cut, run.z24, run.obs24, run.exp24, report, alts, narrator
        )
        out.work_orders.append(wo)
        out.severity_now = wo.severity
    else:
        last = frame["fused"].dropna()
        out.watch = bool(len(last) and last.iloc[-24:].max() >= cfg.policy.watch_ratio * thr)
        out.severity_now = Severity.WATCH if out.watch else Severity.NORMAL
    return out


def analyse_runs(
    runs: dict[str, AssetRun],
    cfg: AnalysisConfig | None = None,
    model: RULModel | None = None,
    narrator: NarrativeGenerator | None = None,
    truth: GroundTruth | None = None,
    as_of: pd.Timestamp | None = None,
) -> FleetAnalysis:
    """Analyse every asset as of ``as_of`` (default: the end of the data). Alerts, RUL, the
    asset gate and the data-quality note use nothing after that instant. Runs computed on a longer
    history are cut at ``as_of``; ``analyse_source(..., as_of=...)`` reads nothing later at all."""
    cfg = cfg or AnalysisConfig()
    if as_of is not None:
        as_of = to_utc_naive(as_of)
    model = model if model is not None else load_default_rul_model()
    assets = {k: analyse_run(r, cfg, model, narrator, as_of) for k, r in runs.items()}
    ends = [r.frame.index[-1] for r in runs.values() if len(r.frame)]
    last = max(ends) if ends else pd.Timestamp(0)
    return FleetAnalysis(assets, last if as_of is None else min(as_of, last), cfg, model, truth)


def analyse_source(
    source: TelemetrySource,
    cfg: AnalysisConfig | None = None,
    model: RULModel | None = None,
    narrator: NarrativeGenerator | None = None,
    truth: GroundTruth | None = None,
    n_jobs: int = 1,
    as_of: pd.Timestamp | None = None,
) -> FleetAnalysis:
    cfg = cfg or AnalysisConfig()
    if as_of is not None:
        as_of = to_utc_naive(as_of)
    runs = run_source(source, cfg.pipeline, n_jobs=n_jobs, as_of=as_of)
    return analyse_runs(runs, cfg, model, narrator, truth, as_of)
