"""Cost-aware alert policy: debounce, hysteresis, severity levels and the maintenance decision.

Stages
1. ``find_episodes`` turns the fused detection score into alert episodes. An alert opens only
   after the score has been above the threshold for ``confirm_hours`` *valid* hours, and closes only
   after ``clear_hours`` valid hours below ``clear_ratio * threshold`` (hysteresis). Hours with no
   trustworthy data neither open nor close anything: bad data never raises or clears an alarm.
2. ``trace_episode`` re-estimates RUL at a fixed cadence inside an open episode, grades severity
   (WARNING, URGENT, CRITICAL) and decides when to act, using the cost-optimal replacement day
   from ``alerting.cost``.
3. The threshold itself scales with the cost of missing a failure on that asset class
   (``cost_exponent``), and is tuned on development seeds by ``evaluate.tuning``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asset_health.alerting.cost import CostModel, latest_safe_days
from asset_health.catalog import PROFILES
from asset_health.config import PolicyConfig, RULConfig
from asset_health.models import (
    AlertEpisode,
    AssetType,
    FaultHypothesis,
    RULEstimate,
    RULStatus,
    Severity,
)
from asset_health.rul.estimate import estimate_rul
from asset_health.rul.model import RULModel


def threshold_for(asset_type: AssetType, cfg: PolicyConfig) -> float:
    """Alert threshold for an asset class: the costlier a miss, the lower (more sensitive)."""
    m = PROFILES[asset_type].cost_multiplier
    return float(cfg.alert_threshold * m ** (-cfg.cost_exponent))


def find_episodes(
    frame: pd.DataFrame,
    asset_id: str,
    threshold: float,
    cfg: PolicyConfig,
    repairs: list[pd.Timestamp] | None = None,
) -> list[AlertEpisode]:
    """Alert episodes from a detection frame (columns ``fused``, ``n_firing``,
    ``decision_valid``, ``hypothesis``, ``rule``). Pure and causal."""
    idx = frame.index
    fused = frame["fused"].to_numpy(dtype=float)
    agree = frame["n_firing"].to_numpy() >= cfg.min_detectors
    valid = frame["decision_valid"].to_numpy(dtype=bool) & ~np.isnan(fused)
    clear_level = cfg.clear_ratio * threshold
    repair_pos = sorted(int(idx.searchsorted(t)) for t in (repairs or []))
    rp = 0

    episodes: list[AlertEpisode] = []
    open_i: int | None = None
    first_i: int | None = None
    run_above = run_below = 0
    peak = 0.0

    def close(i: int, reason: str) -> None:
        nonlocal open_i, first_i, run_above, run_below, peak
        assert open_i is not None
        episodes.append(
            AlertEpisode(
                asset_id=asset_id,
                opened_at=idx[open_i].to_pydatetime(),
                closed_at=idx[min(i, len(idx) - 1)].to_pydatetime(),
                peak_severity=Severity.WARNING,
                closed_reason=reason,
                first_exceeded_at=idx[first_i].to_pydatetime() if first_i is not None else None,
                peak_score=peak,
                threshold=threshold,
            )
        )
        open_i = first_i = None
        run_above = run_below = 0
        peak = 0.0

    for i in range(len(idx)):
        while rp < len(repair_pos) and repair_pos[rp] <= i:
            if open_i is not None:
                close(repair_pos[rp], "maintenance_completed")
            run_above = run_below = 0
            first_i = None
            rp += 1
        if not valid[i]:
            continue  # no trustworthy reading: hold state
        s = fused[i]
        if open_i is None:
            if s >= threshold and agree[i]:
                if run_above == 0:
                    first_i = i
                run_above += 1
                peak = max(peak, s)
                if run_above >= cfg.confirm_hours:
                    open_i = i
                    run_below = 0
            else:
                run_above = 0
                first_i = None
                peak = 0.0
        else:
            peak = max(peak, s)
            if s < clear_level:
                run_below += 1
                if run_below >= cfg.clear_hours:
                    close(i, "recovered")
            else:
                run_below = 0
    if open_i is not None:
        episodes.append(
            AlertEpisode(
                asset_id=asset_id,
                opened_at=idx[open_i].to_pydatetime(),
                closed_at=None,
                peak_severity=Severity.WARNING,
                first_exceeded_at=idx[first_i].to_pydatetime() if first_i is not None else None,
                peak_score=peak,
                threshold=threshold,
            )
        )
    return episodes


def episode_hypothesis(
    frame: pd.DataFrame, end: pd.Timestamp, window_h: int
) -> tuple[FaultHypothesis, list[FaultHypothesis]]:
    """Fault hypothesis from the rule layer over the trailing window: strength-weighted vote."""
    w = frame.loc[end - pd.Timedelta(hours=window_h) : end]
    w = w[w["decision_valid"] & (w["rule"] > 0)]
    if w.empty:
        return FaultHypothesis.UNEXPLAINED, []
    votes = w.groupby("hypothesis")["rule"].sum().sort_values(ascending=False)
    top = FaultHypothesis(votes.index[0])
    alts = [
        FaultHypothesis(h)
        for h, v in votes.items()
        if h != votes.index[0] and v >= 0.3 * votes.iloc[0]
    ]
    return top, alts


@dataclass
class Checkpoint:
    t: pd.Timestamp
    rul: RULEstimate
    severity: Severity
    latest_safe_days: float | None
    act_now: bool


@dataclass
class EpisodeTrace:
    """An alert episode plus its RUL checkpoints and the maintenance decision it led to."""

    episode: AlertEpisode
    checkpoints: list[Checkpoint] = field(default_factory=list)
    action_at: pd.Timestamp | None = None  # decision time (crew mobilised)
    execution_at: pd.Timestamp | None = None  # planned replacement time

    @property
    def hypothesis(self) -> FaultHypothesis:
        return self.episode.hypothesis


def grade(
    est: RULEstimate, costs: CostModel, cfg: PolicyConfig
) -> tuple[Severity, float | None, bool]:
    """(severity, latest safe day count, act now?) for one RUL estimate inside an alert."""
    if est.status != RULStatus.ESTIMATED:
        return Severity.WARNING, None, False
    assert est.p10_days is not None and est.p50_days is not None and est.p90_days is not None
    lsd = latest_safe_days(est.p10_days, est.p50_days, est.p90_days, costs)
    cadence_days = cfg.checkpoint_hours / 24.0
    if est.p10_days <= cfg.critical_days:
        sev = Severity.CRITICAL
    elif lsd <= costs.plan_lead_days + cfg.urgent_margin_days:
        sev = Severity.URGENT
    else:
        sev = Severity.WARNING
    return sev, lsd, lsd <= costs.plan_lead_days + cadence_days


def trace_episode(
    episode: AlertEpisode,
    hi: pd.Series,
    detection: pd.DataFrame,
    asset_type: AssetType,
    model: RULModel | None,
    costs: CostModel,
    pcfg: PolicyConfig,
    rcfg: RULConfig,
    until: pd.Timestamp | None = None,
) -> EpisodeTrace:
    """Walk an episode at the checkpoint cadence: RUL, severity, and when to act."""
    scaled = costs.scaled(asset_type)
    start = pd.Timestamp(episode.opened_at)
    stop = pd.Timestamp(episode.closed_at) if episode.closed_at is not None else hi.index[-1]
    if until is not None:
        stop = min(stop, until)
    trace = EpisodeTrace(episode=episode)
    t = start
    step = pd.Timedelta(hours=pcfg.checkpoint_hours)
    peak = Severity.WARNING
    while t <= stop:
        est = estimate_rul(episode.asset_id, hi, t, model, rcfg, asset_type)
        sev, lsd, act = grade(est, scaled, pcfg)
        if est.status != RULStatus.ESTIMATED and (t - start).days >= pcfg.no_rul_action_days:
            act = True  # no interval to schedule against: inspect and decide, do not wait forever
        trace.checkpoints.append(Checkpoint(t, est, sev, lsd, act))
        if sev.rank > peak.rank:
            peak = sev
        if sev.rank >= Severity.URGENT.rank and episode.urgent_at is None:
            episode.urgent_at = t.to_pydatetime()
        if act and trace.action_at is None:
            trace.action_at = t
            trace.execution_at = t + pd.Timedelta(days=scaled.plan_lead_days)
        t += step
    episode.peak_severity = peak
    hyp, _ = episode_hypothesis(detection, min(stop, detection.index[-1]), pcfg.hypothesis_window_h)
    episode.hypothesis = hyp
    return trace
