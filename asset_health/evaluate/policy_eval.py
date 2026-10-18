"""Evaluate a policy (the system, or a baseline) on one seed artefact."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asset_health.alerting.cost import CostModel
from asset_health.alerting.policy import EpisodeTrace, find_episodes, threshold_for, trace_episode
from asset_health.config import PolicyConfig, RULConfig
from asset_health.evaluate.artifact import SeedArtifact, healthy_hours
from asset_health.evaluate.baselines import (
    NAIVE_CHANNELS,
    Ks,
    age_based_replacements,
    age_limit_days,
    fixed_threshold_episodes,
    fixed_threshold_monitored,
)
from asset_health.evaluate.costing import (
    CostBreakdown,
    charge_alarm_policy,
)
from asset_health.evaluate.events import FailureOutcome, near_data_fault, score_asset
from asset_health.models import AlertEpisode
from asset_health.rul.health_index import health_index
from asset_health.rul.model import RULModel

HOURS_PER_MONTH = 730.5


@dataclass
class SeedEval:
    seed: int
    outcomes: list[FailureOutcome] = field(default_factory=list)
    false_alarms: list[tuple[str, AlertEpisode]] = field(default_factory=list)
    traces: dict[tuple[str, pd.Timestamp], EpisodeTrace] = field(default_factory=dict)
    healthy_months: float = 0.0
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    n_fa_near_data_fault: int = 0
    n_coincidental: int = 0  # false alarms that opened inside a failure window before onset showed

    @property
    def fa_per_asset_month(self) -> float:
        return len(self.false_alarms) / self.healthy_months if self.healthy_months else float("nan")


class Memo:
    """Caches independent of the policy parameters, shared across a tuning sweep."""

    def __init__(self) -> None:
        self.hi: dict[tuple[int, str], pd.Series] = {}
        self.traces: dict[tuple, EpisodeTrace] = {}


def _hi(memo: Memo, seed: int, asset_id: str, z24: pd.DataFrame, rcfg: RULConfig) -> pd.Series:
    key = (seed, asset_id)
    if key not in memo.hi:
        memo.hi[key] = health_index(z24, rcfg.top_k)
    return memo.hi[key]


def evaluate_policy(
    art: SeedArtifact,
    pcfg: PolicyConfig,
    rcfg: RULConfig,
    costs: CostModel,
    model: RULModel | None,
    memo: Memo | None = None,
) -> SeedEval:
    memo = memo or Memo()
    res = SeedEval(art.seed)
    for asset_id, a in art.assets.items():
        thr = threshold_for(a.spec.asset_type, pcfg)
        eps = [] if a.blocked else find_episodes(a.frame, asset_id, thr, pcfg, a.repairs)
        sc = score_asset(art.seed, asset_id, eps, art.truth)
        hi = _hi(memo, art.seed, asset_id, a.z24, rcfg) if not a.blocked else None
        traces: dict[pd.Timestamp, EpisodeTrace] = {}
        for o in sc.outcomes:
            if o.episode is None or hi is None:
                continue
            key = (
                art.seed,
                asset_id,
                o.episode.opened_at,
                thr,
                id(model),
                pcfg.no_rul_action_days,
                pcfg.critical_days,
                pcfg.urgent_margin_days,
            )
            if key not in memo.traces:
                ep = o.episode.model_copy(deep=True)
                memo.traces[key] = trace_episode(
                    ep,
                    hi,
                    a.frame,
                    a.spec.asset_type,
                    model,
                    costs,
                    pcfg,
                    rcfg,
                    until=pd.Timestamp(o.failure.failure),  # nothing is decided after failure
                )
            traces[pd.Timestamp(o.episode.opened_at)] = memo.traces[key]
            res.traces[(asset_id, pd.Timestamp(o.episode.opened_at))] = memo.traces[key]

        def exec_of(o: FailureOutcome, _t=traces):
            tr = _t[pd.Timestamp(o.episode.opened_at)]  # type: ignore[union-attr]
            return tr.execution_at

        charge_alarm_policy(
            res.cost, costs, a.spec.asset_type, sc.outcomes, len(sc.false_alarms), exec_of
        )
        res.outcomes.extend(sc.outcomes)
        res.false_alarms.extend(sc.false_alarms)
        res.n_coincidental += len(sc.coincidental)
        res.healthy_months += float(healthy_hours(a, art.truth).sum()) / HOURS_PER_MONTH
        for _, e in sc.false_alarms:
            if near_data_fault(art.truth, asset_id, pd.Timestamp(e.opened_at)):
                res.n_fa_near_data_fault += 1
    return res


def evaluate_fixed_threshold(
    art: SeedArtifact,
    k: Ks,
    costs: CostModel,
    confirm_hours: int = 3,
    channels: tuple[str, ...] = NAIVE_CHANNELS,
) -> SeedEval:
    """Static limits (``k`` per channel or shared). The false-alarm denominator is the hours the
    limit itself was watching, not the system's valid-decision hours."""
    res = SeedEval(art.seed)
    for asset_id, a in art.assets.items():
        eps = fixed_threshold_episodes(a, k, confirm_hours, channels=channels)
        sc = score_asset(art.seed, asset_id, eps, art.truth)

        def exec_of(o: FailureOutcome):
            return pd.Timestamp(o.episode.opened_at) + pd.Timedelta(days=costs.plan_lead_days)  # type: ignore[union-attr]

        charge_alarm_policy(
            res.cost, costs, a.spec.asset_type, sc.outcomes, len(sc.false_alarms), exec_of
        )
        res.outcomes.extend(sc.outcomes)
        res.false_alarms.extend(sc.false_alarms)
        res.n_coincidental += len(sc.coincidental)
        watched = fixed_threshold_monitored(a, channels)
        res.healthy_months += float(healthy_hours(a, art.truth, watched).sum()) / HOURS_PER_MONTH
        for _, e in sc.false_alarms:
            if near_data_fault(art.truth, asset_id, pd.Timestamp(e.opened_at)):
                res.n_fa_near_data_fault += 1
    return res


@dataclass
class CalendarEval:
    seed: int
    cost: CostBreakdown
    n_replacements: int
    n_prevented: int
    n_failures: int


def evaluate_calendar(art: SeedArtifact, life_fraction: float, costs: CostModel) -> CalendarEval:
    """Age-based replacement at ``life_fraction`` x the class's characteristic life.

    Charged like every other policy: planned cost per replacement, the life thrown away before a
    failure it prevented (``early_loss_per_day`` x days), the unplanned cost of a failure it did
    not prevent. As for the alarm-driven policies, the life of a healthy part that is replaced is
    not charged (its remaining life lies beyond the record)."""
    cost = CostBreakdown()
    n_repl = n_prev = n_fail = 0
    end = pd.Timestamp(art.sim.start) + pd.Timedelta(days=art.sim.days)
    start = pd.Timestamp(art.sim.start)
    for asset_id, a in art.assets.items():
        c = costs.scaled(a.spec.asset_type)
        fails = [f for f in art.truth.failures if f.asset_id == asset_id]
        f = fails[0] if fails else None
        repair_at = None
        if f is not None:
            later = [r for r in a.repairs if r >= pd.Timestamp(f.failure)]
            repair_at = min(later) if later else None
        repl, prevented = age_based_replacements(
            a.spec.component_age_days,
            age_limit_days(a, life_fraction),
            start,
            end,
            f,
            repair_at,
        )
        cost.planned += c.planned_replacement * len(repl)
        cost.n_planned += len(repl)
        n_repl += len(repl)
        if f is None:
            continue
        n_fail += 1
        if prevented is not None:
            # that replacement is already counted above; only the life wasted is extra
            n_prev += 1
            cost.early_loss += c.early_loss_per_day * (
                (pd.Timestamp(f.failure) - prevented).total_seconds() / 86400.0
            )
        else:
            cost.unplanned += c.unplanned_failure
            cost.n_unplanned += 1
    return CalendarEval(art.seed, cost, n_repl, n_prev, n_fail)


def months_of_asset_time(art: SeedArtifact) -> float:
    return len(art.assets) * art.sim.days / 30.4375


def lead_days(outcomes: list[FailureOutcome]) -> np.ndarray:
    return np.array([o.lead_days for o in outcomes if o.lead_days is not None], dtype=float)
