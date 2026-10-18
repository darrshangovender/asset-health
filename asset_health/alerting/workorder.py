"""Structured work-order recommendations (pydantic first, prose derived)."""

from __future__ import annotations

import pandas as pd

from asset_health.alerting.narrative import NarrativeGenerator, narrate
from asset_health.alerting.policy import EpisodeTrace
from asset_health.catalog import CHANNEL_DESCRIPTIONS, CHANNEL_UNITS
from asset_health.models import (
    AssetSpec,
    DataQualityReport,
    Evidence,
    FaultHypothesis,
    RULEstimate,
    RULStatus,
    Severity,
    WorkOrder,
)
from asset_health.rul.health_index import harmful_residuals

ACTIONS: dict[FaultHypothesis, str] = {
    FaultHypothesis.BEARING_OUTER_RACE: (
        "Inspect the bearing housing, check lubrication and re-grease if due, confirm with a "
        "handheld envelope/spectrum reading, and plan bearing replacement."
    ),
    FaultHypothesis.BEARING_INNER_RACE: (
        "Inspect the bearing housing, check lubrication and shaft fit, confirm with a handheld "
        "envelope/spectrum reading, and plan bearing replacement."
    ),
    FaultHypothesis.BEARING_GENERIC: (
        "Check bearing lubrication and housing temperature, take a handheld vibration reading and "
        "re-inspect within 72 hours."
    ),
    FaultHypothesis.IMPELLER_WEAR: (
        "Check pump performance against the curve, inspect the impeller and liner clearances, and "
        "plan an impeller replacement."
    ),
    FaultHypothesis.LINER_WEAR: (
        "Measure the liner profile and closed-side setting, compare crusher power against feed, "
        "and plan a liner change."
    ),
    FaultHypothesis.OVERHEATING: (
        "Check cooling and lubrication (coolant level, radiator or cooler fouling, oil level) "
        "before the next shift and restrict load until cleared."
    ),
    FaultHypothesis.UNEXPLAINED: (
        "Inspect the asset and confirm the reading; no specific fault signature is established."
    ),
}


def evidence_from(
    z24_row: pd.Series, obs_row: pd.Series, exp_row: pd.Series, n: int = 4
) -> list[Evidence]:
    """The ``n`` channels furthest from their regime-expected value, in harmful direction."""
    harm = harmful_residuals(z24_row.to_frame().T).iloc[0].dropna().sort_values(ascending=False)
    out: list[Evidence] = []
    for name, h in harm.items():
        if h < 2.0 or len(out) >= n:
            break
        obs, exp = obs_row.get(name), exp_row.get(name)
        if obs is None or exp is None or pd.isna(obs) or pd.isna(exp):
            continue
        out.append(
            Evidence(
                feature=str(name),
                description=CHANNEL_DESCRIPTIONS.get(str(name), str(name)),
                observed=float(obs),
                expected=float(exp),
                z_score=float(z24_row[name]),
                unit=CHANNEL_UNITS.get(str(name), ""),
            )
        )
    return out


def data_quality_note(report: DataQualityReport) -> str:
    n_adv = len(report.sensor_advisories)
    bad = [c for c, d in report.channels.items() if not d.usable]
    bits = []
    if bad:
        bits.append(f"channel(s) {', '.join(sorted(bad))} unusable and excluded")
    if n_adv:
        bits.append(f"{n_adv} sensor advisory item(s); affected readings were masked, not used")
    return "; ".join(bits)


def build_work_order(
    spec: AssetSpec,
    trace: EpisodeTrace,
    as_of: pd.Timestamp,
    z24: pd.DataFrame,
    obs24: pd.DataFrame,
    exp24: pd.DataFrame,
    report: DataQualityReport,
    alternatives: list[FaultHypothesis],
    narrator: NarrativeGenerator | None = None,
) -> WorkOrder:
    """Work order for an alert episode as of ``as_of`` (uses only data up to that time)."""
    ep = trace.episode
    cps = [c for c in trace.checkpoints if c.t <= as_of]
    last = cps[-1] if cps else None
    rul: RULEstimate | None = last.rul if last else None
    severity = last.severity if last else Severity.WARNING
    row_t = z24.index[z24.index.searchsorted(as_of, side="right") - 1]
    window = z24.loc[row_t - pd.Timedelta(hours=24) : row_t].dropna(how="all")
    if window.empty:
        evidence: list[Evidence] = []
    else:
        t_ev = window.index[-1]
        evidence = evidence_from(z24.loc[t_ev], obs24.loc[t_ev], exp24.loc[t_ev])
    latest = None
    if (
        rul is not None
        and rul.status == RULStatus.ESTIMATED
        and last
        and last.latest_safe_days is not None
    ):
        latest = (as_of + pd.Timedelta(days=last.latest_safe_days)).to_pydatetime()
    action = ACTIONS[ep.hypothesis]
    if severity in (Severity.URGENT, Severity.CRITICAL):
        action = "Expedite: " + action[0].lower() + action[1:]
    wo = WorkOrder(
        work_order_id=f"WO-{spec.asset_id}-{pd.Timestamp(ep.opened_at):%Y%m%d}",
        asset_id=spec.asset_id,
        asset_name=spec.name,
        asset_type=spec.asset_type,
        issued_at=as_of.to_pydatetime(),
        severity=severity,
        failure_mode_hypothesis=ep.hypothesis,
        alternative_hypotheses=alternatives,
        evidence=evidence,
        recommended_action=action,
        latest_safe_date=latest,
        rul=rul,
        data_quality_note=data_quality_note(report),
    )
    return narrate(wo, narrator)
