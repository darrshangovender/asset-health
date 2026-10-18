"""End-to-end behaviour on a fixed seed: injected failures must be caught, with lead time, and
bad data must not raise alarms. These are the tests that make the README claims checkable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_health.alerting.narrative import check_narrative
from asset_health.analysis import AnalysisConfig, analyse_run
from asset_health.models import HYPOTHESIS_FAMILY, RULStatus, Severity
from asset_health.pipeline import run_asset

MIN_LEAD_DAYS = 3.0


def _failures(small_fleet):
    return {f.asset_id: f for f in small_fleet.truth.failures}


def test_the_fixture_actually_injects_failures(small_fleet):
    assert len(small_fleet.truth.failures) >= 2


def test_every_injected_failure_is_detected_with_lead_time(small_fleet, small_analysis):
    missed, short = [], []
    for aid, f in _failures(small_fleet).items():
        a = small_analysis.assets[aid]
        opened = [
            pd.Timestamp(tr.episode.opened_at)
            for tr in a.traces
            if pd.Timestamp(f.onset) <= pd.Timestamp(tr.episode.opened_at) < pd.Timestamp(f.failure)
        ]
        if not opened:
            missed.append(aid)
            continue
        lead = (pd.Timestamp(f.failure) - min(opened)).total_seconds() / 86400
        if lead < MIN_LEAD_DAYS:
            short.append((aid, lead))
    assert not missed, f"missed failures: {missed}"
    assert not short, f"alerts too late to act on: {short}"


def test_alerts_open_during_the_degradation_not_before_it(small_fleet, small_analysis):
    for aid, f in _failures(small_fleet).items():
        for tr in small_analysis.assets[aid].traces:
            opened = pd.Timestamp(tr.episode.opened_at)
            assert opened >= pd.Timestamp(f.onset), f"{aid}: alert before the injected onset"


def test_healthy_assets_raise_no_alerts(small_fleet, small_analysis):
    failing = set(_failures(small_fleet))
    for aid, a in small_analysis.assets.items():
        if aid not in failing:
            assert a.traces == [], f"{aid} is healthy but raised {len(a.traces)} alert(s)"


def test_alert_closes_when_the_part_is_replaced(small_fleet, small_analysis):
    for aid in _failures(small_fleet):
        for tr in small_analysis.assets[aid].traces:
            if tr.episode.closed_at is not None:
                assert tr.episode.closed_reason in {"maintenance_completed", "recovered"}


def test_the_hypothesis_matches_the_injected_mode_when_one_is_named(small_fleet, small_analysis):
    named = wrong = 0
    for aid, f in _failures(small_fleet).items():
        for tr in small_analysis.assets[aid].traces[:1]:
            fam = HYPOTHESIS_FAMILY[tr.episode.hypothesis]
            if fam is not None:
                named += 1
                wrong += fam != f.mode
    assert named >= 1 and wrong == 0


def test_a_work_order_is_issued_mid_degradation_with_evidence_and_a_rul_interval(
    small_fleet, small_runs
):
    f = next(iter(small_fleet.truth.failures))
    run = small_runs[f.asset_id]
    as_of = pd.Timestamp(f.failure) - pd.Timedelta(days=4)
    # the small fleet has no object-level obs/exp columns for a truncated frame: use the full run
    a = analyse_run(run, AnalysisConfig(), None, as_of=as_of)
    assert a.work_orders, "an open alert should produce a work order"
    wo = a.work_orders[0]
    assert wo.synthetic and wo.asset_id == f.asset_id
    assert wo.severity.rank >= Severity.WARNING.rank
    assert wo.evidence, "a work order without evidence is not actionable"
    assert wo.rul is not None
    if wo.rul.status == RULStatus.ESTIMATED:
        assert wo.rul.p10_days <= wo.rul.p50_days <= wo.rul.p90_days
        assert wo.latest_safe_date is not None and wo.latest_safe_date <= (
            as_of + pd.Timedelta(days=wo.rul.p10_days + 1e-6)
        )
    assert check_narrative(wo.narrative, wo) == []
    assert wo.narrative_source == "template"


def test_truncating_the_history_does_not_change_earlier_decisions(small_fleet, small_runs):
    """Causality: nothing about the future leaks into a past score."""
    f = next(iter(small_fleet.truth.failures))
    spec = small_fleet.spec(f.asset_id)
    raw = small_fleet.raw[f.asset_id]
    cut = pd.Timestamp(f.onset) + (pd.Timestamp(f.failure) - pd.Timestamp(f.onset)) * 0.5
    part = run_asset(
        raw[pd.to_datetime(raw["timestamp"]) < cut],
        spec,
        [m for m in small_fleet.maintenance if pd.Timestamp(m.timestamp) < cut],
    )
    full = small_runs[f.asset_id]
    idx = part.frame.index[part.frame.index < cut - pd.Timedelta(days=1)]
    for col in ("fused", "cusum", "iforest", "rule"):
        np.testing.assert_allclose(
            full.frame.loc[idx, col].to_numpy(dtype=float),
            part.frame.loc[idx, col].to_numpy(dtype=float),
            equal_nan=True,
            atol=1e-9,
        )
    assert (full.frame.loc[idx, "decision_valid"] == part.frame.loc[idx, "decision_valid"]).all()


def test_bad_data_never_silently_raises_an_alarm(small_fleet, small_runs):
    """Fill a healthy asset with sentinel codes and frozen sensors: the gate must absorb it."""
    failing = set(_failures(small_fleet))
    aid = next(s.asset_id for s in small_fleet.specs if s.asset_id not in failing)
    spec = small_fleet.spec(aid)
    raw = small_fleet.raw[aid].copy()
    rng = np.random.default_rng(0)
    n = len(raw)
    sensors = [c for c in spec.channels]
    for c in sensors:
        hit = rng.random(n) < 0.04
        raw.loc[hit, c] = rng.choice([9999.0, 65535.0, -999.0], size=int(hit.sum()))
    for c in sensors[:4]:
        start = int(n * 0.6)
        raw.loc[start : start + 300, c] = raw[c].iloc[start - 1]  # frozen transmitter
    run = run_asset(raw, spec, small_fleet.maintenance)
    a = analyse_run(run, AnalysisConfig(), None)
    assert a.traces == [], "data faults were turned into an alarm"


def test_an_asset_with_unusable_data_is_blocked_not_scored(small_fleet):
    f = small_fleet.specs[0]
    raw = small_fleet.raw[f.asset_id].copy()
    raw["load_pct"] = np.nan
    run = run_asset(raw, f, small_fleet.maintenance)
    assert run.blocked and run.frame["fused"].isna().all()
    assert analyse_run(run, AnalysisConfig(), None).traces == []


def test_pipeline_is_deterministic(small_fleet):
    spec = next(s for s in small_fleet.specs if s.asset_type.value == "conveyor_pulley")
    a = run_asset(small_fleet.raw[spec.asset_id], spec, small_fleet.maintenance)
    b = run_asset(small_fleet.raw[spec.asset_id], spec, small_fleet.maintenance)
    pd.testing.assert_frame_equal(a.frame, b.frame)


def test_analysis_lists_work_orders_most_severe_first(small_analysis):
    wos = small_analysis.all_work_orders()
    ranks = [w.severity.rank for w in wos]
    assert ranks == sorted(ranks, reverse=True)


def test_synthetic_label_travels_with_every_output(small_analysis):
    for wo in small_analysis.all_work_orders():
        assert wo.synthetic is True
    assert small_analysis.truth is not None and small_analysis.truth.synthetic


def test_as_of_replays_a_past_moment_without_using_the_future(small_fleet, small_runs):
    from asset_health.analysis import analyse_runs

    f = next(iter(small_fleet.truth.failures))
    as_of = pd.Timestamp(f.failure) - pd.Timedelta(days=8)
    fa = analyse_runs(small_runs, AnalysisConfig(), None, as_of=as_of)
    assert fa.as_of == as_of
    for a in fa.assets.values():
        for tr in a.traces:
            assert pd.Timestamp(tr.episode.opened_at) <= as_of
            assert tr.episode.closed_at is None or pd.Timestamp(tr.episode.closed_at) <= as_of
            assert all(c.t <= as_of for c in tr.checkpoints)
    open_now = [w.asset_id for w in fa.all_work_orders()]
    assert f.asset_id in open_now, "an alert should be open 8 days before the injected failure"
