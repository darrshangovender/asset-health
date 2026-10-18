"""Regression tests for the adversarial review of v0.1.0 (findings F1-F17).

Each test names the finding it pins down. Each one failed on v0.1.0 (several by import error,
because the fix added the API that makes the behaviour testable).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from asset_health.alerting.cost import CostModel
from asset_health.alerting.narrative import check_narrative, template_narrative
from asset_health.analysis import AnalysisConfig, analyse_run, analyse_runs, analyse_source
from asset_health.catalog import PROFILES, spec_from_profile
from asset_health.config import IngestConfig, PolicyConfig, RULConfig, SimConfig
from asset_health.evaluate.artifact import AssetArtifact, SeedArtifact, healthy_hours
from asset_health.evaluate.baselines import fixed_threshold_episodes, fixed_threshold_monitored
from asset_health.evaluate.events import MIN_SEVERITY_AT_OPEN, latent_severity, score_asset
from asset_health.evaluate.markdown import results_to_markdown
from asset_health.evaluate.policy_eval import (
    SeedEval,
    evaluate_calendar,
    evaluate_fixed_threshold,
    evaluate_policy,
)
from asset_health.evaluate.rul_eval import rul_metrics, rul_rows
from asset_health.evaluate.runner import BenchConfig, _decisions, _reference_costs
from asset_health.evaluate.tuning import CALENDAR_FRACTIONS, _near_best
from asset_health.ingest import FrameSource, ingest_asset
from asset_health.ingest.checks import report_as_of
from asset_health.models import (
    AlertEpisode,
    AssetType,
    DQAction,
    DQStatus,
    Evidence,
    FailureEvent,
    FailureMode,
    FaultHypothesis,
    GroundTruth,
    MaintenanceEvent,
    RULEstimate,
    RULStatus,
    Severity,
    WorkOrder,
)
from asset_health.pipeline import run_asset
from asset_health.simulate import generate_fleet

SPEC = spec_from_profile(PROFILES[AssetType.SLURRY_PUMP], "PMP-T", "Test Pump")
T0 = pd.Timestamp("2025-03-01")


def make_raw(n: int = 24 * 30, seed: int = 0, interval_s: int = 3600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols: dict = {"timestamp": pd.date_range(T0, periods=n, freq=f"{interval_s}s")}
    cols["load_pct"] = 60 + rng.normal(0, 5, n)
    cols["speed_rpm"] = SPEC.nominal_rpm * (0.95 + rng.normal(0, 0.01, n))
    cols["ambient_c"] = 25 + rng.normal(0, 2, n)
    base = {"temp_c": 55.0, "current_a": 200.0, "power_kw": 180.0, "pressure_kpa": 400.0}
    base["flow_m3h"] = 700.0
    for c in SPEC.channels:
        cols[c] = base.get(c, 3.0) * (1 + rng.normal(0, 0.03, n))
    return pd.DataFrame(cols)


def check(c, name):
    return next(k for k in c.report.checks if k.name == name)


@pytest.fixture(scope="module")
def pump():
    """One simulated pump that fails mid-record (seed 7, 120 days)."""
    cfg = SimConfig(seed=7, days=120, fleet={AssetType.SLURRY_PUMP: 1}, failure_fraction=1.0)
    return generate_fleet(cfg)


# ------------------------------------------------------------------ F8 export structure (High)


def test_f8_newest_first_export_is_put_back_in_time_order_and_flagged():
    fwd = ingest_asset(make_raw(), SPEC)
    rev = ingest_asset(make_raw().iloc[::-1].reset_index(drop=True), SPEC)
    assert not rev.blocked
    assert check(rev, "row_order").action == DQAction.SORTED
    assert rev.frame.index.equals(fwd.frame.index)  # not re-timed into the future
    pd.testing.assert_frame_equal(rev.frame, fwd.frame)


def test_f8_shuffled_rows_block_the_asset_before_a_grid_is_built():
    raw = make_raw().sample(frac=1.0, random_state=0).reset_index(drop=True)
    c = ingest_asset(raw, SPEC)
    assert c.blocked and len(c.frame) == 0
    assert check(c, "row_order").status == DQStatus.FAIL


def test_f8_ten_minute_export_is_refused_by_default():
    c = ingest_asset(make_raw(n=6 * 24 * 20, interval_s=600), SPEC)
    assert c.blocked
    assert check(c, "cadence").status == DQStatus.FAIL
    assert "600 s" in check(c, "cadence").message


def test_f8_ten_minute_export_is_averaged_onto_the_grid_only_when_asked():
    raw = make_raw(n=6 * 24 * 20, interval_s=600)
    c = ingest_asset(raw, SPEC, IngestConfig(resample=True))
    assert not c.blocked
    assert check(c, "cadence").action == DQAction.RESAMPLED
    assert len(c.frame) == 24 * 20
    assert c.frame.index[-1] <= raw["timestamp"].max()


def test_f8_two_interleaved_collectors_block_the_asset():
    a = make_raw()
    b = a.assign(vib_rms=a["vib_rms"] * 1.01)
    raw = pd.concat([a, b]).sort_index(kind="stable").reset_index(drop=True)
    c = ingest_asset(raw, SPEC)
    assert c.blocked and check(c, "collectors").status == DQStatus.FAIL


def test_f8_a_grid_that_would_run_past_the_data_blocks_the_asset():
    a = make_raw()
    raw = pd.concat([a, a.assign(vib_rms=a["vib_rms"] * 1.01)]).reset_index(drop=True)
    c = ingest_asset(raw, SPEC)  # second copy appended: one backward step, then a replay
    assert c.blocked and check(c, "grid_end").status == DQStatus.FAIL


def test_f8_logger_clock_steps_backwards_are_still_recovered():
    raw = make_raw()
    raw.loc[400:, "timestamp"] = raw.loc[400:, "timestamp"] - pd.Timedelta(minutes=95)
    c = ingest_asset(raw, SPEC)
    assert not c.blocked and check(c, "row_order").status == DQStatus.PASS


@pytest.mark.parametrize("variant", ["reversed", "ten_minute", "resampled"])
def test_f8_no_alert_is_ever_dated_after_the_last_real_sample(pump, variant):
    spec = pump.specs[0]
    raw = pump.raw[spec.asset_id]
    cfg = None
    if variant == "reversed":
        raw = raw.iloc[::-1].reset_index(drop=True)
    else:
        raw = raw.set_index("timestamp").resample("10min").interpolate().reset_index()
        if variant == "resampled":
            from asset_health.config import PipelineConfig

            cfg = PipelineConfig(ingest=IngestConfig(resample=True))
    run = run_asset(raw, spec, pump.maintenance, cfg)
    last = pd.Timestamp(raw["timestamp"].max())
    a = analyse_run(run, AnalysisConfig(), None)
    assert all(pd.Timestamp(t.episode.opened_at) <= last for t in a.traces)
    if variant != "ten_minute":
        assert len(run.frame) and run.frame.index[-1] <= last + pd.Timedelta(hours=24)


# ------------------------------------------------------------------ F9 time zones


def test_f9_tz_aware_maintenance_record_becomes_naive_utc():
    m = MaintenanceEvent(
        asset_id="A", timestamp=datetime(2025, 3, 1, 3, 0, tzinfo=UTC), kind="repair"
    )
    assert m.timestamp.tzinfo is None and m.timestamp == datetime(2025, 3, 1, 3, 0)
    m2 = MaintenanceEvent.model_validate(
        {"asset_id": "A", "timestamp": "2025-03-01T06:00:00+03:00", "kind": "repair"}
    )
    assert m2.timestamp == datetime(2025, 3, 1, 3, 0)


def test_f9_run_asset_accepts_a_utc_aware_repair_against_naive_telemetry():
    ev = [MaintenanceEvent(asset_id="PMP-T", timestamp="2025-03-20T00:00:00Z", kind="repair")]
    run = run_asset(make_raw(), SPEC, ev)  # raised TypeError (naive vs aware) in v0.1.0
    assert run.repairs == [pd.Timestamp("2025-03-20")]


def test_f9_local_time_with_dst_offsets_is_converted_to_utc():
    raw = make_raw(n=24 * 40)
    raw["timestamp"] = pd.date_range("2025-03-10", periods=len(raw), freq="h")  # spans 30 March
    local = raw.copy()
    berlin = raw["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
    local["timestamp"] = berlin.map(pd.Timestamp.isoformat)
    a, b = ingest_asset(raw, SPEC), ingest_asset(local, SPEC)
    assert not b.blocked
    assert a.frame.index.equals(b.frame.index)
    assert b.report.n_gap_samples == 0


# ------------------------------------------------------------------ F11 --as-of uses nothing later


def _outage_after(day: int) -> pd.DataFrame:
    raw = make_raw(n=24 * 60)
    cut = T0 + pd.Timedelta(days=day)
    late = raw["timestamp"] > cut
    keep = ~late | (np.arange(len(raw)) % 20 == 0)  # gateway down: 1 row in 20 after day
    return raw.loc[keep].reset_index(drop=True)


def test_f11_an_outage_after_as_of_does_not_block_the_replay():
    raw = _outage_after(30)
    as_of = T0 + pd.Timedelta(days=25)
    assert ingest_asset(raw, SPEC).blocked  # the whole history is unusable
    replay = ingest_asset(raw, SPEC, as_of=as_of)
    assert not replay.blocked
    assert replay.frame.index[-1] <= as_of


def test_f11_report_as_of_equals_ingesting_the_truncated_history():
    raw = make_raw(n=24 * 60)
    raw.loc[24 * 40 :, "pressure_kpa"] = np.nan  # a channel dies after day 40
    full = ingest_asset(raw, SPEC)
    as_of = T0 + pd.Timedelta(days=35)
    a = report_as_of(full, as_of)
    b = ingest_asset(raw, SPEC, as_of=as_of).report
    assert a.asset_blocked == b.asset_blocked
    for ch in b.channels:
        assert a.channels[ch].valid_fraction == pytest.approx(b.channels[ch].valid_fraction)
    assert a.sensor_advisories == b.sensor_advisories
    assert full.report.sensor_advisories != a.sensor_advisories  # the later dropout is excluded


def test_f11_analyse_source_with_as_of_reads_nothing_later():
    raw = _outage_after(30)
    src = FrameSource([SPEC], {"PMP-T": raw}, [])
    as_of = T0 + pd.Timedelta(days=25)
    fa = analyse_source(src, as_of=as_of, model=None)
    assert not fa.assets["PMP-T"].blocked
    full_runs = {"PMP-T": run_asset(raw, SPEC)}
    with pytest.raises(ValueError, match="Re-run"):
        analyse_runs(full_runs, as_of=as_of)


# ------------------------------------------------------------------ F10 loose accelerometer


def test_f10_a_loose_accelerometer_is_masked_and_raises_no_alert():
    cfg = SimConfig(seed=11, days=120, fleet={AssetType.CONVEYOR_PULLEY: 1}, failure_fraction=0.0)
    fl = generate_fleet(cfg)
    spec = fl.specs[0]
    raw = fl.raw[spec.asset_id].copy()
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2025-03-21 08:00")
    win = ((raw["timestamp"] >= t0) & (raw["timestamp"] < t0 + pd.Timedelta(hours=48))).to_numpy()
    for c in ("vib_rms", "vib_band_low", "vib_band_bpfo", "vib_band_bpfi", "vib_band_hf"):
        f = 2.5 if c == "vib_rms" else 6.0
        raw.loc[win, c] = raw.loc[win, c] * f * np.exp(rng.normal(0, 0.2, win.sum()))
    run = run_asset(raw, spec, fl.maintenance)
    assert check(run.clean, "vibration_sensor").status == DQStatus.WARN
    assert any("accelerometer" in a for a in run.clean.report.sensor_advisories)
    assert analyse_run(run, AnalysisConfig(), None).traces == []  # v0.1.0 opened an alert


def test_f10_a_smooth_rise_in_vibration_is_not_mistaken_for_a_loose_sensor():
    raw = make_raw(n=24 * 40)
    ramp = np.ones(len(raw))
    ramp[24 * 20 :] = np.linspace(1.0, 4.0, len(raw) - 24 * 20)  # genuine wear, x4 over 20 days
    for c in ("vib_rms", "vib_band_hf", "vib_band_bpfo"):
        raw[c] = raw[c] * ramp
    c = ingest_asset(raw, SPEC)
    assert check(c, "vibration_sensor").value == 0


# ------------------------------------------------------------------ F12 narrative guardrail


def _order(severity=Severity.URGENT) -> WorkOrder:
    return WorkOrder(
        work_order_id="WO-PMP-01-20250501",
        asset_id="PMP-01",
        asset_name="SP-01 Cyclone Feed Pump",
        asset_type=AssetType.SLURRY_PUMP,
        issued_at=datetime(2025, 5, 20, 6, 0),
        severity=severity,
        failure_mode_hypothesis=FaultHypothesis.BEARING_OUTER_RACE,
        evidence=[
            Evidence(
                feature="vib_band_bpfo",
                description="outer-race band",
                observed=9.4,
                expected=1.2,
                z_score=11.3,
                unit="(mm/s)^2",
            )
        ],
        recommended_action="Inspect the bearing housing and plan replacement.",
        latest_safe_date=datetime(2025, 5, 27),
        rul=RULEstimate(
            asset_id="PMP-01",
            as_of=datetime(2025, 5, 20),
            status=RULStatus.ESTIMATED,
            p10_days=5.0,
            p50_days=9.0,
            p90_days=16.0,
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        "PMP-01 has an inner race spall.",  # wrong fault, paraphrased
        "Root cause is cavitation and shaft misalignment.",  # faults outside the vocabulary
        "It will fail in about two days; replace within forty-eight hours.",  # number words
        "Median remaining life is 16.0 days, worst case 9.0 days.",  # p90 presented as p50
        "Vibration has risen 20-fold in 5 days; 27 bearings affected.",  # date parts as numbers
        "CRITICAL: trip the pump now.",  # unwarranted severity and escalation
        "Replace it by 30 May 2025.",  # a written-out date that is not in the order
        "The oil is contaminated.",  # the README's own example of a free-text claim
    ],
)
def test_f12_guardrail_rejects_contradictions_v010_accepted(text):
    assert check_narrative(text, _order())


@pytest.mark.parametrize(
    "text",
    [
        "PMP-01 likely has a bearing outer-race defect: vib_band_bpfo is 9.4 against 1.2.",
        "Median remaining life is 9.0 days (80% interval 5.0 to 16.0 days); replace by "
        "27 May 2025.",
    ],
)
def test_f12_guardrail_still_accepts_faithful_text(text):
    assert check_narrative(text, _order()) == []


def test_f12_critical_language_is_allowed_on_a_critical_order():
    wo = _order(Severity.CRITICAL)
    assert check_narrative("CRITICAL: plan the replacement immediately.", wo) == []
    assert check_narrative(template_narrative(wo), wo) == []


# ------------------------------------------------------------------ F1 calendar baseline (High)


def _light_artifacts(fleet, cfg) -> SeedArtifact:
    assets = {}
    for s in fleet.specs:
        reps = [
            pd.Timestamp(m.timestamp)
            for m in fleet.maintenance
            if m.asset_id == s.asset_id and m.kind == "repair"
        ]
        assets[s.asset_id] = AssetArtifact(
            spec=s,
            frame=pd.DataFrame(),
            z24=pd.DataFrame(),
            repairs=reps,
            services=[],
            blocked=False,
            dq_overall="pass",
            dq_usable_fraction=1.0,
            naive=pd.DataFrame(),
        )
    return SeedArtifact(cfg.seed, cfg, fleet.truth, assets)


def test_f1_components_have_ages_and_old_parts_fail_early(small_fleet):
    ages = {s.asset_id: s.component_age_days for s in small_fleet.specs}
    assert all(a >= 0 for a in ages.values()) and len(set(ages.values())) > 1
    start = small_fleet.manifest.start
    for f in small_fleet.truth.failures:
        _, eta = PROFILES[small_fleet.spec(f.asset_id).asset_type].life_weibull
        life = ages[f.asset_id] + (f.failure - start).total_seconds() / 86400.0
        assert 0 < life < 6 * eta  # age at failure is a plausible Weibull life


def test_f1_age_based_replacement_can_beat_run_to_failure_on_wear_out_parts(small_fleet):
    art = _light_artifacts(small_fleet, SimConfig(seed=7, days=170))
    costs = CostModel()
    rtf = _reference_costs([art], costs)["run_to_failure"]["mean"]
    best = min(evaluate_calendar(art, fr, costs).cost.total for fr in CALENDAR_FRACTIONS)
    # v0.1.0 (memoryless failures, credit only inside [onset, failure)): 0.85-1.0 x run to
    # failure on this fleet; with wear-out lives and ages it is well under 0.6
    assert best < 0.6 * rtf


def test_f1_component_ages_do_not_change_the_simulated_signals(monkeypatch):
    import asset_health.simulate.fleet as fleet_mod

    cfg = SimConfig(seed=5, days=90, fleet={AssetType.SLURRY_PUMP: 1}, failure_fraction=1.0)
    with_ages = generate_fleet(cfg, keep_latent=False)
    monkeypatch.setattr(
        fleet_mod, "component_ages", lambda rng, c, specs, f: {s.asset_id: 0.0 for s in specs}
    )
    without = generate_fleet(cfg, keep_latent=False)
    pd.testing.assert_frame_equal(with_ages.raw["PMP-01"], without.raw["PMP-01"])
    assert with_ages.truth == without.truth
    assert with_ages.specs[0].component_age_days > 0 == without.specs[0].component_age_days


# ------------------------------------------------------------------ F2 severity sweep (High)


def test_f2_degradation_scale_shrinks_the_signature_but_keeps_the_timing():
    base = SimConfig(seed=3, days=90, fleet={AssetType.SLURRY_PUMP: 1}, failure_fraction=1.0)
    full = generate_fleet(base)
    weak = generate_fleet(base.model_copy(update={"degradation_scale": 0.3}))
    (f1,), (f3,) = full.truth.failures, weak.truth.failures
    assert f1.failure == f3.failure and f1.onset == f3.onset
    assert f3.amplitude == pytest.approx(0.3 * f1.amplitude)
    lat_full, lat_weak = full.latent["PMP-01"], weak.latent["PMP-01"]
    idx = lat_full.index
    t = idx[idx.searchsorted(pd.Timestamp(f1.failure) - pd.Timedelta(days=2))]
    assert lat_full.loc[t, "severity"] == pytest.approx(lat_weak.loc[t, "severity"])
    hc = "vib_band_hf" if f1.mode == FailureMode.BEARING_SPALL else "vib_band_proc"
    healthy = lat_full[hc].iloc[24 * 23 : 24 * 30].median()
    assert lat_weak.loc[t, hc] - healthy < lat_full.loc[t, hc] - healthy


def test_f2_the_benchmark_runs_the_sweep_on_held_out_seeds():
    bc = BenchConfig()
    assert set(bc.severity_scales) == {0.5, 0.3, 0.15}
    assert bc.quick().severity_scales  # the smoke run exercises it too


def _sweep_results() -> dict:
    def cell(act, cost):
        return {
            "n_failures": 10,
            "detected": 10,
            "actionable": act,
            "median_lead_days": 5.0,
            "false_alarms": 1,
            "false_alarms_per_asset_month": 0.01,
            "cost_per_seed": {"mean": cost, "sd": 1.0},
        }

    ms = {"mean": 100.0, "sd": 1.0}
    rows = [
        {
            "scale": 1.0,
            "system": cell(10, 100),
            "fixed_threshold": cell(9, 200),
            "fixed_threshold_rise": cell(9, 210),
            "calendar_cost_per_seed": ms,
            "run_to_failure_cost_per_seed": ms,
        },
        {
            "scale": 0.15,
            "system": cell(3, 300),
            "fixed_threshold": cell(5, 250),
            "fixed_threshold_rise": cell(4, 260),
            "calendar_cost_per_seed": ms,
            "run_to_failure_cost_per_seed": ms,
        },
    ]
    return {"seeds": [1, 2], "rows": rows}


def test_f2_markdown_states_where_the_system_stops_beating_the_limits():
    from asset_health.evaluate.markdown import _severity_sweep

    r = {"severity_sweep": _sweep_results(), "system": {"overall": {"detected_with_3d_lead": 1}}}
    text = "\n".join(_severity_sweep(r))
    assert "Detectability vs failure severity" in text
    assert "costs less than the cheaper fixed limit at 1x" in text
    assert "does **not** at 0.15x" in text
    assert "a fixed limit warns at least 3d ahead on more failures at 0.15x" in text


# ------------------------------------------------------------------ F3 coincidental alarms


def _failure(onset=50, failure=80, k=3.0) -> FailureEvent:
    d0 = datetime(2025, 1, 1)
    return FailureEvent(
        asset_id="A",
        mode=FailureMode.LINER_WEAR,
        onset=d0 + timedelta(days=onset),
        failure=d0 + timedelta(days=failure),
        shape_k=k,
        amplitude=1.0,
        downtime_hours=48,
    )


def _ep(day: float) -> AlertEpisode:
    return AlertEpisode(
        asset_id="A",
        opened_at=datetime(2025, 1, 1) + timedelta(days=day),
        closed_at=datetime(2025, 1, 1) + timedelta(days=day + 1),
        peak_severity=Severity.WARNING,
    )


def test_f3_an_alarm_before_the_fault_is_physically_there_is_not_a_detection():
    f = _failure()
    gt = GroundTruth(seed=1, failures=[f], maintenance=[], data_faults=[])
    assert latent_severity(f, pd.Timestamp(_ep(51).opened_at)) < MIN_SEVERITY_AT_OPEN
    sc = score_asset(1, "A", [_ep(51)], gt)
    assert not sc.outcomes[0].detected
    assert len(sc.coincidental) == 1 and len(sc.false_alarms) == 1
    sc2 = score_asset(1, "A", [_ep(51), _ep(70)], gt)
    assert sc2.outcomes[0].detected and sc2.outcomes[0].lead_days == pytest.approx(10.0)
    assert len(sc2.coincidental) == 1


# ------------------------------------------------------------------ F4 / F5 fixed limits


def _naive_artifact(n_days=120, seasonal=True) -> AssetArtifact:
    rng = np.random.default_rng(0)
    n = 24 * n_days
    idx = pd.date_range("2025-01-01", periods=n, freq="h").as_unit("ns")
    ambient = 20 + (15 * np.arange(n) / n if seasonal else 0.0)  # winter to summer
    temp = ambient + 30 + rng.normal(0, 0.5, n)
    naive = pd.DataFrame(
        {
            "vib_rms": 3 + rng.normal(0, 0.1, n),
            "temp_c": temp,
            "temp_rise": temp - ambient,
        },
        index=idx,
    )
    frame = pd.DataFrame({"decision_valid": True, "fused": 0.0, "n_firing": 0}, index=idx)
    frame.iloc[: n // 2, 0] = False  # the system was not deciding for half the record
    return AssetArtifact(
        spec=SPEC,
        frame=frame,
        z24=pd.DataFrame(index=idx),
        repairs=[],
        services=[],
        blocked=False,
        dq_overall="pass",
        dq_usable_fraction=1.0,
        naive=naive,
        baseline_hours=48,
    )


def test_f5_the_temperature_rise_variant_does_not_alarm_on_the_season():
    art = _naive_artifact()
    raw = fixed_threshold_episodes(art, {"vib_rms": 5.0, "temp_c": 5.0})
    rise = fixed_threshold_episodes(
        art, {"vib_rms": 5.0, "temp_rise": 5.0}, channels=("vib_rms", "temp_rise")
    )
    assert raw and not rise


def test_f4_the_fixed_limit_false_alarm_rate_uses_its_own_watched_hours():
    art = _naive_artifact()
    gt = GroundTruth(seed=1, failures=[], maintenance=[], data_faults=[])
    sim = SimConfig(days=120)
    ev = evaluate_fixed_threshold(SeedArtifact(1, sim, gt, {"A": art}), 5.0, CostModel())
    watched = fixed_threshold_monitored(art)
    assert ev.healthy_months == pytest.approx(watched.sum() / 730.5)
    assert ev.healthy_months > healthy_hours(art, gt).sum() / 730.5  # not the system's hours


# ------------------------------------------------------------------ F6 / F7 RUL at decision time


def test_f6_f7_rul_is_scored_at_decision_time_and_decisions_are_reported():
    from asset_health.alerting.policy import Checkpoint, EpisodeTrace
    from asset_health.evaluate.events import FailureOutcome

    f = _failure(onset=50, failure=80)
    ep = _ep(60)
    tr = EpisodeTrace(episode=ep)
    for d in range(60, 80):
        t = pd.Timestamp(datetime(2025, 1, 1) + timedelta(days=d))
        est = RULEstimate(
            asset_id="A",
            as_of=t.to_pydatetime(),
            status=RULStatus.ESTIMATED,
            p10_days=1.0,
            p50_days=10.0,
            p90_days=30.0,
        )
        tr.checkpoints.append(Checkpoint(t, est, Severity.WARNING, 5.0, False))
    tr.action_at = pd.Timestamp(ep.opened_at) + pd.Timedelta(days=2)
    tr.execution_at = tr.action_at + pd.Timedelta(days=3)
    ev = SeedEval(seed=1, outcomes=[FailureOutcome(1, f, ep, 20.0)])
    ev.traces[("A", pd.Timestamp(ep.opened_at))] = tr
    df = rul_rows([ev], {(1, "A"): pd.DataFrame()}, RULConfig(), with_trend_only=False)
    assert df["decision"].sum() == 6  # days 60-65: up to the planned replacement
    m = rul_metrics(df)
    assert m["system"]["decision_time"]["overall"]["n"] == 6
    assert m["system"]["all"]["overall"]["n"] == 20
    assert m["system"]["first_estimate"]["overall"]["n"] == 1
    dec = _decisions([ev])
    assert dec["decision_delay_days"]["median"] == pytest.approx(2.0)
    assert dec["replaced_before_failure"] == 1 and dec["acted_at_first_checkpoint"] == 0


# ------------------------------------------------------------------ F16 symmetric tie-break


def test_f16_baselines_use_the_same_one_percent_plateau_as_the_system():
    rows = [
        {"mean_cost_per_seed": 100.0, "false_alarms": 50},
        {"mean_cost_per_seed": 100.9, "false_alarms": 5},
        {"mean_cost_per_seed": 102.0, "false_alarms": 0},
    ]
    near = _near_best(rows)
    assert near == rows[:2]
    assert min(near, key=lambda r: r["false_alarms"]) == rows[1]


# ------------------------------------------------------------------ policy evaluation still works


def test_evaluate_policy_counts_coincidental_alarms_separately():
    n = 24 * 90
    idx = pd.date_range("2025-01-01", periods=n, freq="h").as_unit("ns")
    art = _naive_artifact(n_days=90)
    fused = np.zeros(n)
    fused[24 * 51 : 24 * 52] = 2.0  # coincidental: latent severity < 0.05
    fused[24 * 70 :] = 2.0
    art.frame = pd.DataFrame(
        {
            "fused": fused,
            "n_firing": 3,
            "decision_valid": True,
            "hypothesis": "unexplained_anomaly",
            "rule": 0.0,
        },
        index=idx,
    )
    art.z24 = pd.DataFrame({"vib_rms": np.zeros(n)}, index=idx)
    gt = GroundTruth(seed=1, failures=[_failure()], maintenance=[], data_faults=[])
    ev = evaluate_policy(
        SeedArtifact(1, SimConfig(days=90), gt, {"A": art}),
        PolicyConfig(),
        RULConfig(),
        CostModel(),
        None,
    )
    assert ev.outcomes[0].detected and ev.n_coincidental == 1


def test_markdown_renders_the_new_sections_from_a_results_dict():
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "benchmarks" / "results.json"
    if not p.exists():
        pytest.skip("benchmark not run")
    r = json.loads(p.read_text(encoding="utf-8"))
    if "severity_sweep" not in r:
        pytest.skip("results.json predates the severity sweep")
    md = results_to_markdown(r)
    for needle in (
        "Detectability vs failure severity",
        "temperature rise over ambient",
        "age-based replacement",
        "Remaining useful life at decision time",
        "When the system acted",
        "more than one family",
    ):
        assert needle in md


def _debounce_by_hour(over, confirm, clear):
    """The hour-by-hour state machine of v0.1.0, kept as the reference."""
    out, run, below, open_i = [], 0, 0, None
    for i, v in enumerate(over):
        if open_i is None:
            run = run + 1 if v else 0
            if run >= confirm:
                open_i, below = i, 0
        else:
            below = below + 1 if not v else 0
            if below >= clear:
                out.append((open_i, i))
                open_i, run = None, 0
    if open_i is not None:
        out.append((open_i, None))
    return out


@pytest.mark.parametrize("seed", range(40))
def test_run_based_debounce_matches_the_hourly_state_machine(seed):
    from asset_health.evaluate.baselines import debounce

    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.9)
    over = rng.random(int(rng.integers(0, 400))) < p
    confirm, clear = int(rng.integers(1, 6)), int(rng.integers(1, 30))
    assert debounce(over, confirm, clear) == _debounce_by_hour(over, confirm, clear)


# ------------------------------------------------------------------ F13 / F14 / F15 / F17 wording


def _readme_section(name: str) -> str:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    return text.split(f"## {name}")[1].split("\n## ")[0]


def test_f17_the_rul_interval_is_called_conservative_not_calibrated():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "calibrated interval" not in text and "is calibrated, not decorative" not in text
    assert "conservative" in text


def test_f13_f14_limitations_name_the_physics_and_monitoring_caveats():
    lim = _readme_section("Limitations")
    for needle in ("20-32x order band", "tachometer", "journal bearings", "idlers", "telematics",
                   "throatbush", "liner-profile", "moves every band together"):  # fmt: skip
        assert needle in lim, needle


def test_f7_f15_design_decisions_say_what_is_inert_and_what_is_a_lookup():
    dd = _readme_section("Design decisions")
    assert "cost exponent to 0" in dd and "first checkpoint" in dd
    assert "multi-family asset types" in dd
