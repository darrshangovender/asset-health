from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asset_health.alerting.cost import CostModel
from asset_health.config import PipelineConfig, PolicyConfig, RULConfig, SimConfig
from asset_health.evaluate import BenchConfig, assert_disjoint, save_results
from asset_health.evaluate.artifact import AssetArtifact, SeedArtifact, config_key, healthy_hours
from asset_health.evaluate.baselines import (
    age_based_replacements,
    fixed_threshold_episodes,
)
from asset_health.evaluate.costing import CostBreakdown, charge_alarm_policy, charge_failure
from asset_health.evaluate.events import FailureOutcome, near_data_fault, score_asset
from asset_health.evaluate.policy_eval import evaluate_calendar, evaluate_policy
from asset_health.evaluate.rul_eval import constant_baseline, rul_metrics
from asset_health.evaluate.runner import with_fused
from asset_health.evaluate.stats import bootstrap_ci, mean_sd, quartiles, wilson
from asset_health.models import (
    AlertEpisode,
    AssetType,
    DataFaultEvent,
    FailureEvent,
    FailureMode,
    GroundTruth,
    Severity,
)

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2025, 1, 1)


def day(n: float) -> datetime:
    return T0 + timedelta(days=n)


def ep(open_day: float, close_day: float | None = None, asset="A") -> AlertEpisode:
    return AlertEpisode(
        asset_id=asset,
        opened_at=day(open_day),
        closed_at=None if close_day is None else day(close_day),
        peak_severity=Severity.WARNING,
    )


def truth(onset=50, failure=80, asset="A", faults=None) -> GroundTruth:
    return GroundTruth(
        seed=1,
        failures=[
            FailureEvent(
                asset_id=asset,
                mode=FailureMode.BEARING_SPALL,
                race="outer",
                onset=day(onset),
                failure=day(failure),
                shape_k=3,
                amplitude=1,
                downtime_hours=48,
            )
        ]
        if onset is not None
        else [],
        maintenance=[],
        data_faults=faults or [],
    )


# ----------------------------------------------------------------------------- stats


def test_wilson_interval_known_values():
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=0.005) and hi == pytest.approx(0.943, abs=0.005)
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.7 < lo < 0.75


def test_wilson_interval_contains_the_point_estimate():
    for k in range(0, 21):
        lo, hi = wilson(k, 20)
        assert lo <= k / 20 <= hi


def test_mean_sd_ignores_nan_and_handles_one_value():
    assert mean_sd([1.0, 3.0, float("nan")]) == {
        "mean": 2.0,
        "sd": pytest.approx(1.4142, abs=1e-3),
        "n": 2,
    }
    assert mean_sd([5.0])["sd"] == 0.0
    assert mean_sd([])["n"] == 0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a, b = bootstrap_ci(v), bootstrap_ci(v)
    assert a == b and a[0] < 3.5 < a[1]
    assert bootstrap_ci([1.0])[0] != bootstrap_ci([1.0])[0]


def test_quartiles():
    q = quartiles([1, 2, 3, 4, 5])
    assert q == {"median": 3.0, "q25": 2.0, "q75": 4.0}


# ----------------------------------------------------------------------------- event scoring


def test_alert_inside_the_window_is_a_detection_with_lead_time():
    sc = score_asset(1, "A", [ep(60)], truth())
    (o,) = sc.outcomes
    assert o.detected and o.lead_days == pytest.approx(20.0)
    assert sc.false_alarms == []
    assert o.actionable(3.0) and not o.actionable(30.0)


def test_alert_before_onset_is_a_false_alarm_and_the_failure_is_missed():
    sc = score_asset(1, "A", [ep(20, 30)], truth())
    assert not sc.outcomes[0].detected
    assert len(sc.false_alarms) == 1


def test_alert_after_the_failure_is_a_false_alarm():
    sc = score_asset(1, "A", [ep(85)], truth())
    assert not sc.outcomes[0].detected and len(sc.false_alarms) == 1


def test_second_episode_in_the_same_window_is_a_realert_not_a_false_alarm():
    sc = score_asset(1, "A", [ep(60, 62), ep(70)], truth())
    assert sc.outcomes[0].n_realerts == 1
    assert sc.false_alarms == []
    assert sc.outcomes[0].lead_days == pytest.approx(20.0)  # measured from the first


def test_alert_on_an_asset_that_never_fails_is_a_false_alarm():
    sc = score_asset(1, "B", [ep(40, asset="B")], truth(asset="A"))
    assert sc.outcomes == [] and len(sc.false_alarms) == 1


def test_no_alerts_on_a_failing_asset_is_a_miss():
    sc = score_asset(1, "A", [], truth())
    assert (
        len(sc.outcomes) == 1 and not sc.outcomes[0].detected and sc.outcomes[0].lead_days is None
    )


def test_near_data_fault_flags_alarms_during_injected_faults():
    gt = truth(
        faults=[
            DataFaultEvent(
                asset_id="A", kind="stuck", channel="vib_rms", start=day(10), end=day(11)
            )
        ]
    )
    assert near_data_fault(gt, "A", pd.Timestamp(day(10.5)))
    assert near_data_fault(gt, "A", pd.Timestamp(day(12.5)))  # within the 48 h tail
    assert not near_data_fault(gt, "A", pd.Timestamp(day(20)))
    assert not near_data_fault(gt, "B", pd.Timestamp(day(10.5)))


# ----------------------------------------------------------------------------- costing


def test_early_replacement_costs_planned_plus_wasted_life():
    out = CostBreakdown()
    c = CostModel()
    charge_failure(out, c, AssetType.SLURRY_PUMP, pd.Timestamp(day(80)), pd.Timestamp(day(70)))
    assert out.planned == c.planned_replacement and out.unplanned == 0
    assert out.early_loss == pytest.approx(c.early_loss_per_day * 10)


def test_late_or_missing_replacement_costs_an_unplanned_failure():
    c = CostModel()
    a, b = CostBreakdown(), CostBreakdown()
    charge_failure(a, c, AssetType.SLURRY_PUMP, pd.Timestamp(day(80)), pd.Timestamp(day(81)))
    charge_failure(b, c, AssetType.SLURRY_PUMP, pd.Timestamp(day(80)), None)
    assert a.unplanned == b.unplanned == c.unplanned_failure and a.planned == 0


def test_every_alert_is_inspected_and_false_alarms_are_charged_the_same():
    c = CostModel()
    sc = score_asset(1, "A", [ep(60), ep(10, 12)], truth())
    out = CostBreakdown()
    charge_alarm_policy(
        out,
        c,
        AssetType.SLURRY_PUMP,
        sc.outcomes,
        len(sc.false_alarms),
        lambda o: pd.Timestamp(o.episode.opened_at) + pd.Timedelta(days=3),
    )
    assert out.n_inspections == 2 and out.inspections == pytest.approx(2 * c.false_alarm)
    assert out.n_planned == 1


def test_cost_breakdown_adds_and_totals():
    a = CostBreakdown(inspections=1, planned=2, early_loss=3, unplanned=4, n_planned=1)
    b = CostBreakdown(inspections=10, planned=20, early_loss=30, unplanned=40, n_planned=2)
    s = a + b
    assert s.total == 110 and s.n_planned == 3
    assert s.as_dict()["total"] == 110


def test_heavier_assets_cost_more_to_lose():
    c = CostModel()
    a, b = CostBreakdown(), CostBreakdown()
    charge_failure(a, c, AssetType.PRIMARY_CRUSHER, pd.Timestamp(day(80)), None)
    charge_failure(b, c, AssetType.CONVEYOR_IDLER, pd.Timestamp(day(80)), None)
    assert a.total > 10 * b.total


# ----------------------------------------------------------------------------- baselines


def _artifact(naive: pd.DataFrame, repairs=()) -> AssetArtifact:
    from asset_health.catalog import PROFILES, spec_from_profile

    spec = spec_from_profile(PROFILES[AssetType.SLURRY_PUMP], "A", "A")
    frame = pd.DataFrame({"decision_valid": True, "fused": 0.0}, index=naive.index)
    return AssetArtifact(
        spec=spec,
        frame=frame,
        z24=pd.DataFrame(index=naive.index),
        repairs=list(repairs),
        services=[],
        blocked=False,
        dq_overall="pass",
        dq_usable_fraction=1.0,
        naive=naive,
        baseline_hours=48,
    )


def _naive(n=24 * 80, jump_at=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T0, periods=n, freq="h").as_unit("ns")
    vib = 3 + rng.normal(0, 0.1, n)
    if jump_at is not None:
        vib[jump_at:] += 2.0
    return pd.DataFrame({"vib_rms": vib, "temp_c": 60 + rng.normal(0, 0.5, n)}, index=idx)


def test_fixed_threshold_alarms_on_a_step_and_not_on_a_quiet_signal():
    quiet = fixed_threshold_episodes(_artifact(_naive()), k=5.0)
    stepped = fixed_threshold_episodes(_artifact(_naive(jump_at=24 * 50)), k=5.0)
    assert quiet == []
    assert len(stepped) == 1
    assert pd.Timestamp(stepped[0].opened_at) >= pd.Timestamp(T0) + pd.Timedelta(days=50)


def test_fixed_threshold_cannot_alarm_while_its_limit_is_being_set():
    n = _naive()
    n.iloc[:48, 0] += 10  # inside the settle period
    assert fixed_threshold_episodes(_artifact(n), k=5.0) == []


def test_fixed_threshold_has_no_regime_awareness_a_busy_shift_trips_it():
    n = _naive()
    n.iloc[24 * 40 : 24 * 41, 0] += 1.5  # a day of higher load
    assert fixed_threshold_episodes(_artifact(n), k=5.0)


def test_fixed_threshold_does_not_filter_a_stuck_high_reading():
    n = _naive()
    n.iloc[24 * 40 : 24 * 45, 0] = 9.0  # frozen transmitter
    assert fixed_threshold_episodes(_artifact(n), k=5.0)


def _days(ts):
    return [round((t - pd.Timestamp(T0)).total_seconds() / 86400) for t in ts]


def test_age_based_replacement_is_due_when_the_part_reaches_the_age_limit():
    # a part 10 days old with a 30-day age limit: due on day 20, then every 30 days
    r, prevented = age_based_replacements(10.0, 30.0, pd.Timestamp(day(0)), pd.Timestamp(day(120)))
    assert _days(r) == [20, 50, 80, 110] and prevented is None


def test_age_based_replacement_of_an_overdue_part_happens_on_day_zero():
    r, _ = age_based_replacements(400.0, 30.0, pd.Timestamp(day(0)), pd.Timestamp(day(70)))
    assert _days(r) == [0, 30, 60]


def test_a_replacement_before_the_failure_prevents_it_even_before_onset():
    # F1 regression: a new part does not fail on the old part's schedule. The old rule only
    # credited replacements inside [onset, failure); one at day 20, before onset (50), prevented
    # nothing although the worn part had been removed.
    f = truth(onset=50, failure=80).failures[0]
    r, prevented = age_based_replacements(
        10.0, 30.0, pd.Timestamp(day(0)), pd.Timestamp(day(120)), f
    )
    assert prevented == pd.Timestamp(day(20))
    assert _days(r) == [20, 50, 80, 110]


def test_a_failure_before_the_due_date_happens_and_the_clock_restarts_at_the_repair():
    f = truth(onset=50, failure=80).failures[0]
    r, prevented = age_based_replacements(
        0.0, 100.0, pd.Timestamp(day(0)), pd.Timestamp(day(200)), f, pd.Timestamp(day(82))
    )
    assert prevented is None
    assert _days(r) == [182]


# ----------------------------------------------------------------------------- policy evaluation


def _seed_artifact(fused: np.ndarray, with_failure=True) -> SeedArtifact:
    n = len(fused)
    idx = pd.date_range(T0, periods=n, freq="h").as_unit("ns")
    art = _artifact(_naive(n=n))
    art.frame = pd.DataFrame(
        {
            "fused": fused,
            "n_firing": 3,
            "decision_valid": True,
            "hypothesis": "unexplained_anomaly",
            "rule": 0.0,
            "cusum": 0.0,
            "iforest": 0.0,
        },
        index=idx,
    )
    art.z24 = pd.DataFrame(
        {"vib_rms": np.zeros(n), "vib_band_hf": np.zeros(n), "temp_c": np.zeros(n)}, index=idx
    )
    sim = SimConfig(days=max(61, n // 24))
    gt = truth(onset=50, failure=80) if with_failure else truth(onset=None)
    return SeedArtifact(1, sim, gt, {"A": art})


def test_evaluate_policy_detects_a_late_rising_score_with_lead_and_charges_costs():
    n = 24 * 90
    fused = np.zeros(n)
    fused[24 * 60 :] = 2.0
    art = _seed_artifact(fused)
    ev = evaluate_policy(art, PolicyConfig(), RULConfig(), CostModel(), None)
    (o,) = ev.outcomes
    assert o.detected and o.lead_days == pytest.approx(20.0, abs=0.5)
    assert ev.false_alarms == [] and ev.cost.total > 0
    assert ev.cost.n_planned == 1 and ev.cost.n_unplanned == 0


def test_evaluate_policy_misses_a_flat_score_and_charges_the_failure():
    art = _seed_artifact(np.zeros(24 * 90))
    ev = evaluate_policy(art, PolicyConfig(), RULConfig(), CostModel(), None)
    assert not ev.outcomes[0].detected and ev.cost.n_unplanned == 1


def test_healthy_hours_exclude_the_failure_window():
    art = _seed_artifact(np.zeros(24 * 90))
    h = healthy_hours(art.assets["A"], art.truth)
    assert h.sum() == 24 * 90 - (24 * 30 + 1)


def test_false_alarm_rate_uses_healthy_months():
    n = 24 * 90
    fused = np.zeros(n)
    fused[24 * 10 : 24 * 12] = 2.0
    art = _seed_artifact(fused, with_failure=False)
    ev = evaluate_policy(art, PolicyConfig(), RULConfig(), CostModel(), None)
    assert len(ev.false_alarms) == 1
    assert ev.fa_per_asset_month == pytest.approx(1 / (n / 730.5))


def test_calendar_evaluation_counts_replacements_and_prevented_failures():
    art = _seed_artifact(np.zeros(24 * 90))
    # the pump's characteristic life is 700 days: 30/700 of it is a 30-day replacement age
    cal = evaluate_calendar(art, 30.0 / 700.0, CostModel())
    assert cal.n_replacements >= 2 and cal.n_failures == 1
    assert cal.n_prevented == 1  # replaced on day 30, before the day-80 failure


def test_with_fused_changes_the_score_but_not_validity():
    art = _seed_artifact(np.zeros(24 * 10))
    a = art.assets["A"]
    a.frame["cusum"] = 2.0
    a.frame["iforest"] = 0.0
    a.frame["rule"] = 0.0
    only = with_fused(art, {"cusum": 1.0})
    assert only.assets["A"].frame["fused"].iloc[0] == pytest.approx(2.0)
    assert (only.assets["A"].frame["n_firing"] == 1).all()
    assert only.assets["A"].frame["decision_valid"].all()


# ----------------------------------------------------------------------------- RUL metrics


def test_rul_metrics_math_on_a_constructed_table():
    df = pd.DataFrame(
        {
            "seed": [1, 1, 1, 1],
            "asset": ["A"] * 4,
            "true": [2.0, 4.0, 8.0, 20.0],
            "p10": [1.0, 3.0, 9.0, 10.0],
            "p50": [3.0, 4.0, 10.0, 15.0],
            "p90": [5.0, 8.0, 12.0, 25.0],
        }
    )
    m = rul_metrics(df)
    s = m["system"]["all"]["overall"]
    assert s["n"] == 4 and s["mae_days"] == pytest.approx((1 + 0 + 2 + 5) / 4)
    assert s["coverage"] == pytest.approx(0.75)  # the third point (8) is below its p10 (9)
    assert s["mean_width_days"] == pytest.approx((4 + 5 + 3 + 15) / 4)
    assert m["system"]["all"]["true_rul_0_7d"]["n"] == 2


def test_constant_baseline_uses_training_quantiles():
    c = constant_baseline(np.arange(1.0, 101.0))
    assert c.p10 < c.p50 < c.p90 and c.p50 == pytest.approx(50.5)


def test_rul_metrics_on_an_empty_table():
    assert rul_metrics(pd.DataFrame()) == {"n": 0}


# ----------------------------------------------------------------------------- leakage guards


def test_default_benchmark_seed_sets_are_pairwise_disjoint():
    bc = BenchConfig()
    assert_disjoint(bc.dev_seeds, bc.cal_seeds, bc.test_seeds)
    assert not set(bc.test_seeds) & {42, 101}  # the demo seed and the exploration seed


def test_quick_benchmark_seed_sets_are_disjoint_too():
    bc = BenchConfig().quick()
    assert_disjoint(bc.dev_seeds, bc.cal_seeds, bc.test_seeds)


def test_overlapping_seed_sets_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint((1, 2), (3,), (2, 4))
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint((1, 2), (2,), (4,))


def test_cache_key_changes_with_any_config_change():
    sim, pipe = SimConfig(seed=1), PipelineConfig()
    base = config_key(sim, pipe)
    assert config_key(sim, pipe) == base
    assert config_key(sim.model_copy(update={"seed": 2}), pipe) != base
    assert config_key(sim.model_copy(update={"days": 200}), pipe) != base
    p2 = pipe.model_copy(update={"detect": pipe.detect.model_copy(update={"cusum_k": 2.0})})
    assert config_key(sim, p2) != base


def test_shipped_rul_model_never_saw_a_benchmark_test_seed():
    from asset_health.analysis import load_default_rul_model

    model = load_default_rul_model()
    if model is None:
        pytest.skip("no shipped model")
    bc = BenchConfig()
    assert not set(model.train_seeds) & set(bc.test_seeds)
    assert not set(model.cal_seeds) & set(bc.test_seeds)
    assert not set(model.train_seeds) & set(model.cal_seeds)
    assert model.synthetic is True


def test_results_json_serialises_numpy_and_tuples(tmp_path):
    p = tmp_path / "r.json"
    save_results({"a": (1, 2), "b": np.float64(1.5), "c": np.int64(3)}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": [1, 2], "b": 1.5, "c": 3}


def test_failure_outcome_helpers():
    f = truth().failures[0]
    o = FailureOutcome(1, f, ep(60), 20.0)
    assert o.detected and o.mode == FailureMode.BEARING_SPALL and o.actionable(3)
    assert not FailureOutcome(1, f, None, None).actionable(3)


# ----------------------------------------------------------------------------- tuning and rendering


def _row(thr, exp, md, confirm, cost):
    return {
        "alert_threshold": thr,
        "cost_exponent": exp,
        "min_detectors": md,
        "confirm_hours": confirm,
        "mean_cost_per_seed": cost,
    }


def test_select_best_takes_the_cheapest_when_there_is_no_plateau():
    from asset_health.evaluate.tuning import select_best

    table = [_row(1.0, 0.0, 2, 8, 300.0), _row(0.6, 0.0, 1, 4, 400.0), _row(1.25, 0.3, 2, 8, 350.0)]
    assert select_best(table) == table[0]


def test_select_best_prefers_the_simplest_configuration_on_a_plateau():
    from asset_health.evaluate.tuning import select_best

    table = [
        _row(1.25, 0.0, 2, 8, 367.15),  # nominally cheapest, by 0.1 kUSD of noise
        _row(1.0, 0.0, 2, 8, 367.26),
        _row(0.4, 0.3, 2, 8, 367.30),
        _row(0.8, 0.0, 1, 8, 400.0),
    ]
    best = select_best(table)
    assert best["alert_threshold"] == 1.0 and best["cost_exponent"] == 0.0


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    spec = importlib.util.spec_from_file_location("bench_run", ROOT / "benchmarks" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n<!-- RESULTS:BEGIN -->\nold\n<!-- RESULTS:END -->\nafter\n", encoding="utf-8"
    )
    mod.update_readme(readme, "NEW TABLE\n")
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "NEW TABLE" in text
    assert text.startswith("before\n") and text.endswith("after\n")
    nomarkers = tmp_path / "other.md"
    nomarkers.write_text("untouched", encoding="utf-8")
    mod.update_readme(nomarkers, "x")
    assert nomarkers.read_text(encoding="utf-8") == "untouched"


@pytest.mark.skipif(not (ROOT / "benchmarks" / "results.json").exists(), reason="benchmark not run")
def test_markdown_covers_every_section_of_the_results():
    from asset_health.evaluate.markdown import results_to_markdown

    r = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    md = results_to_markdown(r)
    for needle in (
        "Policy comparison",
        "Detection by failure mode",
        "Remaining useful life",
        "Ablations",
        "Policy tuning on the development seeds",
    ):
        assert needle in md


@pytest.mark.skipif(not (ROOT / "benchmarks" / "results.json").exists(), reason="benchmark not run")
def test_results_json_is_internally_consistent():
    r = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    s = r["system"]["overall"]
    assert s["detected"] <= s["n_failures"]
    assert sum(v["n"] for v in r["system"]["by_mode"].values()) == s["n_failures"]
    assert sum(v["detected"] for v in r["system"]["by_mode"].values()) == s["detected"]
    assert s["false_alarms"] == sum(p["false_alarms"] for p in r["system"]["per_seed"])
    assert len(r["system"]["per_seed"]) == len(r["seeds"]["test"])
    rul = r["system"]["rul"]["system"]["decision_time"]["overall"]
    assert 0.0 <= rul["coverage"] <= 1.0 and rul["mae_days"] >= 0
    n_all = r["system"]["rul"]["system"]["all"]["overall"]["n"]
    assert rul["n"] <= n_all  # decision-time rows are a subset of all checkpoints
    sweep = r["severity_sweep"]["rows"]
    assert [row["scale"] for row in sweep] == [1.0, 0.5, 0.3, 0.15]
    assert sweep[0]["system"]["detected"] == s["detected"]
