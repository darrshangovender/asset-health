from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from asset_health.alerting import (
    CostModel,
    LogNormalRUL,
    check_narrative,
    find_episodes,
    latest_safe_days,
    narrate,
    template_narrative,
    threshold_for,
)
from asset_health.alerting.cost import expected_cost_of_action_at, optimal_action_day
from asset_health.alerting.llm import AnthropicNarrator
from asset_health.alerting.narrative import NarrativeGenerator
from asset_health.alerting.policy import grade
from asset_health.catalog import PROFILES
from asset_health.config import PolicyConfig
from asset_health.models import (
    AssetType,
    Evidence,
    FaultHypothesis,
    RULEstimate,
    RULStatus,
    Severity,
    WorkOrder,
)

# ----------------------------------------------------------------------------- cost model


def test_costs_scale_with_asset_class_but_inspection_does_not():
    base = CostModel()
    crusher = base.scaled(AssetType.PRIMARY_CRUSHER)
    idler = base.scaled(AssetType.CONVEYOR_IDLER)
    assert crusher.unplanned_failure == pytest.approx(base.unplanned_failure * 4.0)
    assert idler.unplanned_failure == pytest.approx(base.unplanned_failure * 0.25)
    assert crusher.false_alarm == base.false_alarm == idler.false_alarm


def test_failure_penalty_is_the_extra_cost_over_a_planned_job():
    c = CostModel(planned_replacement=10, unplanned_failure=90)
    assert c.failure_penalty == 80


def test_lognormal_from_quantiles_reproduces_them():
    d = LogNormalRUL.from_quantiles(4.0, 10.0, 25.0)
    assert d.ppf(0.5) == pytest.approx(10.0, rel=1e-6)
    assert d.ppf(0.1) == pytest.approx(4.0, rel=0.2)
    assert d.ppf(0.9) == pytest.approx(25.0, rel=0.2)
    assert d.cdf(10.0) == pytest.approx(0.5, abs=1e-6)


def test_lognormal_cdf_is_vectorised_and_zero_at_nonpositive_days():
    d = LogNormalRUL(mu=2.0, sigma=0.5)
    arr = d.cdf(np.array([0.0, 1.0, 100.0]))
    assert arr[0] == 0.0 and arr[1] < 0.01 and arr[2] > 0.99


def test_expected_excess_matches_monte_carlo():
    d = LogNormalRUL(mu=math.log(10.0), sigma=0.5)
    rng = np.random.default_rng(0)
    r = np.exp(rng.normal(d.mu, d.sigma, 400_000))
    for day in (0.0, 3.0, 10.0, 25.0):
        assert d.expected_excess(day) == pytest.approx(np.maximum(r - day, 0).mean(), rel=0.02)


def test_expected_cost_trades_early_loss_against_failure_risk():
    d = LogNormalRUL(mu=math.log(10.0), sigma=0.5)
    c = CostModel()
    early = expected_cost_of_action_at(0.0, d, c)  # replace now: throw away the whole life
    late = expected_cost_of_action_at(60.0, d, c)  # wait: almost certainly fails first
    mid = expected_cost_of_action_at(1.5, d, c)  # just ahead of the lower tail
    assert mid < early and mid < late


def test_optimal_day_moves_earlier_as_failure_gets_costlier():
    d = LogNormalRUL(mu=math.log(20.0), sigma=0.6)
    cheap = optimal_action_day(
        d, CostModel(unplanned_failure=20, planned_replacement=15, early_loss_per_day=2)
    )
    dear = optimal_action_day(
        d, CostModel(unplanned_failure=2000, planned_replacement=15, early_loss_per_day=2)
    )
    assert dear < cheap


def test_optimal_day_moves_later_as_waste_gets_costlier():
    d = LogNormalRUL(mu=math.log(20.0), sigma=0.6)
    low = optimal_action_day(d, CostModel(early_loss_per_day=0.05))
    high = optimal_action_day(d, CostModel(early_loss_per_day=3.0))
    assert high >= low


def test_latest_safe_days_never_exceeds_the_pessimistic_end_or_goes_negative():
    for p10, p50, p90 in ((1, 3, 9), (5, 12, 30), (0.0, 0.5, 2.0)):
        d = latest_safe_days(p10, p50, p90, CostModel())
        assert 0.0 <= d <= p10 + 1e-9


@settings(max_examples=40, deadline=None)
@given(st.floats(0.2, 20), st.floats(1.0, 3.0), st.floats(1.0, 3.0), st.floats(0.5, 50))
def test_latest_safe_days_is_bounded_for_any_interval(p10, a, b, penalty):
    p50, p90 = p10 * a, p10 * a * b
    c = CostModel(unplanned_failure=15 + penalty)
    d = latest_safe_days(p10, p50, p90, c)
    assert 0 <= d <= p10 + 1e-9


# ----------------------------------------------------------------------------- episodes


def frame(fused, valid=None, n_firing=None, start="2025-05-01"):
    n = len(fused)
    idx = pd.date_range(start, periods=n, freq="h")
    fused = np.asarray(fused, dtype=float)
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return pd.DataFrame(
        {
            "fused": np.where(valid, fused, np.nan),
            "n_firing": np.full(n, 3) if n_firing is None else np.asarray(n_firing),
            "decision_valid": valid,
            "hypothesis": FaultHypothesis.UNEXPLAINED.value,
            "rule": 0.0,
        },
        index=idx,
    )


CFG = PolicyConfig(confirm_hours=6, clear_hours=24, min_detectors=1)


def test_a_short_blip_does_not_open_an_alert():
    f = frame([0.1] * 50 + [2.0] * 4 + [0.1] * 50)
    assert find_episodes(f, "A", 1.0, CFG) == []


def test_an_alert_opens_only_after_the_debounce_completes():
    f = frame([0.1] * 50 + [2.0] * 40)
    (ep,) = find_episodes(f, "A", 1.0, CFG)
    assert pd.Timestamp(ep.opened_at) == f.index[55]  # 6th consecutive hour above
    assert pd.Timestamp(ep.first_exceeded_at) == f.index[50]
    assert ep.closed_at is None and ep.peak_score == pytest.approx(2.0)


def test_hysteresis_a_dip_between_clear_and_alert_levels_keeps_the_alert_open():
    f = frame([2.0] * 20 + [0.8] * 60 + [2.0] * 5)  # 0.8 is below alert (1.0) but above clear (0.6)
    eps = find_episodes(f, "A", 1.0, CFG)
    assert len(eps) == 1 and eps[0].closed_at is None


def test_alert_closes_only_after_the_clear_period():
    f = frame([2.0] * 20 + [0.2] * 23 + [2.0] * 5 + [0.2] * 30)
    eps = find_episodes(f, "A", 1.0, CFG)
    assert len(eps) == 1  # the 23 h dip is shorter than clear_hours=24
    assert eps[0].closed_reason == "recovered"


def test_two_separate_excursions_make_two_episodes():
    f = frame([2.0] * 20 + [0.1] * 60 + [2.0] * 20 + [0.1] * 40)
    assert len(find_episodes(f, "A", 1.0, CFG)) == 2


def test_hours_without_data_neither_open_nor_close_an_alert():
    fused = [2.0] * 10 + [0.0] * 100 + [0.1] * 10
    valid = [True] * 10 + [False] * 100 + [True] * 10
    f = frame(fused, valid)
    (ep,) = find_episodes(f, "A", 1.0, CFG)
    assert ep.closed_at is None  # silence is not recovery


def test_a_long_data_outage_cannot_create_an_alert():
    f = frame([0.0] * 200, valid=[False] * 200)
    assert find_episodes(f, "A", 1.0, CFG) == []


def test_invalid_hours_do_not_count_toward_the_debounce():
    fused = [2.0] * 3 + [9.0] * 50 + [2.0] * 3
    valid = [True] * 3 + [False] * 50 + [True] * 3
    assert len(find_episodes(frame(fused, valid), "A", 1.0, CFG)) == 1  # 3 + 3 valid hours = 6


def test_min_detectors_requires_agreement():
    f = frame([2.0] * 40, n_firing=[1] * 40)
    assert find_episodes(f, "A", 1.0, CFG.model_copy(update={"min_detectors": 2})) == []
    assert len(find_episodes(f, "A", 1.0, CFG.model_copy(update={"min_detectors": 1}))) == 1


def test_a_repair_closes_an_open_alert():
    f = frame([2.0] * 80)
    repair = f.index[60]
    ep = find_episodes(f, "A", 1.0, CFG, [repair])[0]
    assert ep.closed_reason == "maintenance_completed"
    assert pd.Timestamp(ep.closed_at) == repair


def test_score_does_not_re_alert_straight_after_a_repair_until_debounced():
    f = frame([2.0] * 80)
    eps = find_episodes(f, "A", 1.0, CFG, [f.index[40]])
    assert len(eps) == 2
    assert pd.Timestamp(eps[1].opened_at) >= f.index[40] + timedelta(hours=5)


def test_threshold_scales_with_the_cost_of_a_miss():
    cfg = PolicyConfig(alert_threshold=1.0, cost_exponent=0.3)
    assert threshold_for(AssetType.PRIMARY_CRUSHER, cfg) < threshold_for(AssetType.SLURRY_PUMP, cfg)
    assert threshold_for(AssetType.CONVEYOR_IDLER, cfg) > threshold_for(AssetType.SLURRY_PUMP, cfg)
    flat = PolicyConfig(alert_threshold=1.0, cost_exponent=0.0)
    assert threshold_for(AssetType.PRIMARY_CRUSHER, flat) == 1.0
    assert PROFILES[AssetType.PRIMARY_CRUSHER].cost_multiplier == 4.0


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(0, 3, allow_nan=False), min_size=10, max_size=250),
    st.floats(0.5, 2.0),
)
def test_episodes_are_ordered_non_overlapping_and_bounded_by_crossings(values, thr):
    f = frame(values)
    eps = find_episodes(f, "A", thr, CFG)
    last_close = None
    for e in eps:
        assert e.closed_at is None or e.closed_at >= e.opened_at
        if last_close is not None:
            assert e.opened_at >= last_close
        last_close = e.closed_at
        assert e.peak_score >= thr
    n_above = int((np.array(values) >= thr).sum())
    assert len(eps) <= n_above // CFG.confirm_hours + 1


# ----------------------------------------------------------------------------- severity grading


def est(p10, p50, p90, status=RULStatus.ESTIMATED):
    return RULEstimate(
        asset_id="A",
        as_of=datetime(2025, 5, 1),
        status=status,
        p10_days=p10,
        p50_days=p50,
        p90_days=p90,
    )


def test_grading_follows_the_rul_interval():
    c = CostModel().scaled(AssetType.SLURRY_PUMP)
    cfg = PolicyConfig()
    assert grade(est(0.5, 1.5, 4.0), c, cfg)[0] == Severity.CRITICAL
    sev, lsd, act = grade(est(3.0, 6.0, 10.0), c, cfg)
    assert sev == Severity.URGENT and act and lsd is not None
    assert grade(est(40.0, 60.0, 90.0), c, cfg)[0] == Severity.WARNING


def test_grading_without_an_interval_stays_at_warning_and_does_not_act():
    c = CostModel()
    r = RULEstimate(asset_id="A", as_of=datetime(2025, 5, 1), status=RULStatus.NOT_DEGRADING)
    assert grade(r, c, PolicyConfig()) == (Severity.WARNING, None, False)


# ----------------------------------------------------------------------------- narrative


def make_wo(**kw) -> WorkOrder:
    base = dict(
        work_order_id="WO-PMP-01-20250501",
        asset_id="PMP-01",
        asset_name="SP-01 Cyclone Feed Pump",
        asset_type=AssetType.SLURRY_PUMP,
        issued_at=datetime(2025, 5, 20, 6, 0),
        severity=Severity.URGENT,
        failure_mode_hypothesis=FaultHypothesis.BEARING_OUTER_RACE,
        alternative_hypotheses=[FaultHypothesis.BEARING_GENERIC],
        evidence=[
            Evidence(
                feature="vib_band_bpfo",
                description="outer-race band",
                observed=9.4,
                expected=1.2,
                z_score=11.3,
                unit="(mm/s)^2",
            ),
            Evidence(
                feature="temp_c",
                description="temperature",
                observed=71.5,
                expected=64.0,
                z_score=5.2,
                unit="degC",
            ),
        ],
        recommended_action="Inspect the bearing housing and plan replacement.",
        latest_safe_date=datetime(2025, 5, 27),
        rul=est(5.0, 9.0, 16.0),
    )
    base.update(kw)
    return WorkOrder(**base)


def test_template_narrative_passes_its_own_guardrail():
    wo = make_wo()
    assert check_narrative(template_narrative(wo), wo) == []


def test_template_narrative_cites_only_structured_facts():
    text = template_narrative(make_wo())
    assert "vib_band_bpfo" in text and "9.0 days" in text and "2025-05-27" in text
    assert "80% interval 5.0 to 16.0" in text


def test_guardrail_rejects_an_invented_number():
    wo = make_wo()
    problems = check_narrative("Vibration is 31.7 mm/s, far above normal.", wo)
    assert any("31.7" in p for p in problems)


def test_guardrail_rejects_an_invented_date():
    wo = make_wo()
    assert any("2025-06-30" in p for p in check_narrative("Replace by 2025-06-30.", wo))


def test_guardrail_rejects_a_feature_that_is_not_evidence():
    wo = make_wo()
    assert any("pressure_kpa" in p for p in check_narrative("pressure_kpa has dropped.", wo))


def test_guardrail_rejects_a_fault_that_is_not_a_hypothesis():
    wo = make_wo()
    assert check_narrative("This looks like impeller wear.", wo)
    assert not check_narrative("This looks like a bearing outer-race defect.", wo)


def test_guardrail_accepts_numbers_that_are_in_the_object():
    wo = make_wo()
    assert check_narrative("vib_band_bpfo is 9.4 against 1.2 expected, z 11.3.", wo) == []


class GoodLLM:
    def generate(self, work_order: WorkOrder) -> str:
        return (
            "Pump SP-01 likely has a bearing outer-race defect; "
            "vib_band_bpfo is 9.4 vs 1.2 expected."
        )


class LyingLLM:
    def generate(self, work_order: WorkOrder) -> str:
        return "The bearing will fail on 2025-06-30 with vibration at 55 mm/s."


class BrokenLLM:
    def generate(self, work_order: WorkOrder) -> str:
        raise TimeoutError("API unavailable")


def test_protocol_is_runtime_checkable():
    assert isinstance(GoodLLM(), NarrativeGenerator)
    assert not isinstance(object(), NarrativeGenerator)


def test_narrate_default_is_the_deterministic_template():
    wo = narrate(make_wo())
    assert wo.narrative_source == "template" and wo.narrative
    assert narrate(make_wo()).narrative == wo.narrative


def test_narrate_uses_an_llm_text_only_if_it_passes_the_guardrail():
    wo = narrate(make_wo(), GoodLLM())
    assert wo.narrative_source == "llm" and "bearing outer-race defect" in wo.narrative


def test_narrate_falls_back_when_the_llm_invents_facts():
    wo = narrate(make_wo(), LyingLLM())
    assert wo.narrative_source == "template_after_guardrail"
    assert "55" not in wo.narrative and "2025-06-30" not in wo.narrative


def test_narrate_falls_back_when_the_llm_is_down():
    wo = narrate(make_wo(), BrokenLLM())
    assert wo.narrative_source == "template_after_guardrail" and wo.narrative


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs

        class Block:
            text = "Bearing outer-race defect on PMP-01."

        class Msg:
            content = (Block(),)

        return Msg()


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_anthropic_narrator_sends_only_the_structured_order():
    client = FakeClient()
    n = AnthropicNarrator("some-model", client=client)
    text = n.generate(make_wo())
    assert text.startswith("Bearing outer-race")
    sent = client.messages.kwargs
    assert sent["model"] == "some-model" and sent["temperature"] == 0
    assert "narrative" not in sent["messages"][0]["content"]
    assert "vib_band_bpfo" in sent["messages"][0]["content"]


def test_anthropic_narrator_requires_an_explicit_model():
    with pytest.raises(ValueError):
        AnthropicNarrator("", client=FakeClient())


@settings(max_examples=40, deadline=None)
@given(
    st.floats(0.1, 500),
    st.floats(0.1, 500),
    st.floats(-30, 30),
    st.floats(0.5, 30),
    st.floats(1.0, 3.0),
    st.floats(1.0, 3.0),
)
def test_template_always_passes_the_guardrail(obs, exp, z, p10, a, b):
    wo = make_wo(
        evidence=[
            Evidence(
                feature="vib_rms",
                description="rms",
                observed=obs,
                expected=exp,
                z_score=z,
                unit="mm/s",
            )
        ],
        rul=est(p10, p10 * a, p10 * a * b),
    )
    assert check_narrative(template_narrative(wo), wo) == []


def test_numbers_in_the_data_quality_note_and_action_are_allowed_in_the_narrative():
    """Regression: the template quotes the data-quality note, whose counts ('27 sensor advisory
    items') the guardrail used to reject, so a legitimate narrative fell back for no reason."""
    wo = make_wo(
        data_quality_note="27 sensor advisory item(s); affected readings were masked, not used",
        recommended_action="Re-inspect within 72 hours.",
    )
    text = template_narrative(wo)
    assert "27 sensor advisory" in text and "72 hours" in text
    assert check_narrative(text, wo) == []
    assert check_narrative("The oil analysis shows 31 ppm iron.", wo)  # still rejects the invented
