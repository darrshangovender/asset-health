from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from asset_health.catalog import PROFILES
from asset_health.models import (
    HYPOTHESIS_FAMILY,
    SYNTHETIC_NOTICE,
    AlertEpisode,
    AssetSpec,
    AssetType,
    BearingGeometry,
    DQStatus,
    FailureEvent,
    FailureMode,
    FaultHypothesis,
    GroundTruth,
    RULEstimate,
    RULStatus,
    Severity,
)

T0 = datetime(2025, 1, 1)


def test_bearing_frequencies_match_hand_calculation():
    # 8 balls, 9.5 mm ball on a 38 mm pitch circle, zero contact angle: d/D = 0.25
    b = BearingGeometry(n_elements=8, ball_diameter_mm=9.5, pitch_diameter_mm=38.0)
    assert b.bpfo_order == pytest.approx(4 * 0.75)
    assert b.bpfi_order == pytest.approx(4 * 1.25)
    assert b.ftf_order == pytest.approx(0.5 * 0.75)
    assert b.bsf_order == pytest.approx(38.0 / (2 * 9.5) * (1 - 0.25**2))


def test_bearing_identity_bpfo_plus_bpfi_equals_n():
    for t in PROFILES.values():
        b = t.bearing
        assert b.bpfo_order + b.bpfi_order == pytest.approx(b.n_elements)


def test_bearing_contact_angle_lowers_bpfo():
    flat = BearingGeometry(n_elements=14, ball_diameter_mm=40, pitch_diameter_mm=210)
    angled = BearingGeometry(
        n_elements=14, ball_diameter_mm=40, pitch_diameter_mm=210, contact_angle_deg=30
    )
    assert angled.bpfo_order > flat.bpfo_order  # cos(a) < 1 shrinks d/D cos a, raising (1 - ..)


def test_bearing_rejects_ball_larger_than_pitch():
    with pytest.raises(ValidationError):
        BearingGeometry(n_elements=8, ball_diameter_mm=50, pitch_diameter_mm=40)


def test_bearing_rejects_too_few_elements():
    with pytest.raises(ValidationError):
        BearingGeometry(n_elements=2, ball_diameter_mm=5, pitch_diameter_mm=40)


def test_models_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        BearingGeometry(n_elements=8, ball_diameter_mm=9.5, pitch_diameter_mm=38, bogus=1)


def test_asset_spec_nominal_rpm():
    p = PROFILES[AssetType.SLURRY_PUMP]
    spec = AssetSpec(
        asset_id="P",
        asset_type=p.asset_type,
        name="x",
        shaft_hz=p.shaft_hz,
        bearing=p.bearing,
        channels=p.channels,
    )
    assert spec.nominal_rpm == pytest.approx(p.shaft_hz * 60)


def test_failure_event_requires_failure_after_onset():
    with pytest.raises(ValidationError):
        FailureEvent(
            asset_id="A",
            mode=FailureMode.BEARING_SPALL,
            onset=T0,
            failure=T0,
            shape_k=2,
            amplitude=1,
            downtime_hours=48,
        )


def test_failure_event_duration_days():
    f = FailureEvent(
        asset_id="A",
        mode=FailureMode.OVERHEATING,
        onset=T0,
        failure=T0 + timedelta(days=12),
        shape_k=2,
        amplitude=1,
        downtime_hours=48,
    )
    assert f.duration_days == pytest.approx(12.0)


def test_ground_truth_is_always_synthetic():
    gt = GroundTruth(seed=1, failures=[], maintenance=[], data_faults=[])
    assert gt.synthetic is True
    assert "SYNTHETIC" in gt.notice
    with pytest.raises(ValidationError):
        GroundTruth(synthetic=False, seed=1, failures=[], maintenance=[], data_faults=[])


def test_synthetic_notice_does_not_claim_real_data():
    assert "not measured data" in SYNTHETIC_NOTICE
    assert "any real operator" in SYNTHETIC_NOTICE


def test_severity_ordering():
    order = [Severity.NORMAL, Severity.WATCH, Severity.WARNING, Severity.URGENT, Severity.CRITICAL]
    assert [s.rank for s in order] == sorted(s.rank for s in order)
    assert len({s.rank for s in order}) == 5


def test_dq_status_ranking():
    assert DQStatus.PASS.rank < DQStatus.WARN.rank < DQStatus.FAIL.rank


def test_every_hypothesis_has_a_family_entry():
    assert set(HYPOTHESIS_FAMILY) == set(FaultHypothesis)
    assert HYPOTHESIS_FAMILY[FaultHypothesis.UNEXPLAINED] is None


def test_rul_estimate_requires_full_interval_when_estimated():
    with pytest.raises(ValidationError):
        RULEstimate(asset_id="A", as_of=T0, status=RULStatus.ESTIMATED, p50_days=5.0)


def test_rul_estimate_rejects_unordered_quantiles():
    with pytest.raises(ValidationError):
        RULEstimate(
            asset_id="A", as_of=T0, status=RULStatus.ESTIMATED, p10_days=9, p50_days=5, p90_days=12
        )


def test_rul_estimate_accepts_ordered_quantiles_and_non_estimates():
    ok = RULEstimate(
        asset_id="A", as_of=T0, status=RULStatus.ESTIMATED, p10_days=2, p50_days=5, p90_days=12
    )
    assert ok.p10_days <= ok.p50_days <= ok.p90_days
    none = RULEstimate(asset_id="A", as_of=T0, status=RULStatus.NOT_DEGRADING, note="flat")
    assert none.p50_days is None


def test_alert_episode_defaults():
    e = AlertEpisode(asset_id="A", opened_at=T0, closed_at=None, peak_severity=Severity.WARNING)
    assert e.hypothesis == FaultHypothesis.UNEXPLAINED
    assert e.urgent_at is None


def test_bearing_geometry_survives_a_json_round_trip_with_its_computed_orders():
    """Regression: manifest.json carries bpfo_order etc.; reading it back used to fail with
    'extra inputs are not permitted', which broke every CLI command on a written dataset."""
    b = BearingGeometry(
        n_elements=14, ball_diameter_mm=40, pitch_diameter_mm=210, contact_angle_deg=12
    )
    text = b.model_dump_json()
    assert "bpfo_order" in text
    again = BearingGeometry.model_validate_json(text)
    assert again == b and again.bpfo_order == pytest.approx(b.bpfo_order)


def test_serialised_orders_are_not_trusted_as_inputs():
    b = BearingGeometry.model_validate(
        {"n_elements": 8, "ball_diameter_mm": 9.5, "pitch_diameter_mm": 38.0, "bpfo_order": 99.0}
    )
    assert b.bpfo_order == pytest.approx(
        3.0
    )  # recomputed from the geometry, the stale value ignored


def test_fleet_manifest_round_trips_through_json(tiny_fleet):
    from asset_health.models import FleetManifest

    text = tiny_fleet.manifest.model_dump_json()
    assert FleetManifest.model_validate_json(text) == tiny_fleet.manifest
