from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from asset_health.detect import (
    DEFAULT_RULES,
    best_hypothesis,
    cusum_arrays,
    cusum_frame,
    rule_strengths,
)
from asset_health.detect.ensemble import hold_mask
from asset_health.detect.iforest import fit_forest, one_sided
from asset_health.models import AssetType, FaultHypothesis


def test_cusum_is_zero_for_a_quiet_signal():
    out = cusum_arrays(np.zeros(200), k=1.0, h=10.0, two_sided=False)
    assert np.all(out == 0.0)


def test_cusum_ignores_shifts_below_the_reference_value():
    out = cusum_arrays(np.full(500, 0.8), k=1.0, h=10.0, two_sided=False)
    assert out.max() == 0.0


def test_cusum_crosses_the_decision_interval_on_a_sustained_shift():
    z = np.concatenate([np.zeros(50), np.full(100, 3.0)])
    out = cusum_arrays(z, k=1.0, h=20.0, two_sided=False)
    first = int(np.argmax(out >= 1.0))
    assert 55 <= first <= 65  # 20 / (3 - 1) = 10 samples after the shift
    assert out[:50].max() == 0.0


def test_cusum_one_sided_ignores_downward_shifts_two_sided_does_not():
    z = np.full(100, -3.0)
    assert cusum_arrays(z, 1.0, 20.0, two_sided=False).max() == 0.0
    assert cusum_arrays(z, 1.0, 20.0, two_sided=True).max() >= 1.0


def test_cusum_nan_holds_state_instead_of_resetting_or_alarming():
    z = np.concatenate([np.full(10, 3.0), np.full(30, np.nan), np.zeros(5)])
    out = cusum_arrays(z, 1.0, 100.0, two_sided=False)
    assert out[10] == out[39] > 0  # unchanged across the gap
    assert np.ptp(out[10:40]) == pytest.approx(0.0, abs=1e-12)


def test_cusum_reset_zeroes_the_state():
    z = np.full(30, 3.0)
    reset = np.zeros(30, dtype=bool)
    reset[15] = True
    out = cusum_arrays(z, 1.0, 1000.0, False, reset)
    assert out[14] > out[15] - 1e-12
    assert out[15] == pytest.approx(2.0 / 1000.0)


def test_cusum_state_is_capped_so_it_can_recover():
    z = np.concatenate([np.full(1000, 5.0), np.zeros(200)])
    out = cusum_arrays(z, 1.0, 10.0, False, cap=3.0)
    assert out.max() == pytest.approx(3.0)
    assert out[-1] == 0.0  # winds down in bounded time


def test_cusum_matrix_matches_scalar_version():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, (300, 3))
    z[50:80, 1] = np.nan
    frame = pd.DataFrame(z, columns=["vib_rms", "pressure_kpa", "temp_c"])
    got = cusum_frame(frame, 1.0, 15.0)
    for c, two in (("vib_rms", False), ("pressure_kpa", True), ("temp_c", False)):
        np.testing.assert_allclose(got[c], cusum_arrays(frame[c].to_numpy(), 1.0, 15.0, two))


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-6, 6, allow_nan=False), min_size=1, max_size=200))
def test_cusum_is_bounded_and_nonnegative(values):
    out = cusum_arrays(np.array(values), 1.0, 10.0, True)
    assert np.all(out >= 0) and np.all(out <= 3.0 + 1e-9)


# ----------------------------------------------------------------------------- rules


def z24(**cols) -> pd.DataFrame:
    n = 5
    base = {
        c: np.zeros(n)
        for c in (
            "vib_band_bpfo",
            "bpfo_share",
            "vib_band_bpfi",
            "bpfi_share",
            "vib_band_hf",
            "hf_share",
            "temp_c",
            "vib_band_low",
            "vib_band_proc",
            "proc_share",
            "pressure_kpa",
            "flow_m3h",
            "power_kw",
            "current_a",
        )
    }
    base.update({k: np.full(n, v) for k, v in cols.items()})
    return pd.DataFrame(base)


def hyp(frame, atype):
    s = rule_strengths(frame, atype)
    _, h, _ = best_hypothesis(s)
    return FaultHypothesis(h.iloc[-1])


def test_no_residual_means_no_rule_fires():
    s = rule_strengths(z24(), AssetType.SLURRY_PUMP)
    assert (s == 0).all().all()
    _, h, name = best_hypothesis(s)
    assert (h == FaultHypothesis.UNEXPLAINED.value).all() and (name == "").all()


def test_outer_race_signature_names_the_outer_race():
    assert hyp(z24(vib_band_bpfo=9.0, vib_band_hf=4.0), AssetType.CONVEYOR_PULLEY) == (
        FaultHypothesis.BEARING_OUTER_RACE
    )


def test_inner_race_signature_names_the_inner_race():
    assert hyp(z24(vib_band_bpfi=9.0, bpfi_share=6.0), AssetType.SLURRY_PUMP) == (
        FaultHypothesis.BEARING_INNER_RACE
    )


def test_pressure_drop_with_vane_pass_rise_names_impeller_wear():
    assert hyp(z24(pressure_kpa=-6.0, vib_band_proc=6.0), AssetType.SLURRY_PUMP) == (
        FaultHypothesis.IMPELLER_WEAR
    )


def test_impeller_rule_does_not_apply_to_a_conveyor():
    s = rule_strengths(z24(pressure_kpa=-6.0, vib_band_proc=6.0), AssetType.CONVEYOR_IDLER)
    assert "impeller_wear" not in s.columns


def test_power_drop_names_liner_wear_on_a_crusher_only():
    assert hyp(z24(power_kw=-6.0, vib_band_proc=5.0), AssetType.PRIMARY_CRUSHER) == (
        FaultHypothesis.LINER_WEAR
    )
    s = rule_strengths(z24(power_kw=-6.0), AssetType.SLURRY_PUMP)
    assert "liner_wear" not in s.columns


def test_temperature_alone_names_overheating():
    assert hyp(z24(temp_c=7.0), AssetType.HAUL_TRUCK_ENGINE) == FaultHypothesis.OVERHEATING


def test_rule_strength_opens_at_the_gate_and_corroboration_raises_it():
    eng = AssetType.HAUL_TRUCK_ENGINE
    at = rule_strengths(z24(temp_c=4.0), eng)["overheating"].iloc[-1]
    over = rule_strengths(z24(temp_c=8.0), eng)["overheating"].iloc[-1]
    below = rule_strengths(z24(temp_c=3.9), eng)["overheating"].iloc[-1]
    backed = rule_strengths(z24(temp_c=4.0, vib_band_hf=3.0, hf_share=3.0), eng)[
        "overheating"
    ].iloc[-1]
    assert below == 0.0
    assert at == pytest.approx(0.8)  # gate just open, nothing corroborating
    assert backed == pytest.approx(1.0)  # corroborated by the high-frequency evidence
    assert over > backed


def test_every_rule_has_a_hypothesis_and_an_explanation():
    for r in DEFAULT_RULES:
        assert r.explanation and r.any_of
        assert r.hypothesis != FaultHypothesis.UNEXPLAINED


def test_an_unrelated_channel_does_not_trigger_a_race_rule():
    s = rule_strengths(z24(vib_band_low=9.0), AssetType.SLURRY_PUMP)
    assert s["outer_race_defect"].iloc[-1] == 0.0


# ----------------------------------------------------------------------------- forest


def test_forest_needs_enough_baseline():
    z = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)), columns=list("abcd"))
    assert fit_forest(z, 20, 0.99, 1.1, 0) is None


def test_forest_scores_outliers_above_the_baseline():
    rng = np.random.default_rng(0)
    cols = ["vib_rms", "vib_band_hf", "temp_c", "power_kw"]
    train = pd.DataFrame(rng.normal(0, 1, (400, 4)), columns=cols)
    scorer = fit_forest(train, 40, 0.99, 1.1, 0)
    assert scorer is not None
    normal = scorer.score(pd.DataFrame(rng.normal(0, 1, (100, 4)), columns=cols)).median()
    odd = scorer.score(pd.DataFrame(np.full((20, 4), 8.0), columns=cols)).median()
    assert odd > 1.0 > normal


def test_one_sided_floors_harmless_directions_only():
    z = pd.DataFrame({"vib_rms": [-3.0, 2.0], "pressure_kpa": [-3.0, 2.0]})
    out = one_sided(z)
    assert out["vib_rms"].tolist() == [0.0, 2.0]
    assert out["pressure_kpa"].tolist() == [-3.0, 2.0]


def test_forest_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    cols = ["vib_rms", "vib_band_hf", "temp_c"]
    train = pd.DataFrame(rng.normal(0, 1, (300, 3)), columns=cols)
    a = fit_forest(train, 20, 0.99, 1.1, 3).score(train).to_numpy()
    b = fit_forest(train, 20, 0.99, 1.1, 3).score(train).to_numpy()
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------- ensemble


def test_hold_mask_marks_a_window_after_each_service():
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    hold, reset = hold_mask(idx, [idx[50]], 10)
    assert reset.sum() == 1 and reset[50]
    assert hold[50:61].all() and not hold[49] and not hold[61]


def test_not_ready_features_give_a_silent_detection_frame(small_runs):
    # a fresh run with an asset whose features are not ready must never carry a decision
    for run in small_runs.values():
        pre = run.frame.loc[run.frame["decision_valid"].eq(False), "fused"]
        assert pre.isna().all()


def test_decisions_are_only_made_after_the_baseline(small_runs):
    for run in small_runs.values():
        for seg in run.segments:
            f = run.frame.loc[seg.start : seg.end]
            late = f[f["decision_valid"]]
            assert (late.index >= seg.features.baseline_end).all()


def test_fused_score_is_a_bounded_weighted_sum(small_runs):
    for run in small_runs.values():
        fused = run.frame["fused"].dropna()
        assert (fused >= 0).all() and (fused <= 3.0 + 1e-9).all()
