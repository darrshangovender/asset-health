from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from asset_health.config import RULConfig
from asset_health.models import RULStatus
from asset_health.rul import (
    RULModel,
    RunToFailure,
    estimate_rul,
    fit_rul_model,
    fit_trend,
    health_index,
    top_contributors,
    trend_rul,
)
from asset_health.rul.estimate import estimate_from_fit
from asset_health.rul.health_index import harmful_residuals
from asset_health.rul.model import conformal_margin, feature_vector, fit_quantile_model
from asset_health.rul.trend import theil_sen

CFG = RULConfig()
T0 = pd.Timestamp("2025-03-01")


def hourly(values, start=T0) -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), name="hi")


def ramp_hi(days=40, onset_day=10, slope=0.4, noise=0.0, seed=0) -> pd.Series:
    """Healthy near 0.3, then a linear rise of ``slope`` per day."""
    n = days * 24
    t = np.arange(n) / 24.0
    rng = np.random.default_rng(seed)
    v = 0.3 + np.clip(t - onset_day, 0, None) * slope + rng.normal(0, noise, n)
    return hourly(np.clip(v, 0, None))


# ----------------------------------------------------------------------------- health index


def test_harmful_residuals_orient_by_channel_direction():
    z = pd.DataFrame({"vib_rms": [-4.0, 4.0], "pressure_kpa": [-4.0, 4.0]})
    h = harmful_residuals(z)
    assert h["vib_rms"].tolist() == [0.0, 4.0]  # a quieter machine is not worse
    assert h["pressure_kpa"].tolist() == [4.0, 4.0]  # a pressure drop is


def test_health_index_is_mean_of_top_k_harmful_residuals():
    z = pd.DataFrame(
        {"vib_rms": [9.0], "vib_band_hf": [6.0], "temp_c": [3.0], "vib_band_low": [0.5]}
    )
    assert health_index(z, top_k=3).iloc[0] == pytest.approx(6.0)


def test_health_index_needs_k_trustworthy_channels():
    z = pd.DataFrame({"vib_rms": [9.0], "vib_band_hf": [np.nan], "temp_c": [np.nan]})
    assert np.isnan(health_index(z, top_k=3).iloc[0])


def test_health_index_caps_extreme_values():
    z = pd.DataFrame({"vib_rms": [500.0], "vib_band_hf": [500.0], "temp_c": [500.0]})
    assert health_index(z, top_k=3, cap=40.0).iloc[0] == pytest.approx(40.0)


def test_health_index_of_an_empty_frame_is_empty():
    assert len(health_index(pd.DataFrame())) == 0


def test_top_contributors_lists_largest_first():
    row = pd.Series({"vib_rms": 2.0, "temp_c": 9.0, "pressure_kpa": -5.0})
    names = [n for n, _ in top_contributors(row, 2)]
    assert names == ["temp_c", "pressure_kpa"]


# ----------------------------------------------------------------------------- theil-sen


def test_theil_sen_recovers_an_exact_line():
    x = np.arange(9.0)
    m, lo, hi = theil_sen(x, 2.0 * x + 1.0, 0.8)
    assert m == pytest.approx(2.0) and lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_theil_sen_ignores_one_wild_outlier():
    x = np.arange(9.0)
    y = 2.0 * x + 1.0
    y[4] += 100
    assert theil_sen(x, y, 0.8)[0] == pytest.approx(2.0, abs=0.2)


def test_theil_sen_interval_brackets_the_median_and_widens_with_noise():
    x = np.arange(9.0)
    rng = np.random.default_rng(0)
    quiet = theil_sen(x, x + rng.normal(0, 0.05, 9), 0.8)
    noisy = theil_sen(x, x + rng.normal(0, 1.5, 9), 0.8)
    assert quiet[1] <= quiet[0] <= quiet[2]
    assert (noisy[2] - noisy[1]) > (quiet[2] - quiet[1])


@settings(max_examples=40, deadline=None)
@given(st.floats(-5, 5), st.floats(-20, 20))
def test_theil_sen_is_exact_on_any_noise_free_line(slope, intercept):
    x = np.arange(8.0)
    m, lo, hi = theil_sen(x, slope * x + intercept, 0.8)
    assert m == pytest.approx(slope, abs=1e-9)
    assert lo <= m + 1e-9 <= hi + 2e-9


# ----------------------------------------------------------------------------- trend


def test_fit_trend_measures_the_slope_of_a_ramp():
    hi = ramp_hi(slope=0.5)
    fit = fit_trend(hi, T0 + pd.Timedelta(days=30), CFG)
    assert fit.ok and fit.slope == pytest.approx(0.5, abs=0.02)
    assert fit.hi_now == pytest.approx(0.3 + 20 * 0.5, abs=0.2)
    assert fit.slope_lo <= fit.slope <= fit.slope_hi


def test_fit_trend_is_causal_future_values_are_ignored():
    hi = ramp_hi(slope=0.5)
    t = T0 + pd.Timedelta(days=25)
    a = fit_trend(hi, t, CFG)
    spoiled = hi.copy()
    spoiled.loc[spoiled.index > t] = 999.0
    b = fit_trend(spoiled, t, CFG)
    assert a == b


def test_fit_trend_without_recent_data_is_not_ok():
    hi = ramp_hi()
    hi.iloc[400:] = np.nan
    fit = fit_trend(hi, T0 + pd.Timedelta(days=30), CFG)
    assert not fit.ok and "no recent" in fit.reason


def test_fit_trend_needs_enough_daily_samples():
    hi = ramp_hi(days=40)
    hi.iloc[: 24 * 28] = np.nan  # only ~3 days of history before the checkpoint
    fit = fit_trend(hi, T0 + pd.Timedelta(days=31), CFG)
    assert not fit.ok and "daily samples" in fit.reason


def test_age_counts_days_since_the_index_was_last_healthy():
    hi = ramp_hi(onset_day=10, slope=0.6)
    fit = fit_trend(hi, T0 + pd.Timedelta(days=30), CFG)
    assert fit.age_days == pytest.approx(30 - (10 + (1.0 - 0.3) / 0.6), abs=1.0)


def test_trend_rul_is_none_for_a_flat_or_healthy_index():
    flat = fit_trend(hourly(np.full(24 * 30, 3.0)), T0 + pd.Timedelta(days=25), CFG)
    assert trend_rul(flat, 9.0, 2.0, CFG) is None
    healthy = fit_trend(ramp_hi(slope=0.01), T0 + pd.Timedelta(days=30), CFG)
    assert trend_rul(healthy, 9.0, 2.0, CFG) is None


def test_trend_rul_quantiles_are_ordered_and_near_the_geometry():
    hi = ramp_hi(slope=0.5)
    fit = fit_trend(hi, T0 + pd.Timedelta(days=30), CFG)  # index ~10.3: past a level of 9
    assert trend_rul(fit, 9.0, 0.5, CFG) is not None
    fit2 = fit_trend(hi, T0 + pd.Timedelta(days=24), CFG)  # index ~7.3, level 9 => ~3.4 days
    p10, p50, p90 = trend_rul(fit2, 9.0, 0.5, CFG)
    assert p10 <= p50 <= p90
    assert p50 == pytest.approx((9.0 - fit2.hi_now) / fit2.slope, rel=0.3)


def test_trend_rul_is_deterministic():
    fit = fit_trend(ramp_hi(slope=0.5), T0 + pd.Timedelta(days=22), CFG)
    assert trend_rul(fit, 9.0, 2.0, CFG) == trend_rul(fit, 9.0, 2.0, CFG)


def test_trend_rul_respects_the_horizon():
    fit = fit_trend(ramp_hi(slope=0.06, onset_day=2), T0 + pd.Timedelta(days=35), CFG)
    res = trend_rul(fit, 40.0, 1.0, CFG)
    assert res is not None and max(res) <= CFG.horizon_days


# ----------------------------------------------------------------------------- estimate


def test_estimate_statuses_cover_the_failure_to_estimate_cases():
    hi = ramp_hi(slope=0.5)
    ok = estimate_rul("A", hi, T0 + pd.Timedelta(days=24), None, CFG)
    assert ok.status == RULStatus.ESTIMATED and "uncalibrated" in ok.note
    healthy = estimate_rul("A", hi, T0 + pd.Timedelta(days=5), None, CFG)
    assert healthy.status == RULStatus.NOT_DEGRADING and healthy.p50_days is None
    gap = hi.copy()
    gap.iloc[:] = np.nan
    assert estimate_rul("A", gap, T0 + pd.Timedelta(days=24), None, CFG).status == (
        RULStatus.INSUFFICIENT_TREND
    )


def test_an_estimate_always_carries_an_interval_never_a_bare_point():
    est = estimate_rul("A", ramp_hi(slope=0.5), T0 + pd.Timedelta(days=24), None, CFG)
    assert est.p10_days is not None and est.p90_days is not None
    assert est.p10_days <= est.p50_days <= est.p90_days


def test_estimate_ignores_the_future():
    hi = ramp_hi(slope=0.5)
    t = T0 + pd.Timedelta(days=24)
    a = estimate_rul("A", hi, t, None, CFG)
    spoiled = hi.copy()
    spoiled.loc[spoiled.index > t] = 0.0
    b = estimate_rul("A", spoiled, t, None, CFG)
    assert (a.p10_days, a.p50_days, a.p90_days) == (b.p10_days, b.p50_days, b.p90_days)


# ----------------------------------------------------------------------------- model


def _toy_episodes(seed: int, n=6):
    eps = []
    for k in range(n):
        slope = 0.35 + 0.1 * k
        onset = 6
        fail_day = onset + 9.0 / slope
        hi = ramp_hi(
            days=int(fail_day) + 2, onset_day=onset, slope=slope, noise=0.1, seed=seed * 10 + k
        )
        hi = hi.where(hi.index <= T0 + pd.Timedelta(days=fail_day))
        eps.append(RunToFailure(seed, f"A{k}", hi, T0 + pd.Timedelta(days=fail_day)))
    return eps


@pytest.fixture(scope="module")
def toy_model() -> RULModel:
    return fit_rul_model(_toy_episodes(1) + _toy_episodes(2), _toy_episodes(3), CFG)


def test_fitted_model_records_its_seeds_and_sizes(toy_model):
    assert toy_model.train_seeds == [1, 2] and toy_model.cal_seeds == [3]
    assert toy_model.n_train_events == 12 and toy_model.n_cal_events == 6
    assert toy_model.synthetic is True
    assert toy_model.level_mean == pytest.approx(9.3, abs=1.0)


def test_training_and_calibration_seeds_must_be_disjoint():
    with pytest.raises(ValueError, match="disjoint"):
        fit_rul_model(_toy_episodes(1), _toy_episodes(1), CFG)


def test_model_quantiles_are_ordered(toy_model):
    fit = fit_trend(_toy_episodes(1)[0].hi, T0 + pd.Timedelta(days=12), CFG)
    tr = trend_rul(fit, toy_model.level_mean, toy_model.level_sd, CFG)
    q = toy_model.predict_days(feature_vector(fit, tr[1], toy_model.level_mean), CFG.horizon_days)[
        0
    ]
    assert q[0] <= q[1] <= q[2]


def test_model_round_trips_through_json(toy_model, tmp_path):
    path = tmp_path / "m.json"
    toy_model.save(path)
    again = RULModel.load(path)
    assert again == toy_model
    x = np.array([[2.0, 5.0, 0.4, 1.5, 1.2, 3.0, 3.7]])
    assert x.shape[1] == len(toy_model.feature_names)
    np.testing.assert_allclose(again.predict_log(x), toy_model.predict_log(x))


def test_calibrated_estimate_reports_both_methods(toy_model):
    hi = _toy_episodes(1)[0].hi
    est = estimate_rul("A0", hi, T0 + pd.Timedelta(days=14), toy_model, CFG)
    assert est.status == RULStatus.ESTIMATED
    assert est.method_trend_p50 is not None and est.method_quantile_p50 == est.p50_days


def test_conformal_margin_reaches_the_target_coverage_on_calibration_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = 1.5 + 0.5 * x[:, 0] + rng.normal(0, 0.4, 400)
    w = np.ones(400)
    model = fit_quantile_model(x, y, w, CFG, 9.0, 2.0)
    xc = rng.normal(size=(600, 4))
    yc = 1.5 + 0.5 * xc[:, 0] + rng.normal(0, 0.4, 600)
    m = conformal_margin(model, xc, yc, np.ones(600), 0.8)
    wide = model.model_copy(update={"conformal_margin": m})
    q = wide.predict_log(xc)
    assert ((yc >= q[:, 0]) & (yc <= q[:, 2])).mean() >= 0.78


def test_wider_target_coverage_needs_a_larger_margin():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 4))
    y = rng.normal(size=300)
    model = fit_quantile_model(x, y, np.ones(300), CFG, 9.0, 2.0)
    m80 = conformal_margin(model, x, y, np.ones(300), 0.8)
    m95 = conformal_margin(model, x, y, np.ones(300), 0.95)
    assert m95 >= m80


def test_estimate_from_fit_marks_flat_trends_not_degrading():
    fit = fit_trend(hourly(np.full(24 * 30, 3.0)), T0 + pd.Timedelta(days=25), CFG)
    est = estimate_from_fit("A", T0, fit, None, CFG)
    assert est.status == RULStatus.NOT_DEGRADING and "not rising" in est.note
