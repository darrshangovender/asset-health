from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from asset_health.catalog import PROFILES, band_orders
from asset_health.config import SimConfig
from asset_health.models import AssetType, FailureEvent, FailureMode
from asset_health.simulate import generate_fleet, write_dataset
from asset_health.simulate.degradation import effects_for, severity_curve
from asset_health.simulate.physics import (
    ORDER_AXIS,
    SpectralDrivers,
    band_matrix,
    line_kernel,
    synthesize_vibration,
)
from asset_health.simulate.regime import ar1, first_order_lag, stoppage_mask

# ----------------------------------------------------------------------------- degradation


def test_severity_curve_endpoints_and_monotone():
    h = np.linspace(0, 100, 501)
    s = severity_curve(h, 20.0, 80.0, 3.0)
    assert s[h <= 20].max() == 0.0
    assert s[-1] == pytest.approx(1.0)
    assert s[h >= 80] == pytest.approx(1.0)
    assert np.all(np.diff(s) >= -1e-12)


def test_severity_curve_larger_k_is_more_back_loaded():
    h = np.linspace(0, 1, 11)
    low = severity_curve(h, 0.0, 1.0, 1.0)
    high = severity_curve(h, 0.0, 1.0, 4.0)
    assert high[5] < low[5]


def test_severity_curve_rejects_inverted_window():
    with pytest.raises(ValueError):
        severity_curve(np.arange(3.0), 5.0, 5.0, 2.0)


def test_effects_neutral_without_failure():
    e = effects_for(None, np.arange(10.0), 0)
    assert np.all(e.severity == 0) and np.all(e.outer_power == 0)
    assert np.all(e.hf_mult == 1) and np.all(e.power_mult == 1)


def _event(mode, race=None):
    return FailureEvent(
        asset_id="A",
        mode=mode,
        race=race,
        onset=pd.Timestamp("2025-01-11").to_pydatetime(),
        failure=pd.Timestamp("2025-01-31").to_pydatetime(),
        shape_k=2.5,
        amplitude=1.0,
        downtime_hours=48,
    )


def test_bearing_spall_outer_race_drives_outer_power_only_first():
    start = int(np.datetime64("2025-01-01", "ns").astype("int64"))
    hours = np.arange(0, 24 * 31, 1.0)
    e = effects_for(_event(FailureMode.BEARING_SPALL, "outer"), hours, start)
    assert e.outer_power[-1] > e.inner_power[-1] > 0
    assert e.outer_power[: 24 * 10].max() == 0.0
    inner = effects_for(_event(FailureMode.BEARING_SPALL, "inner"), hours, start)
    assert inner.inner_power[-1] > inner.outer_power[-1]


@pytest.mark.parametrize(
    "mode,attr,direction",
    [
        (FailureMode.IMPELLER_WEAR, "head_mult", "down"),
        (FailureMode.IMPELLER_WEAR, "proc_mult", "up"),
        (FailureMode.LINER_WEAR, "power_mult", "down"),
        (FailureMode.LINER_WEAR, "proc_mult", "up"),
        (FailureMode.OVERHEATING, "temp_delta", "up"),
    ],
)
def test_failure_modes_move_the_physically_expected_channels(mode, attr, direction):
    start = int(np.datetime64("2025-01-01", "ns").astype("int64"))
    hours = np.arange(0, 24 * 31, 1.0)
    e = effects_for(_event(mode), hours, start)
    v = getattr(e, attr)
    base = 0.0 if attr == "temp_delta" else 1.0
    assert (v[-1] > base) if direction == "up" else (v[-1] < base)
    assert v[0] == base


# ----------------------------------------------------------------------------- physics


def test_line_kernel_has_unit_area():
    k = line_kernel(5.0)
    assert k.sum() * (ORDER_AXIS[1] - ORDER_AXIS[0]) == pytest.approx(1.0, abs=1e-6)


def _spec(atype=AssetType.SLURRY_PUMP):
    from asset_health.catalog import spec_from_profile

    return spec_from_profile(PROFILES[atype], "X", "x")


def _drivers(n=400, **kw):
    one = np.ones(n)
    return SpectralDrivers(machine_mult=one, proc_mult=one, defect_mult=one, disturbance=one, **kw)


def test_healthy_rms_matches_the_power_budget():
    rng = np.random.default_rng(0)
    spec = _spec()
    res = synthesize_vibration(rng, spec, 3.0, _drivers(2000))
    assert np.median(res.rms) == pytest.approx(3.0 * spec.baseline_scale, rel=0.03)


def test_outer_race_defect_raises_the_bpfo_band_not_bpfi():
    # not the pump: its 3 x BPFO line (12.27) falls inside the 2 x BPFI window (a real harmonic
    # coincidence of that geometry); the pulley's lines are clear of each other
    spec = _spec(AssetType.CONVEYOR_PULLEY)
    n = 600
    healthy = synthesize_vibration(np.random.default_rng(1), spec, 3.0, _drivers(n))
    d = _drivers(n, outer_power=np.full(n, 0.2))
    faulty = synthesize_vibration(np.random.default_rng(1), spec, 3.0, d)
    bpfo = faulty.bands["vib_band_bpfo"].mean() / healthy.bands["vib_band_bpfo"].mean()
    bpfi = faulty.bands["vib_band_bpfi"].mean() / healthy.bands["vib_band_bpfi"].mean()
    assert bpfo > 5.0
    assert bpfi == pytest.approx(1.0, abs=0.15)


def test_inner_race_defect_raises_the_bpfi_band():
    spec = _spec()
    n = 600
    healthy = synthesize_vibration(np.random.default_rng(2), spec, 3.0, _drivers(n))
    faulty = synthesize_vibration(
        np.random.default_rng(2), spec, 3.0, _drivers(n, inner_power=np.full(n, 0.2))
    )
    assert faulty.bands["vib_band_bpfi"].mean() > 5 * healthy.bands["vib_band_bpfi"].mean()


def test_hf_multiplier_raises_only_the_hf_band_family():
    spec = _spec()
    n = 600
    base = synthesize_vibration(np.random.default_rng(3), spec, 3.0, _drivers(n))
    hf = synthesize_vibration(
        np.random.default_rng(3), spec, 3.0, _drivers(n, hf_mult=np.full(n, 8.0))
    )
    assert hf.bands["vib_band_hf"].mean() > 2 * base.bands["vib_band_hf"].mean()
    assert hf.bands["vib_band_low"].mean() == pytest.approx(
        base.bands["vib_band_low"].mean(), rel=0.1
    )


def test_band_matrix_matches_catalog_windows():
    spec = _spec()
    m, names = band_matrix(spec)
    assert set(names) == set(band_orders(spec))
    assert m.shape == (len(ORDER_AXIS), len(names))
    assert set(np.unique(m)) <= {0.0, 1.0}


def test_synthesis_is_seed_deterministic():
    spec = _spec()
    a = synthesize_vibration(np.random.default_rng(9), spec, 3.0, _drivers(50))
    b = synthesize_vibration(np.random.default_rng(9), spec, 3.0, _drivers(50))
    np.testing.assert_array_equal(a.rms, b.rms)


# ----------------------------------------------------------------------------- regime helpers


def test_ar1_marginal_std():
    x = ar1(np.random.default_rng(0), 20000, 0.9, 2.0)
    assert x.std() == pytest.approx(2.0, rel=0.1)


def test_first_order_lag_converges_to_a_step():
    x = np.concatenate([np.zeros(50), np.ones(500)])
    y = first_order_lag(x, 1.0, 5.0)
    assert y[0] == 0.0 and y[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(y) >= -1e-12)


def test_stoppage_mask_respects_rate():
    m = stoppage_mask(np.random.default_rng(0), 24 * 400, 1.0, 0.0, (1.0, 5.0))
    assert not m.any()


# ----------------------------------------------------------------------------- fleet


def test_fleet_is_deterministic(tiny_cfg):
    a = generate_fleet(tiny_cfg, keep_latent=False)
    b = generate_fleet(tiny_cfg, keep_latent=False)
    for k in a.raw:
        pd.testing.assert_frame_equal(a.raw[k], b.raw[k])
    assert a.truth.model_dump_json() == b.truth.model_dump_json()


def test_different_seeds_give_different_data(tiny_cfg):
    other = tiny_cfg.model_copy(update={"seed": tiny_cfg.seed + 1})
    a = generate_fleet(tiny_cfg, keep_latent=False)
    b = generate_fleet(other, keep_latent=False)
    k = next(iter(a.raw))
    assert not np.allclose(
        a.raw[k]["vib_rms"].dropna().iloc[:50], b.raw[k]["vib_rms"].dropna().iloc[:50]
    )


def test_manifest_and_truth_are_labelled_synthetic(tiny_fleet):
    assert tiny_fleet.manifest.synthetic is True
    assert "SYNTHETIC" in tiny_fleet.manifest.notice
    assert tiny_fleet.truth.synthetic is True


def test_every_asset_reports_its_channels(tiny_fleet):
    for spec in tiny_fleet.specs:
        df = tiny_fleet.raw[spec.asset_id]
        assert {"timestamp", "load_pct", "speed_rpm", "ambient_c", *spec.channels} <= set(
            df.columns
        )


def test_failures_respect_the_baseline_window(tiny_fleet, tiny_cfg):
    for f in tiny_fleet.truth.failures:
        assert f.onset >= tiny_cfg.start + timedelta(days=tiny_cfg.baseline_days)
        assert f.failure <= tiny_cfg.start + timedelta(days=tiny_cfg.days - 20)
        assert f.mode in dict(
            PROFILES[
                next(s.asset_type for s in tiny_fleet.specs if s.asset_id == f.asset_id)
            ].failure_modes
        )


def test_every_failure_is_followed_by_a_repair_record(tiny_fleet):
    repairs = {m.asset_id for m in tiny_fleet.maintenance if m.kind == "repair"}
    for f in tiny_fleet.truth.failures:
        end = f.failure + timedelta(hours=f.downtime_hours)
        if end < tiny_fleet.manifest.start + timedelta(days=tiny_fleet.manifest.days):
            assert f.asset_id in repairs


def test_failure_fraction_zero_means_no_failures():
    cfg = SimConfig(seed=5, days=70, fleet={AssetType.SLURRY_PUMP: 1}, failure_fraction=0.0)
    fleet = generate_fleet(cfg, keep_latent=False)
    assert fleet.truth.failures == []


def test_vibration_rises_before_a_bearing_failure(tiny_fleet):
    for f in tiny_fleet.truth.failures:
        if f.mode != FailureMode.BEARING_SPALL:
            continue
        lat = tiny_fleet.latent[f.asset_id]
        early = lat.loc[f.onset - timedelta(days=5) : f.onset, "vib_rms"].median()
        late = lat.loc[f.failure - timedelta(days=1) : f.failure, "vib_rms"].median()
        assert late > early


def test_machine_is_stopped_during_the_repair_window(tiny_fleet):
    for f in tiny_fleet.truth.failures:
        lat = tiny_fleet.latent[f.asset_id]
        w = lat.loc[
            f.failure + timedelta(hours=2) : f.failure + timedelta(hours=f.downtime_hours - 2)
        ]
        if len(w):
            assert not w["running"].any()


def test_data_faults_are_logged_with_ground_truth(tiny_fleet):
    kinds = {e.kind for e in tiny_fleet.truth.data_faults}
    assert {"stuck", "dropout"} <= kinds
    for e in tiny_fleet.truth.data_faults:
        assert e.end >= e.start


def test_write_dataset_round_trip_csv(tiny_fleet, tmp_path):
    out = write_dataset(tiny_fleet, tmp_path / "ds")
    assert (out / "SYNTHETIC_DATA.txt").read_text(encoding="utf-8").startswith("SYNTHETIC DATA")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert {p.stem for p in (out / "assets").glob("*.csv")} == set(tiny_fleet.raw)


def test_write_dataset_rejects_unknown_format(tiny_fleet, tmp_path):
    with pytest.raises(ValueError):
        write_dataset(tiny_fleet, tmp_path / "x", "xlsx")


def test_sim_config_validation():
    with pytest.raises(ValueError):
        SimConfig(days=60, baseline_days=30)  # not enough room after the baseline
    with pytest.raises(ValueError):
        SimConfig(fleet={AssetType.SLURRY_PUMP: 0})
