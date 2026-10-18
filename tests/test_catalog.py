from __future__ import annotations

from itertools import pairwise

import pytest

from asset_health.catalog import (
    ALL_CHANNELS,
    CHANNEL_DESCRIPTIONS,
    CHANNEL_DIRECTION,
    CHANNEL_UNITS,
    PROFILES,
    REGIME_CHANNELS,
    SENSOR_CHANNELS,
    SHARE_FEATURES,
    band_orders,
    channel_limits,
    rated_current,
    spec_from_profile,
)
from asset_health.models import AssetType


@pytest.mark.parametrize("atype", list(AssetType))
def test_every_asset_type_has_a_profile(atype):
    assert atype in PROFILES
    assert PROFILES[atype].asset_type == atype


@pytest.mark.parametrize("atype", list(AssetType))
def test_monitored_channels_are_reported_channels(atype):
    p = PROFILES[atype]
    assert set(p.monitored) <= set(p.channels)
    assert set(p.channels) <= set(SENSOR_CHANNELS)


@pytest.mark.parametrize("atype", list(AssetType))
def test_degradation_windows_exist_for_every_failure_mode(atype):
    p = PROFILES[atype]
    modes = {m for m, _ in p.failure_modes}
    assert modes == set(p.degradation_days)
    for lo, hi in p.degradation_days.values():
        assert 0 < lo < hi


@pytest.mark.parametrize("atype", list(AssetType))
def test_defect_windows_do_not_overlap(atype):
    """Band windows must be disjoint or a defect would leak into the wrong channel."""
    windows = band_orders(PROFILES[atype])
    spans = [(lo, hi, name) for name, ws in windows.items() for lo, hi in ws]
    spans.sort()
    for (_, hi_a, name_a), (lo_b, _, name_b) in pairwise(spans):
        assert hi_a <= lo_b + 1e-9, f"{name_a} overlaps {name_b}"


@pytest.mark.parametrize("atype", list(AssetType))
def test_proc_band_only_where_a_process_order_exists(atype):
    p = PROFILES[atype]
    has_band = "vib_band_proc" in p.channels
    assert has_band == (p.proc_order is not None)


def test_failure_mode_weights_positive():
    for p in PROFILES.values():
        assert all(w > 0 for _, w in p.failure_modes)


def test_all_channels_have_units_and_directions():
    for c in SENSOR_CHANNELS:
        assert c in CHANNEL_UNITS
        assert c in CHANNEL_DIRECTION
    for c in REGIME_CHANNELS:
        assert c in CHANNEL_UNITS
    assert set(ALL_CHANNELS) == set(REGIME_CHANNELS) | set(SENSOR_CHANNELS)


def test_share_features_have_descriptions():
    for share in SHARE_FEATURES.values():
        assert share in CHANNEL_DESCRIPTIONS


@pytest.mark.parametrize("atype", list(AssetType))
def test_channel_limits_cover_every_channel_and_are_ordered(atype):
    spec = spec_from_profile(PROFILES[atype], "X", "x")
    limits = channel_limits(spec)
    for c in (*REGIME_CHANNELS, *spec.channels):
        assert c in limits
        lo, hi = limits[c]
        assert lo < hi


def test_rated_current_is_physically_plausible():
    # 450 kW at 690 V: roughly 450e3 / (sqrt(3) * 690 * 0.84) = 450 A order of magnitude
    i = rated_current(PROFILES[AssetType.CONVEYOR_PULLEY])
    assert 350 < i < 600


def test_engine_is_decoupled_from_ambient_and_others_are_not():
    assert PROFILES[AssetType.HAUL_TRUCK_ENGINE].ambient_coupling == 0.0
    for t in AssetType:
        if t != AssetType.HAUL_TRUCK_ENGINE:
            assert PROFILES[t].ambient_coupling == 1.0
