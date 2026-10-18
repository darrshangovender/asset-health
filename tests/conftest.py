"""Shared fixtures. The expensive ones are session-scoped and built once."""

from __future__ import annotations

import sys

import pytest

from asset_health.analysis import AnalysisConfig, analyse_runs
from asset_health.config import SimConfig
from asset_health.ingest import FrameSource
from asset_health.models import AssetType
from asset_health.pipeline import run_source
from asset_health.simulate import generate_fleet

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SMALL_FLEET = {
    AssetType.CONVEYOR_PULLEY: 1,
    AssetType.PRIMARY_CRUSHER: 1,
    AssetType.SLURRY_PUMP: 2,
    AssetType.HAUL_TRUCK_ENGINE: 1,
}


def small_sim(seed: int = 7, days: int = 170, **kw) -> SimConfig:
    return SimConfig(seed=seed, days=days, fleet=SMALL_FLEET, failure_fraction=0.6, **kw)


@pytest.fixture(scope="session")
def tiny_cfg() -> SimConfig:
    """Two assets, 90 days: for tests that only need simulator output."""
    return SimConfig(
        seed=3,
        days=90,
        fleet={AssetType.SLURRY_PUMP: 1, AssetType.CONVEYOR_PULLEY: 1},
        failure_fraction=0.5,
    )


@pytest.fixture(scope="session")
def tiny_fleet(tiny_cfg):
    return generate_fleet(tiny_cfg)


@pytest.fixture(scope="session")
def small_fleet():
    return generate_fleet(small_sim())


@pytest.fixture(scope="session")
def small_runs(small_fleet):
    src = FrameSource(small_fleet.specs, small_fleet.raw, small_fleet.maintenance)
    return run_source(src)


@pytest.fixture(scope="session")
def small_analysis(small_fleet, small_runs):
    return analyse_runs(small_runs, AnalysisConfig(), truth=small_fleet.truth)
