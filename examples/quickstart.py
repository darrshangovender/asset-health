"""Simulate a small SYNTHETIC fleet in memory, analyse it, and print the work orders.

    python examples/quickstart.py

Takes about a minute on a laptop. Everything is offline and seeded.
"""

from __future__ import annotations

import sys

import pandas as pd

from asset_health.analysis import analyse_source
from asset_health.config import SimConfig
from asset_health.ingest import FrameSource
from asset_health.models import SYNTHETIC_NOTICE, AssetType
from asset_health.simulate import generate_fleet


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(SYNTHETIC_NOTICE, "\n")
    cfg = SimConfig(
        seed=11,
        days=170,
        fleet={
            AssetType.SLURRY_PUMP: 2,
            AssetType.CONVEYOR_PULLEY: 1,
            AssetType.HAUL_TRUCK_ENGINE: 1,
        },
        failure_fraction=0.75,
    )
    fleet = generate_fleet(cfg)
    source = FrameSource(fleet.specs, fleet.raw, fleet.maintenance)
    # replay the moment six days before the last injected failure: nothing later is used
    as_of = pd.Timestamp(max(f.failure for f in fleet.truth.failures)) - pd.Timedelta(days=6)
    analysis = analyse_source(source, truth=fleet.truth, n_jobs=1, as_of=as_of)

    print("Injected failures (the answer key, never shown to the detectors):")
    for f in fleet.truth.failures:
        print(
            f"  {f.asset_id}: {f.mode.value}, degradation from {f.onset:%d %b}, "
            f"failure {f.failure:%d %b}"
        )

    print("\nAlerts raised:")
    for aid, a in analysis.assets.items():
        for tr in a.traces:
            e = tr.episode
            fail = next((f for f in fleet.truth.failures if f.asset_id == aid), None)
            lead = (
                f"{(fail.failure - e.opened_at).total_seconds() / 86400:.1f} days "
                "before the failure"
                if fail is not None and fail.onset <= e.opened_at < fail.failure
                else "no failure follows (false alarm)"
            )
            print(f"  {aid}: opened {e.opened_at:%d %b %H:%M}, {e.hypothesis.value}, {lead}")

    print(f"\nOpen work orders as of {as_of:%d %b %Y}:")
    wos = analysis.all_work_orders()
    if not wos:
        print("  none")
    for wo in wos:
        print(f"\n  {wo.work_order_id} [{wo.severity.value}]")
        print("  " + wo.narrative)


if __name__ == "__main__":
    main()
