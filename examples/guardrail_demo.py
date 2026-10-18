"""The narrative guardrail: an LLM may only cite what is in the structured work order.

    python examples/guardrail_demo.py

No network, no API key: the "LLMs" below are stubs, one honest and one that invents facts.
"""

from __future__ import annotations

from datetime import datetime

from asset_health.alerting import check_narrative, narrate
from asset_health.models import (
    AssetType,
    Evidence,
    FaultHypothesis,
    RULEstimate,
    RULStatus,
    Severity,
    WorkOrder,
)


class HonestStub:
    def generate(self, work_order: WorkOrder) -> str:
        e = work_order.evidence[0]
        return (
            f"{work_order.asset_id} likely has a bearing outer-race defect: "
            f"{e.feature} is {e.observed} against {e.expected} expected."
        )


class InventingStub:
    def generate(self, work_order: WorkOrder) -> str:
        return (
            "The bearing will seize on 2025-06-30; vibration has reached 55 mm/s "
            "and the oil is contaminated."
        )


def example_order() -> WorkOrder:
    return WorkOrder(
        work_order_id="WO-PMP-01-20250501",
        asset_id="PMP-01",
        asset_name="SP-01 Cyclone Feed Pump",
        asset_type=AssetType.SLURRY_PUMP,
        issued_at=datetime(2025, 5, 20, 6, 0),
        severity=Severity.URGENT,
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


def main() -> None:
    wo = example_order()
    for name, llm in (
        ("honest stub", HonestStub()),
        ("inventing stub", InventingStub()),
        ("none", None),
    ):
        out = narrate(wo, llm)
        print(f"--- {name}: narrative_source = {out.narrative_source}")
        print(out.narrative)
        if llm is not None:
            print("guardrail objections:", check_narrative(llm.generate(wo), wo) or "none")
        print()


if __name__ == "__main__":
    main()
