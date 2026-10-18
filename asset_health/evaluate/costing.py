"""Cost accounting for maintenance policies on one or more seeds.

Every policy is charged with the same bookkeeping, using the (illustrative) ``CostModel``:

* each alert is **inspected**: ``false_alarm`` cost, whether it turns out real or not;
* an inspected, genuinely degrading asset is replaced at the policy's execution time: the planned
  replacement cost plus ``early_loss_per_day`` for each day of remaining life thrown away;
* a failure that happens before the replacement (or that was never alerted) costs the unplanned
  failure cost instead.

Inspections are assumed to tell a degrading component from a healthy one (an optimistic simplifier
that is applied identically to every alarm-driven policy; see the README limitations).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from asset_health.alerting.cost import CostModel
from asset_health.evaluate.events import FailureOutcome
from asset_health.models import AssetType


@dataclass
class CostBreakdown:
    inspections: float = 0.0
    planned: float = 0.0
    early_loss: float = 0.0
    unplanned: float = 0.0
    n_inspections: int = 0
    n_planned: int = 0
    n_unplanned: int = 0

    @property
    def total(self) -> float:
        return self.inspections + self.planned + self.early_loss + self.unplanned

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(*(a + b for a, b in zip(self._vals(), other._vals(), strict=True)))

    def _vals(self) -> tuple:
        return tuple(asdict(self).values())

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["total"] = self.total
        return d


def charge_failure(
    out: CostBreakdown,
    costs: CostModel,
    asset_type: AssetType,
    failure: pd.Timestamp,
    execution: pd.Timestamp | None,
) -> None:
    """Add the cost of one failure given when (if ever) the part was replaced."""
    c = costs.scaled(asset_type)
    if execution is not None and execution < failure:
        out.planned += c.planned_replacement
        out.n_planned += 1
        out.early_loss += c.early_loss_per_day * (failure - execution).total_seconds() / 86400.0
    else:
        out.unplanned += c.unplanned_failure
        out.n_unplanned += 1


def charge_alarm_policy(
    out: CostBreakdown,
    costs: CostModel,
    asset_type: AssetType,
    outcomes: list[FailureOutcome],
    n_false_alarms: int,
    execution_of,
) -> None:
    """Charge one asset. ``execution_of(outcome)`` gives the replacement time for a detection."""
    c = costs.scaled(asset_type)
    out.inspections += c.false_alarm * n_false_alarms
    out.n_inspections += n_false_alarms
    for o in outcomes:
        fail = pd.Timestamp(o.failure.failure)
        if o.detected:
            out.inspections += c.false_alarm
            out.n_inspections += 1
            charge_failure(out, costs, asset_type, fail, execution_of(o))
        else:
            charge_failure(out, costs, asset_type, fail, None)
