"""Cost model and the cost-optimal maintenance date given an RUL distribution.

All money figures are ILLUSTRATIVE (kUSD-like units) and exist so that the *ratios* between a
false alarm, a planned replacement and an unplanned failure are explicit and tunable. They are not
any real operator's costs. Replace them with the site's own figures; the policy re-optimises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import ndtr, ndtri

from asset_health.catalog import PROFILES
from asset_health.models import AssetType


class CostModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str = "kUSD (illustrative)"
    false_alarm: float = Field(1.5, ge=0, description="Inspection call-out that finds nothing.")
    planned_replacement: float = Field(15.0, gt=0, description="Planned stop, parts, labour.")
    unplanned_failure: float = Field(
        120.0, gt=0, description="Breakdown: lost production, secondary damage, emergency labour."
    )
    early_loss_per_day: float = Field(
        0.25, ge=0, description="Component life thrown away per day replaced ahead of failure."
    )
    plan_lead_days: float = Field(
        3.0, ge=0, description="Time to mobilise crew and parts once the decision is taken."
    )

    def scaled(self, asset_type: AssetType) -> CostModel:
        """Per-asset-type costs: bigger machines cost proportionally more to stop or lose."""
        m = PROFILES[asset_type].cost_multiplier
        return self.model_copy(
            update={
                "planned_replacement": self.planned_replacement * m,
                "unplanned_failure": self.unplanned_failure * m,
                "early_loss_per_day": self.early_loss_per_day * m,
            }
        )

    @property
    def failure_penalty(self) -> float:
        """Extra cost of an unplanned failure relative to the same job done in a planned stop."""
        return self.unplanned_failure - self.planned_replacement


@dataclass(frozen=True)
class LogNormalRUL:
    """RUL days as a log-normal fitted to (p10, p50, p90)."""

    mu: float
    sigma: float

    @classmethod
    def from_quantiles(cls, p10: float, p50: float, p90: float) -> LogNormalRUL:
        lo, mid, hi = (math.log(max(v, 0.05)) for v in (p10, p50, p90))
        z90 = float(ndtri(0.9))
        sigma = max((hi - lo) / (2.0 * z90), 0.05)
        return cls(mu=mid, sigma=sigma)

    def cdf(self, days: float | np.ndarray) -> float | np.ndarray:
        """P(RUL <= days); accepts scalars or arrays."""
        d = np.asarray(days, dtype=float)
        with np.errstate(divide="ignore"):
            z = (np.log(np.where(d > 0, d, 1.0)) - self.mu) / self.sigma
        out = np.where(d > 0, ndtr(z), 0.0)
        return float(out) if out.ndim == 0 else out

    def ppf(self, q: float) -> float:
        return float(math.exp(self.mu + self.sigma * float(ndtri(q))))

    def expected_excess(self, d: float | np.ndarray) -> float | np.ndarray:
        """E[(RUL - d)+]: expected life thrown away if the part is replaced at day d."""
        arr = np.asarray(d, dtype=float)
        pos = np.where(arr > 0, arr, 1.0)
        a = (self.mu + self.sigma**2 - np.log(pos)) / self.sigma
        b = (self.mu - np.log(pos)) / self.sigma
        mean = math.exp(self.mu + 0.5 * self.sigma**2)
        out = np.where(arr > 0, mean * ndtr(a) - pos * ndtr(b), mean)
        return float(out) if out.ndim == 0 else out


def expected_cost_of_action_at(
    d: float | np.ndarray, dist: LogNormalRUL, costs: CostModel
) -> float | np.ndarray:
    """Expected cost (beyond the planned replacement itself) of replacing at day ``d``:
    life lost if it was early, plus the failure penalty if the part fails first."""
    return costs.early_loss_per_day * dist.expected_excess(d) + costs.failure_penalty * dist.cdf(d)


def optimal_action_day(dist: LogNormalRUL, costs: CostModel, max_days: float = 120.0) -> float:
    """Cost-minimising replacement day (grid search, 0.25 day resolution)."""
    grid = np.arange(0.0, max_days + 0.25, 0.25)
    cost = expected_cost_of_action_at(grid, dist, costs)
    return float(grid[int(np.argmin(cost))])


def latest_safe_days(
    p10: float, p50: float, p90: float, costs: CostModel, max_days: float = 120.0
) -> float:
    """Latest replacement day, counted from now, that is both cost-optimal and no later than the
    pessimistic (p10) end of the RUL interval. Never negative."""
    dist = LogNormalRUL.from_quantiles(p10, p50, p90)
    d_cost = optimal_action_day(dist, costs, max_days)
    return float(max(0.0, min(d_cost, p10)))
