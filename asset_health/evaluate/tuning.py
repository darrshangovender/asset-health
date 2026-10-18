"""Tune the few global policy parameters on DEVELOPMENT seeds only.

The objective is total cost (see ``costing``), so the thresholds trade a miss against a false alarm
at the stated cost ratios. Nothing here sees an evaluation seed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from asset_health.alerting.cost import CostModel
from asset_health.config import PolicyConfig, RULConfig
from asset_health.evaluate.artifact import SeedArtifact
from asset_health.evaluate.baselines import NAIVE_CHANNELS
from asset_health.evaluate.policy_eval import (
    Memo,
    evaluate_calendar,
    evaluate_fixed_threshold,
    evaluate_policy,
)
from asset_health.rul.model import RULModel

DEFAULT_GRID = {
    "alert_threshold": (0.4, 0.5, 0.6, 0.8, 1.0, 1.25),
    "cost_exponent": (0.0, 0.15, 0.3),
    "min_detectors": (1, 2),
    "confirm_hours": (4, 8),
}


@dataclass
class TuneResult:
    best: PolicyConfig
    table: list[dict]


def tune_policy(
    arts: list[SeedArtifact],
    base: PolicyConfig,
    rcfg: RULConfig,
    costs: CostModel,
    model: RULModel | None,
    grid: dict[str, tuple] | None = None,
) -> TuneResult:
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    memo = Memo()
    table: list[dict] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = base.model_copy(update=dict(zip(keys, combo, strict=True)))
        evs = [evaluate_policy(a, p, rcfg, costs, model, memo) for a in arts]
        total = sum(e.cost.total for e in evs) / len(evs)
        det = sum(o.detected for e in evs for o in e.outcomes)
        n = sum(len(e.outcomes) for e in evs)
        fa = sum(len(e.false_alarms) for e in evs)
        table.append(
            {
                **dict(zip(keys, combo, strict=True)),
                "mean_cost_per_seed": total,
                "detected": det,
                "failures": n,
                "false_alarms": fa,
            }
        )
    best_row = select_best(table)
    best = base.model_copy(update={k: best_row[k] for k in keys})
    return TuneResult(best, table)


TIE_TOLERANCE = 0.01


def select_best(table: list[dict]) -> dict:
    """Cheapest configuration; among those within 1% of the cheapest (a plateau, where the
    difference is noise) prefer the simplest: no cost scaling, a threshold closest to 1 (the unit
    each detector is calibrated in), then the shortest debounce and the stricter agreement rule."""
    floor = min(r["mean_cost_per_seed"] for r in table)
    near = [r for r in table if r["mean_cost_per_seed"] <= floor * (1 + TIE_TOLERANCE)]
    return min(
        near,
        key=lambda r: (
            r.get("cost_exponent", 0.0),
            abs(r.get("alert_threshold", 1.0) - 1.0),
            r.get("confirm_hours", 0),
            -r.get("min_detectors", 1),
            r["mean_cost_per_seed"],
        ),
    )


def _near_best(rows: list[dict]) -> list[dict]:
    """The same 1% plateau rule the system's tuning uses (``select_best``)."""
    floor = min(r["mean_cost_per_seed"] for r in rows)
    return [r for r in rows if r["mean_cost_per_seed"] <= floor * (1 + TIE_TOLERANCE)]


def tune_fixed_threshold(
    arts: list[SeedArtifact],
    costs: CostModel,
    ks: tuple[float, ...] = (3, 4, 5, 6, 8, 10, 14),
    channels: tuple[str, ...] = NAIVE_CHANNELS,
) -> tuple[dict[str, float], list[dict]]:
    """Per-channel ``k`` for the static limits, by cost on the development seeds. Within 1% of the
    cheapest (the same plateau rule as the system) the fewest false alarms wins."""
    rows = []
    for combo in itertools.product(ks, repeat=len(channels)):
        k = dict(zip(channels, (float(x) for x in combo), strict=True))
        evs = [evaluate_fixed_threshold(a, k, costs, channels=channels) for a in arts]
        rows.append(
            {
                "k": k,
                "mean_cost_per_seed": sum(e.cost.total for e in evs) / len(evs),
                "false_alarms": sum(len(e.false_alarms) for e in evs),
                "detected": sum(o.detected for e in evs for o in e.outcomes),
            }
        )
    best = min(_near_best(rows), key=lambda r: (r["false_alarms"], r["mean_cost_per_seed"]))
    return dict(best["k"]), rows


CALENDAR_FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5)


def tune_calendar(
    arts: list[SeedArtifact],
    costs: CostModel,
    fractions: tuple[float, ...] = CALENDAR_FRACTIONS,
) -> tuple[float, list[dict]]:
    """Replacement age as a fraction of characteristic life, by cost on the development seeds.
    Within 1% of the cheapest the longest interval (fewest replacements) wins."""
    rows = []
    for t in fractions:
        evs = [evaluate_calendar(a, t, costs) for a in arts]
        rows.append(
            {
                "life_fraction": t,
                "mean_cost_per_seed": sum(e.cost.total for e in evs) / len(evs),
                "replacements": sum(e.n_replacements for e in evs),
                "prevented": sum(e.n_prevented for e in evs),
                "failures": sum(e.n_failures for e in evs),
            }
        )
    best = max(_near_best(rows), key=lambda r: r["life_fraction"])
    return float(best["life_fraction"]), rows
