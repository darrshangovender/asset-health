"""Explainable rule layer: named fault hypotheses over regime residuals.

Each rule says which residual features must be abnormal ("any_of": the strongest one gates the
rule) and which supporting features raise confidence. A rule's strength is 0 when the gate is
closed; otherwise ``0.8 * gate + 0.2 * mean(support)`` where ``gate`` is the strongest gating
residual as a multiple of its threshold and ``support`` the corroborating residuals, each capped
at 1. A gate just open with no corroboration therefore scores 0.8; 1.0 needs corroboration (or a
gate 25% beyond threshold). The output is a hypothesis plus the evidence features, so a work
order can cite exactly what fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from asset_health.models import AssetType, FaultHypothesis


@dataclass(frozen=True)
class Cond:
    feature: str
    direction: Literal["up", "down"]
    threshold: float  # in robust sigma units of the 24h-median residual


@dataclass(frozen=True)
class Rule:
    name: str
    hypothesis: FaultHypothesis
    applies_to: frozenset[AssetType]
    any_of: tuple[Cond, ...]
    support: tuple[Cond, ...] = ()
    explanation: str = ""

    def conditions(self) -> tuple[Cond, ...]:
        return (*self.any_of, *self.support)


_ALL = frozenset(AssetType)
_BEARING_HOSTS = frozenset(
    {
        AssetType.CONVEYOR_IDLER,
        AssetType.CONVEYOR_PULLEY,
        AssetType.PRIMARY_CRUSHER,
        AssetType.SLURRY_PUMP,
        AssetType.HAUL_TRUCK_ENGINE,
    }
)

DEFAULT_RULES: tuple[Rule, ...] = (
    Rule(
        "outer_race_defect",
        FaultHypothesis.BEARING_OUTER_RACE,
        _BEARING_HOSTS,
        any_of=(Cond("vib_band_bpfo", "up", 3.5), Cond("bpfo_share", "up", 3.5)),
        support=(
            Cond("vib_band_hf", "up", 2.5),
            Cond("hf_share", "up", 2.5),
            Cond("temp_c", "up", 2.5),
        ),
        explanation="Energy at the outer-race ball-pass frequency (BPFO) is above its regime-"
        "expected level; broadband high-frequency energy and temperature corroborate.",
    ),
    Rule(
        "inner_race_defect",
        FaultHypothesis.BEARING_INNER_RACE,
        _BEARING_HOSTS,
        any_of=(Cond("vib_band_bpfi", "up", 3.5), Cond("bpfi_share", "up", 3.5)),
        support=(
            Cond("vib_band_hf", "up", 2.5),
            Cond("hf_share", "up", 2.5),
            Cond("temp_c", "up", 2.5),
        ),
        explanation="Energy at the inner-race ball-pass frequency (BPFI) is above its regime-"
        "expected level; broadband high-frequency energy and temperature corroborate.",
    ),
    Rule(
        "bearing_broadband",
        FaultHypothesis.BEARING_GENERIC,
        _BEARING_HOSTS,
        any_of=(Cond("vib_band_hf", "up", 4.5), Cond("hf_share", "up", 4.5)),
        support=(Cond("temp_c", "up", 2.5), Cond("vib_band_low", "up", 2.5)),
        explanation="Broadband high-frequency vibration is elevated without a clean race-"
        "frequency signature: early bearing wear or lubrication loss.",
    ),
    Rule(
        "impeller_wear",
        FaultHypothesis.IMPELLER_WEAR,
        frozenset({AssetType.SLURRY_PUMP}),
        any_of=(
            Cond("pressure_kpa", "down", 3.5),
            Cond("flow_m3h", "down", 3.5),
            Cond("vib_band_proc", "up", 4.5),
            Cond("proc_share", "up", 4.5),
        ),
        support=(
            Cond("vib_band_proc", "up", 3.0),
            Cond("proc_share", "up", 3.0),
            Cond("pressure_kpa", "down", 2.5),
            Cond("flow_m3h", "down", 2.5),
            Cond("power_kw", "up", 2.5),
        ),
        explanation="Discharge pressure/flow have fallen at the same speed and vane-pass "
        "vibration has risen: the hydraulic signature of a worn impeller.",
    ),
    Rule(
        "liner_wear",
        FaultHypothesis.LINER_WEAR,
        frozenset({AssetType.PRIMARY_CRUSHER}),
        any_of=(
            Cond("power_kw", "down", 3.5),
            Cond("current_a", "down", 3.5),
            Cond("vib_band_proc", "up", 4.5),
            Cond("proc_share", "up", 4.5),
        ),
        support=(
            Cond("vib_band_proc", "up", 3.0),
            Cond("proc_share", "up", 3.0),
            Cond("power_kw", "down", 2.5),
            Cond("vib_band_low", "up", 2.5),
        ),
        explanation="Power at a given feed has fallen while impact-order vibration has risen: "
        "the cavity profile is changing as liners wear.",
    ),
    Rule(
        "overheating",
        FaultHypothesis.OVERHEATING,
        _ALL,
        any_of=(Cond("temp_c", "up", 4.0),),
        support=(Cond("vib_band_hf", "up", 2.0), Cond("hf_share", "up", 2.0)),
        explanation="Temperature is well above the regime-expected value (load, speed and "
        "ambient accounted for): cooling or lubrication is degrading.",
    ),
)


def _cond_strength(z24: pd.DataFrame, c: Cond) -> pd.Series:
    if c.feature not in z24.columns:
        return pd.Series(0.0, index=z24.index)
    v = z24[c.feature] if c.direction == "up" else -z24[c.feature]
    return (v / c.threshold).clip(lower=0.0, upper=3.0).fillna(0.0)


def rule_strengths(
    z24: pd.DataFrame, asset_type: AssetType, rules: tuple[Rule, ...] = DEFAULT_RULES
) -> pd.DataFrame:
    """Per-rule strength series (columns are rule names)."""
    out = {}
    for rule in rules:
        if asset_type not in rule.applies_to:
            continue
        gate = pd.concat([_cond_strength(z24, c) for c in rule.any_of], axis=1).max(axis=1)
        if rule.support:
            sup = pd.concat([_cond_strength(z24, c) for c in rule.support], axis=1)
            sup = sup.clip(upper=1.0).mean(axis=1)
            strength = 0.8 * gate + 0.2 * sup
        else:
            strength = gate
        out[rule.name] = strength.where(gate >= 1.0, 0.0)
    return pd.DataFrame(out, index=z24.index)


def best_hypothesis(
    strengths: pd.DataFrame, rules: tuple[Rule, ...] = DEFAULT_RULES
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (score, hypothesis, rule_name) per row; UNEXPLAINED where nothing fires."""
    by_name = {r.name: r for r in rules}
    if strengths.shape[1] == 0:
        idx = strengths.index
        return (
            pd.Series(0.0, index=idx),
            pd.Series(FaultHypothesis.UNEXPLAINED.value, index=idx),
            pd.Series("", index=idx),
        )
    arr = strengths.to_numpy()
    best = arr.argmax(axis=1)
    score = arr.max(axis=1)
    names = np.array(strengths.columns)[best]
    names = np.where(score > 0, names, "")
    hyp = [by_name[n].hypothesis.value if n else FaultHypothesis.UNEXPLAINED.value for n in names]
    return (
        pd.Series(score, index=strengths.index),
        pd.Series(hyp, index=strengths.index),
        pd.Series(names, index=strengths.index),
    )
