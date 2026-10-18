"""Degradation curves and their physical side-effects (SYNTHETIC).

A failure run is described by a severity ``s(t)`` in [0, 1]: 0 before onset, 1 at functional
failure. The curve is exponential-ish, ``(exp(k u) - 1) / (exp(k) - 1)`` with ``u`` the elapsed
fraction of the run, so larger ``k`` means a long quiet period then a fast final slide, as seen
in bearing spalls. Each failure mode then maps severity onto the channels it physically disturbs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asset_health.models import FailureEvent, FailureMode


def severity_curve(
    hours: np.ndarray, onset_h: float, failure_h: float, shape_k: float
) -> np.ndarray:
    if failure_h <= onset_h:
        raise ValueError("failure must be after onset")
    u = np.clip((hours - onset_h) / (failure_h - onset_h), 0.0, 1.0)
    return (np.exp(shape_k * u) - 1.0) / (np.exp(shape_k) - 1.0)


@dataclass
class Effects:
    """Neutral by default; a failure mode overwrites the parts it disturbs."""

    n: int
    severity: np.ndarray = field(default=None)  # type: ignore[assignment]
    outer_power: np.ndarray = field(default=None)  # type: ignore[assignment]
    inner_power: np.ndarray = field(default=None)  # type: ignore[assignment]
    hf_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    floor_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    shaft_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    proc_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    temp_delta: np.ndarray = field(default=None)  # type: ignore[assignment]
    power_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    head_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    flow_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    eff_mult: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        zeros = ("severity", "outer_power", "inner_power", "temp_delta")
        ones = (
            "hf_mult",
            "floor_mult",
            "shaft_mult",
            "proc_mult",
            "power_mult",
            "head_mult",
            "flow_mult",
            "eff_mult",
        )
        for name in zeros:
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(self.n))
        for name in ones:
            if getattr(self, name) is None:
                setattr(self, name, np.ones(self.n))


def effects_for(event: FailureEvent | None, hours: np.ndarray, start_ts_ns: int) -> Effects:
    """Map a ground-truth failure event to channel disturbances."""
    n = len(hours)
    eff = Effects(n)
    if event is None:
        return eff
    onset_h = (np.datetime64(event.onset, "ns").astype("int64") - start_ts_ns) / 3.6e12
    fail_h = (np.datetime64(event.failure, "ns").astype("int64") - start_ts_ns) / 3.6e12
    s = severity_curve(hours, onset_h, fail_h, event.shape_k)
    a = event.amplitude
    eff.severity = s

    if event.mode == FailureMode.BEARING_SPALL:
        main = 0.30 * a * s**2.2
        second = 0.12 * a * np.clip(s - 0.5, 0, None) ** 1.5
        if event.race == "inner":
            eff.inner_power, eff.outer_power = main, second
        else:
            eff.outer_power, eff.inner_power = main, second
        eff.hf_mult = 1.0 + 5.0 * a * s**2
        eff.floor_mult = 1.0 + 0.4 * a * s**2.5
        eff.shaft_mult = 1.0 + 0.8 * s**3
        eff.temp_delta = 14.0 * np.sqrt(a) * s**3
        eff.power_mult = 1.0 + 0.012 * s**3
    elif event.mode == FailureMode.IMPELLER_WEAR:
        eff.proc_mult = 1.0 + 14.0 * a * s**1.5
        eff.shaft_mult = 1.0 + 1.2 * a * s
        eff.hf_mult = 1.0 + 1.0 * a * s**2
        eff.head_mult = 1.0 - np.minimum(0.22 * a * s, 0.4)
        eff.flow_mult = 1.0 - np.minimum(0.10 * a * s, 0.25)
        eff.eff_mult = 1.0 - np.minimum(0.15 * a * s, 0.3)
        eff.temp_delta = 3.0 * a * s**2
    elif event.mode == FailureMode.LINER_WEAR:
        eff.proc_mult = 1.0 + 9.0 * a * s**1.5
        eff.shaft_mult = 1.0 + 1.0 * a * s
        eff.power_mult = 1.0 - np.minimum(0.15 * a * s, 0.3)
        eff.temp_delta = 5.0 * a * s**2
        eff.hf_mult = 1.0 + 0.6 * a * s**2
    elif event.mode == FailureMode.OVERHEATING:
        eff.temp_delta = 24.0 * a * s**1.4
        eff.hf_mult = 1.0 + 0.9 * a * s**2
        eff.floor_mult = 1.0 + 0.2 * a * s
        eff.power_mult = 1.0 - 0.10 * a * np.clip((s - 0.7) / 0.3, 0.0, 1.0)
    return eff
