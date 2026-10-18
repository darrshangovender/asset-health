"""Event-level scoring: which alerts were detections, which were false alarms.

Rules (also in the README):
* A failure has a *window* ``[onset, failure]`` taken from the simulator's ground truth.
* An alert episode is a **detection** if it OPENED inside the window of a failure on the same
  asset *and* the injected degradation had reached a latent severity of at least
  ``MIN_SEVERITY_AT_OPEN`` (0.05 on the 0-1 scale) by then. Lead time is ``failure - opened_at``
  (the debounced time an operator would have been told, not the time the score first rose). Only
  the first such episode in a window counts.
* An episode that opened inside the window while there was essentially nothing to find (latent
  severity below the floor) is a **coincidental** alarm: an inspection then would find nothing,
  so it is scored as a false alarm and counted separately. Without this rule a noisy baseline
  gets credit for alarms that merely happen to fall in a long (liner, impeller) window.
* Every other episode is a **false alarm**: opened before onset (premature), after the failure, or
  on an asset with no failure. Later episodes inside an already-detected window are re-alerts,
  reported separately.
* False-alarm rate = false alarms per healthy monitored asset-month (hours with a valid decision,
  outside failure windows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asset_health.models import AlertEpisode, FailureEvent, FailureMode, GroundTruth
from asset_health.simulate.degradation import severity_curve

MIN_SEVERITY_AT_OPEN = 0.05


def latent_severity(f: FailureEvent, t: pd.Timestamp) -> float:
    """Ground-truth degradation severity (0 at onset, 1 at failure) at time ``t``."""
    on, fa = pd.Timestamp(f.onset), pd.Timestamp(f.failure)
    h = (pd.Timestamp(t) - on).total_seconds() / 3600.0
    span = (fa - on).total_seconds() / 3600.0
    return float(severity_curve(np.array([h]), 0.0, span, f.shape_k)[0])


@dataclass
class FailureOutcome:
    seed: int
    failure: FailureEvent
    episode: AlertEpisode | None  # the detecting episode, if any
    lead_days: float | None
    n_realerts: int = 0

    @property
    def detected(self) -> bool:
        return self.episode is not None

    def actionable(self, min_lead_days: float) -> bool:
        return self.lead_days is not None and self.lead_days >= min_lead_days

    @property
    def mode(self) -> FailureMode:
        return self.failure.mode


@dataclass
class Scored:
    outcomes: list[FailureOutcome] = field(default_factory=list)
    false_alarms: list[tuple[str, AlertEpisode]] = field(default_factory=list)  # (asset, episode)
    coincidental: list[tuple[str, AlertEpisode]] = field(default_factory=list)  # subset of above


def score_asset(
    seed: int,
    asset_id: str,
    episodes: list[AlertEpisode],
    truth: GroundTruth,
    min_severity: float = MIN_SEVERITY_AT_OPEN,
) -> Scored:
    out = Scored()
    fails = [f for f in truth.failures if f.asset_id == asset_id]
    used: set[int] = set()
    coincidental: set[int] = set()
    for f in fails:
        on, fa = pd.Timestamp(f.onset), pd.Timestamp(f.failure)
        inside = [i for i, e in enumerate(episodes) if on <= pd.Timestamp(e.opened_at) < fa]
        early = [
            i
            for i in inside
            if latent_severity(f, pd.Timestamp(episodes[i].opened_at)) < min_severity
        ]
        # an alarm opened before there was anything to find, and the alarms before it, are not
        # detections; the first episode opened once the fault is physically there is
        if early:
            last_early = max(early)
            coincidental.update(i for i in inside if i <= last_early)
            inside = [i for i in inside if i > last_early]
        if inside:
            first = episodes[inside[0]]
            lead = (fa - pd.Timestamp(first.opened_at)).total_seconds() / 86400.0
            out.outcomes.append(FailureOutcome(seed, f, first, lead, len(inside) - 1))
            used.update(inside)
        else:
            out.outcomes.append(FailureOutcome(seed, f, None, None))
    for i, e in enumerate(episodes):
        if i not in used:
            out.false_alarms.append((asset_id, e))
            if i in coincidental:
                out.coincidental.append((asset_id, e))
    return out


def near_data_fault(
    truth: GroundTruth, asset_id: str, t: pd.Timestamp, slack_h: float = 48.0
) -> bool:
    """Was a false alarm opened while (or just after) an injected data fault was active?"""
    slack = pd.Timedelta(hours=slack_h)
    for e in truth.data_faults:
        if e.asset_id != asset_id or e.kind in {"missing_row", "duplicate_row", "clock_drift"}:
            continue
        if pd.Timestamp(e.start) - slack <= t <= pd.Timestamp(e.end) + slack:
            return True
    return False
