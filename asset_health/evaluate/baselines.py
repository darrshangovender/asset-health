"""Naive baselines the system is measured against.

* **Fixed statistical limits** (two variants). Per asset life segment, a static limit on overall
  vibration and on temperature, set from the commissioning window as median + ``k`` robust sigma,
  with ``k`` tuned per channel on the development seeds (the same tuning benefit the system gets).
  No regime normalisation, no data-quality gating, no RUL; an alarm lasts while a signal stays
  above its limit, debounced by ``confirm_hours``.

  - ``raw``: limits on raw ``vib_rms`` and raw ``temp_c``. A three-week winter commissioning
    window cannot anticipate summer, so this variant alarms every hot season.
  - ``rise``: limits on ``vib_rms`` and on temperature *rise above the logged ambient*, the
    obvious repair of the seasonal problem.

  Neither is a DCS alarm list with engineering limits (ISO 20816 zones, OEM trip temperatures);
  those were not modelled, and the README says so.
* **Age-based (calendar) replacement**: replace the monitored component whenever its age reaches
  ``fraction x eta`` (``eta`` = the class's characteristic Weibull life), counting from its
  install date. A component that is already older than that when the record starts is replaced
  on day 0. A replacement before the failure prevents it: the new part has a fresh life, and like
  every part fitted during the record (including the system's replacements) it is assumed to
  outlive the record.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from asset_health.catalog import PROFILES
from asset_health.evaluate.artifact import AssetArtifact
from asset_health.models import AlertEpisode, FailureEvent, GroundTruth, Severity

NAIVE_VARIANTS: dict[str, tuple[str, str]] = {
    "raw": ("vib_rms", "temp_c"),
    "rise": ("vib_rms", "temp_rise"),
}
NAIVE_CHANNELS = NAIVE_VARIANTS["raw"]

Ks = float | dict[str, float]


def _k_for(k: Ks, ch: str) -> float:
    return float(k[ch]) if isinstance(k, dict) else float(k)


def _over_and_armed(
    art: AssetArtifact, k: Ks, channels: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Hours above a limit, and hours in which the baseline is actually watching (a limit has
    been set and at least one of its channels has a reading)."""
    df = art.naive
    idx = df.index
    n = len(idx)
    seg_edges = [
        idx[0],
        *[b for b in art.repairs if idx[0] < b <= idx[-1]],
        idx[-1] + pd.Timedelta(1, "s"),
    ]
    over = np.zeros(n, dtype=bool)
    armed = np.zeros(n, dtype=bool)
    for s, e in pairwise(seg_edges):
        a = int(idx.searchsorted(s))
        b = int(idx.searchsorted(e))
        c0 = a + art.baseline_hours
        c1 = min(c0 + 21 * 24, b)
        if c1 - c0 < 120:
            continue
        for ch in channels:
            if ch not in df:
                continue
            x = df[ch].to_numpy(dtype=float)
            ref = x[c0:c1]
            ref = ref[~np.isnan(ref)]
            if len(ref) < 120:
                continue
            med = float(np.median(ref))
            sig = max(1.4826 * float(np.median(np.abs(ref - med))), 1e-9)
            seg = x[a:b]
            hit = seg > med + _k_for(k, ch) * sig
            hit[: c1 - a] = False  # no alarms while the limit is still being set
            over[a:b] |= hit & ~np.isnan(seg)
            watching = ~np.isnan(seg)
            watching[: c1 - a] = False
            armed[a:b] |= watching
    return over, armed


def fixed_threshold_episodes(
    art: AssetArtifact,
    k: Ks,
    confirm_hours: int = 3,
    clear_hours: int = 24,
    channels: tuple[str, ...] = NAIVE_CHANNELS,
) -> list[AlertEpisode]:
    """Alarm episodes from static limits on the raw signals of ``art.naive``."""
    idx = art.naive.index
    over, _ = _over_and_armed(art, k, channels)
    return [
        _episode(art, idx, i, j, "recovered" if j is not None else "")
        for i, j in debounce(over, confirm_hours, clear_hours)
    ]


def debounce(
    over: np.ndarray, confirm_hours: int, clear_hours: int
) -> list[tuple[int, int | None]]:
    """(open, close) indices: open after ``confirm_hours`` consecutive hours over the limit,
    close after ``clear_hours`` consecutive hours under it. Walks runs, not hours (same result as
    an hour-by-hour state machine, which ``tests`` check)."""
    over = np.asarray(over, dtype=bool)
    if len(over) == 0:
        return []
    edges = np.flatnonzero(np.diff(over.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    lengths = np.diff(np.concatenate([starts, [len(over)]]))
    out: list[tuple[int, int | None]] = []
    open_i: int | None = None
    for s, n_run in zip(starts.tolist(), lengths.tolist(), strict=True):
        if over[s]:
            if open_i is None and n_run >= confirm_hours:
                open_i = s + confirm_hours - 1
        elif open_i is not None and n_run >= clear_hours:
            out.append((open_i, s + clear_hours - 1))
            open_i = None
    if open_i is not None:
        out.append((open_i, None))
    return out


def fixed_threshold_monitored(
    art: AssetArtifact, channels: tuple[str, ...] = NAIVE_CHANNELS
) -> np.ndarray:
    """Hours the fixed limit is watching: its own false-alarm denominator (not the system's)."""
    _, armed = _over_and_armed(art, 1.0, channels)
    return armed


def _episode(
    art: AssetArtifact, idx: pd.DatetimeIndex, i: int, j: int | None, reason: str
) -> AlertEpisode:
    return AlertEpisode(
        asset_id=art.spec.asset_id,
        opened_at=idx[i].to_pydatetime(),
        closed_at=idx[j].to_pydatetime() if j is not None else None,
        peak_severity=Severity.WARNING,
        closed_reason=reason,
    )


def age_limit_days(art: AssetArtifact, life_fraction: float) -> float:
    """Replacement age for this asset's class: ``life_fraction`` x characteristic life."""
    return life_fraction * PROFILES[art.spec.asset_type].life_weibull[1]


def age_based_replacements(
    age_at_start_days: float,
    interval_days: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    failure: FailureEvent | None = None,
    repair_at: pd.Timestamp | None = None,
) -> tuple[list[pd.Timestamp], pd.Timestamp | None]:
    """Replacement times under age-based replacement, and the replacement that prevented the
    failure (``None`` if it was not prevented or there is none).

    The first replacement is due when the component reaches ``interval_days`` of age (day 0 if it
    is already older), then every ``interval_days``. If the failure comes first it happens; the
    repaired part's clock starts at ``repair_at``.
    """
    step = pd.Timedelta(days=interval_days)
    r = start + pd.Timedelta(days=max(0.0, interval_days - age_at_start_days))
    prevented = None
    if failure is not None:
        f = pd.Timestamp(failure.failure)
        if r < f and r < end:
            prevented = r
        else:
            rep = (
                repair_at
                if repair_at is not None
                else f + pd.Timedelta(hours=failure.downtime_hours)
            )
            r = rep + step
    out: list[pd.Timestamp] = []
    while r < end:
        out.append(r)
        r += step
    return out, prevented


def truth_failures(truth: GroundTruth, asset_id: str):
    return [f for f in truth.failures if f.asset_id == asset_id]
