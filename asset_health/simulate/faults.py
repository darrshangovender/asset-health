"""Data-fault injection (SYNTHETIC): what a real historian export looks like on a bad day.

Faults injected, each with ground truth so the ingest checks can be scored:
  dropout       a channel reads NaN for hours to days
  stuck         a channel repeats its last good value (frozen transmitter / PLC tag)
  sentinel      out-of-range codes (-999, 9999, 65535)
  glitch        isolated in-range spikes (x3 to x6)
  comm_outage   whole rows missing for a block (gateway down)
  missing_row   isolated whole-row loss
  duplicate_row historian replay: an identical row repeated
  clock_drift   the logger clock runs fast/slow, plus occasional NTP steps
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from asset_health.catalog import REGIME_CHANNELS
from asset_health.config import DataFaultConfig
from asset_health.models import DataFaultEvent

SENTINELS = (-999.0, 9999.0, 65535.0)


def _ts(index: pd.DatetimeIndex, i: int) -> datetime:
    return index[min(max(i, 0), len(index) - 1)].to_pydatetime()


def apply_data_faults(
    rng: np.random.Generator,
    frame: pd.DataFrame,
    asset_id: str,
    cfg: DataFaultConfig,
    dt_s: float,
) -> tuple[pd.DataFrame, list[DataFaultEvent]]:
    """Return a dirty copy with a ``timestamp`` column (reported time) and the fault log.

    ``frame`` has a regular true-time DatetimeIndex and one column per channel.
    """
    index = frame.index.as_unit("ns")
    n = len(frame)
    days = n * dt_s / 86400.0
    values = frame.to_numpy(dtype=float, copy=True)
    cols = list(frame.columns)
    events: list[DataFaultEvent] = []

    def hours_to_samples(h: float) -> int:
        return max(1, round(h * 3600.0 / dt_s))

    for j, name in enumerate(cols):
        w = 0.5 if name in REGIME_CHANNELS else 1.0
        for _ in range(rng.poisson(cfg.dropout_per_channel_day * w * days)):
            s = int(rng.integers(0, n))
            d = hours_to_samples(rng.uniform(*cfg.dropout_hours))
            values[s : s + d, j] = np.nan
            events.append(
                DataFaultEvent(
                    asset_id=asset_id,
                    kind="dropout",
                    channel=name,
                    start=_ts(index, s),
                    end=_ts(index, s + d - 1),
                )
            )
        for _ in range(rng.poisson(cfg.stuck_per_channel_day * w * days)):
            s = int(rng.integers(1, n))
            d = hours_to_samples(rng.uniform(*cfg.stuck_hours))
            frozen = values[s - 1, j]
            if np.isnan(frozen):
                continue
            values[s : s + d, j] = frozen
            events.append(
                DataFaultEvent(
                    asset_id=asset_id,
                    kind="stuck",
                    channel=name,
                    start=_ts(index, s),
                    end=_ts(index, s + d - 1),
                    detail=f"frozen at {frozen:.4g}",
                )
            )
        sent = np.flatnonzero(rng.random(n) < cfg.sentinel_prob * w)
        for i in sent:
            values[i, j] = rng.choice(SENTINELS)
            events.append(
                DataFaultEvent(
                    asset_id=asset_id,
                    kind="sentinel",
                    channel=name,
                    start=_ts(index, int(i)),
                    end=_ts(index, int(i)),
                )
            )
        glitch = np.flatnonzero(rng.random(n) < cfg.glitch_prob * w)
        for i in glitch:
            if np.isnan(values[i, j]) or values[i, j] <= 0 or values[i, j] in SENTINELS:
                continue
            values[i, j] = values[i, j] * rng.uniform(3.0, 6.0)
            events.append(
                DataFaultEvent(
                    asset_id=asset_id,
                    kind="glitch",
                    channel=name,
                    start=_ts(index, int(i)),
                    end=_ts(index, int(i)),
                )
            )

    keep = np.ones(n, dtype=bool)
    for _ in range(rng.poisson(cfg.comm_outage_per_day * days)):
        s = int(rng.integers(0, n))
        d = hours_to_samples(rng.uniform(*cfg.comm_outage_hours))
        keep[s : s + d] = False
        events.append(
            DataFaultEvent(
                asset_id=asset_id,
                kind="comm_outage",
                start=_ts(index, s),
                end=_ts(index, s + d - 1),
            )
        )
    for i in np.flatnonzero(rng.random(n) < cfg.missing_row_prob):
        if keep[i]:
            keep[i] = False
            events.append(
                DataFaultEvent(
                    asset_id=asset_id,
                    kind="missing_row",
                    start=_ts(index, int(i)),
                    end=_ts(index, int(i)),
                )
            )

    # reported timestamps (true time + logger clock error)
    t0 = index[0].value
    elapsed_s = (index.asi8 - t0) / 1e9
    offset = rng.normal(0.0, 1.5, n)  # sub-sample jitter
    detail = ""
    if rng.random() < cfg.clock_drift_asset_fraction:
        drift = rng.uniform(*cfg.clock_drift_s_per_day) * rng.choice([-1.0, 1.0])
        offset = offset + drift * elapsed_s / 86400.0
        detail = f"drift {drift:+.0f} s/day"
        n_steps = int(rng.integers(0, cfg.clock_steps_max + 1))
        for _ in range(n_steps):
            at = int(rng.integers(n // 10, n))
            mag = rng.uniform(*cfg.clock_step_s)
            if rng.random() < 0.2:
                mag = rng.uniform(4000.0, 6000.0)  # large NTP correction => timestamps go backwards
            sign = -1.0 if rng.random() < 0.5 or mag > 3600 else 1.0
            offset[at:] += sign * mag
        detail += f", {n_steps} step(s)"
        events.append(
            DataFaultEvent(
                asset_id=asset_id,
                kind="clock_drift",
                start=_ts(index, 0),
                end=_ts(index, n - 1),
                detail=detail,
            )
        )
    reported = (index.asi8 + np.round(offset).astype("int64") * 10**9).astype("datetime64[ns]")

    out = pd.DataFrame(values, columns=cols)
    out.insert(0, "timestamp", reported)
    out = out.loc[keep].reset_index(drop=True)

    dup_rows = np.flatnonzero(rng.random(len(out)) < 0.0005)
    if len(dup_rows):
        out = pd.concat([out, out.iloc[dup_rows]]).sort_index(kind="stable").reset_index(drop=True)
        for i in dup_rows:
            ts = out["timestamp"].iloc[min(int(i), len(out) - 1)].floor("s").to_pydatetime()
            events.append(DataFaultEvent(asset_id=asset_id, kind="duplicate_row", start=ts, end=ts))
    return out, events
