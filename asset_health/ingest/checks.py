"""Data-quality checks with explicit outcomes.

Principle: bad data must never silently become an alarm. Every check returns a status
(pass / warn / fail) and an action (none / masked / retimed / ...). Masked values become NaN and
are tracked in a ``valid`` mask, so downstream stages can tell "reading is normal" from "no
trustworthy reading".

Structure first: rows out of time order, a cadence that differs from the configured one, two
collectors interleaved, or a re-timed grid that would run past the last real timestamp block the
asset (``StructureError``) before any grid is built. Bad structure is never a warning.

The value masks (range, frozen, spike, erratic vibration) are *causal*: a mask at time t uses
data up to t, so re-running on a truncated history reproduces them for the overlapping rows. The
asset-level verdict (blocked or not, channel coverage, advisories) summarises the whole history
it is given; to replay a past moment, pass ``as_of`` (or truncate the input), and nothing later
is used.

Timestamps are normalised to naive UTC (``asset_health.timeutil``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asset_health.catalog import REGIME_CHANNELS, VIB_BAND_CHANNELS, channel_limits
from asset_health.config import IngestConfig
from asset_health.models import (
    AssetSpec,
    ChannelDQ,
    CheckResult,
    DataQualityReport,
    DQAction,
    DQStatus,
    MaintenanceEvent,
    Segment,
)
from asset_health.timeutil import to_utc_naive, to_utc_naive_series

VIBRATION_CHANNELS: tuple[str, ...] = ("vib_rms", *VIB_BAND_CHANNELS)


class SchemaError(ValueError):
    """Raised for input that is not a telemetry frame at all (no timestamp column ...)."""


@dataclass
class CleanAsset:
    """Output of ingest: a regular hourly grid plus validity masks. Not a pydantic model on
    purpose (it carries DataFrames); its ``report`` is."""

    spec: AssetSpec
    frame: pd.DataFrame  # DatetimeIndex on the regular grid; masked values are NaN
    valid: pd.DataFrame  # bool, same shape: True where a trustworthy reading exists
    present: pd.Series  # bool: a row was actually received for this slot
    report: DataQualityReport
    maintenance: list[MaintenanceEvent] = field(default_factory=list)
    cfg: IngestConfig | None = None

    @property
    def blocked(self) -> bool:
        return self.report.asset_blocked


def _segments(mask: np.ndarray, index: pd.DatetimeIndex, min_len: int = 1) -> list[Segment]:
    segs: list[Segment] = []
    if not mask.any():
        return segs
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for lo, hi in zip(edges[::2], edges[1::2], strict=True):
        if hi - lo >= min_len:
            segs.append(
                Segment(
                    start=index[lo].to_pydatetime(),
                    end=index[hi - 1].to_pydatetime(),
                    n_samples=int(hi - lo),
                )
            )
    return segs


class StructureError(ValueError):
    """The export's structure makes any grid a guess (rows out of order, wrong cadence, two
    collectors interleaved, a grid that would run past the data). The asset is blocked."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(message)
        self.check = check


@dataclass
class _Ordered:
    raw: pd.DataFrame
    notes: list[CheckResult]
    n_dup: int
    n_back: int
    n_conflict: int
    cadence_s: float
    resampled: bool


def _same_as_previous(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return (df[cols].eq(df[cols].shift()) | df[cols].isna() & df[cols].shift().isna()).all(axis=1)


def _order_rows(raw: pd.DataFrame, cfg: IngestConfig) -> _Ordered:
    """Decide the row order and cadence before any grid is built.

    * A file in time order with a few backward steps (a logger clock corrected by NTP) keeps its
      order: the sequence-based retime below is what recovers the true slots then.
    * A cleanly newest-first file (the default sort of many historian and SQL exports) is put back
      into time order and flagged.
    * Anything else (shuffled rows, merged exports) is refused: there is no safe grid to build.
    * Rows that share a timestamp but carry different values are a second collector; more than a
      trace of them is refused.
    * The cadence is inferred (median forward step) and must match the configured one, unless
      ``resample`` is set, in which case a finer cadence is averaged onto the configured grid.
    """
    interval = float(cfg.sample_interval_s)
    value_cols = [c for c in raw.columns if c != "timestamp"]
    notes: list[CheckResult] = []

    secs = raw["timestamp"].astype("int64").to_numpy() / 1e9
    dt = np.diff(secs)
    n_fwd, n_back = int((dt > 0).sum()), int((dt < 0).sum())
    steps = max(1, n_fwd + n_back)
    if n_back / steps > 1.0 - cfg.max_backwards_fraction:
        raw = raw.iloc[::-1].reset_index(drop=True)
        notes.append(
            CheckResult(
                name="row_order",
                status=DQStatus.WARN,
                action=DQAction.SORTED,
                message=f"rows arrived newest-first ({n_back} backward steps); reversed into time "
                "order",
                value=float(n_back),
            )
        )
        n_back = n_fwd  # the few forward steps of a reversed file are its clock corrections
    elif n_back / steps > cfg.max_backwards_fraction:
        raise StructureError(
            "row_order",
            f"rows are not in time order: {n_back} of {steps} steps go backwards "
            f"({n_back / steps:.1%} > {cfg.max_backwards_fraction:.0%}); the export is shuffled "
            "or merged. Sort it by timestamp per source before ingesting.",
        )
    else:
        notes.append(
            CheckResult(
                name="row_order",
                status=DQStatus.PASS,
                message=f"rows in time order ({n_back} backward clock step(s))",
                value=float(n_back),
            )
        )

    # historian replays: identical consecutive rows
    same_ts = raw["timestamp"].diff().dt.total_seconds().fillna(1.0) == 0.0
    dup = same_ts & _same_as_previous(raw, value_cols)
    n_dup = int(dup.sum())
    raw = raw.loc[~dup].reset_index(drop=True)

    # a second collector: same timestamp, different values
    same_ts = raw["timestamp"].diff().dt.total_seconds().fillna(1.0) == 0.0
    n_conflict = int(same_ts.sum())
    if n_conflict / max(1, len(raw)) > cfg.max_conflicting_fraction:
        raise StructureError(
            "collectors",
            f"{n_conflict} rows share a timestamp with a different row "
            f"({n_conflict / len(raw):.1%}): two collectors or a merged export. Keep one source "
            "per asset.",
        )
    raw = raw.loc[~same_ts].reset_index(drop=True)

    secs = raw["timestamp"].astype("int64").to_numpy() / 1e9
    fwd = np.diff(secs)
    fwd = fwd[fwd > 0]
    cadence = float(np.median(fwd)) if len(fwd) else interval
    resampled = False
    if abs(cadence - interval) > cfg.cadence_tolerance * interval:
        finer = cadence < interval
        if not (cfg.resample and finer):
            hint = (
                "set IngestConfig.resample=True to average it onto the configured grid"
                if finer
                else "resample the export, or set IngestConfig.sample_interval_s to match"
            )
            raise StructureError(
                "cadence",
                f"inferred sampling cadence is {cadence:.0f} s but the pipeline is configured for "
                f"{interval:.0f} s; refusing to re-time it ({hint}).",
            )
        bucket = (np.floor(secs / interval) * interval * 1e9).astype("int64")
        grouped = raw[value_cols].groupby(bucket, sort=True).mean()
        raw = grouped.reset_index(drop=True)
        raw.insert(0, "timestamp", pd.to_datetime(grouped.index.to_numpy()).as_unit("ns"))
        resampled = True
        notes.append(
            CheckResult(
                name="cadence",
                status=DQStatus.WARN,
                action=DQAction.RESAMPLED,
                message=f"cadence {cadence:.0f} s averaged onto the {interval:.0f} s grid "
                "(opt-in; logger clock drift not corrected)",
                value=cadence,
            )
        )
    else:
        notes.append(
            CheckResult(
                name="cadence",
                status=DQStatus.PASS,
                message=f"inferred cadence {cadence:.0f} s matches the configured {interval:.0f} s",
                value=cadence,
            )
        )
    return _Ordered(raw, notes, n_dup, n_back, n_conflict, cadence, resampled)


def _retime(
    raw: pd.DataFrame, cfg: IngestConfig
) -> tuple[pd.DataFrame, pd.Series, dict[str, float], list[CheckResult]]:
    """Order and dedupe the rows, then rebuild a regular grid from the *sequence* of samples.

    The logger samples on a fixed cadence, so slot advance per interval is
    ``max(1, round(dt / interval))`` even when its clock drifts or is stepped (including steps
    backwards). That recovers the true slot positions under clock drift, which naive rounding of
    the reported timestamp does not. It is only safe once the rows are known to be in time order
    at the configured cadence (``_order_rows``); otherwise ``StructureError`` is raised.
    """
    interval = float(cfg.sample_interval_s)
    o = _order_rows(raw, cfg)
    raw = o.raw

    secs = raw["timestamp"].astype("int64").to_numpy() / 1e9
    dt = np.diff(secs)
    slots = np.maximum(1, np.rint(dt / interval)).astype(int)
    normal = (slots == 1) & (np.abs(dt - interval) < 0.25 * interval)
    drift = float(np.median(dt[normal] - interval) * 86400.0 / interval) if normal.any() else 0.0
    virtual = np.concatenate([[0], np.cumsum(slots)])
    t0 = np.rint(secs[0] / interval) * interval
    grid_secs = t0 + virtual * interval
    # the grid may not run past the data: a re-timed series that ends well after the last real
    # timestamp would put alerts on dates for which nothing was measured
    span_days = (secs.max() - secs.min()) / 86400.0
    allowed = cfg.max_retime_overrun_h * 3600.0 + 1.5 * abs(drift) * span_days + interval
    overrun = grid_secs[-1] - secs.max()
    if overrun > allowed:
        raise StructureError(
            "grid_end",
            f"re-timed grid would end {overrun / 3600.0:.0f} h after the last reported timestamp "
            f"(allowed {allowed / 3600.0:.0f} h): the row sequence does not match the clock.",
        )
    raw.index = pd.to_datetime(grid_secs, unit="s").as_unit("ns")
    full = pd.date_range(raw.index[0], raw.index[-1], freq=f"{cfg.sample_interval_s}s")
    present = pd.Series(False, index=full)
    present.loc[raw.index] = True
    frame = raw.drop(columns="timestamp").reindex(full)
    stats = {
        "n_dup": float(o.n_dup),
        "n_back": float(o.n_back),
        "n_conflict": float(o.n_conflict),
        "cadence_s": o.cadence_s,
        "drift": 0.0 if o.resampled else drift,
        "n_gap": float((slots - 1).sum()),
        "longest_gap_h": float((slots.max() - 1) * interval / 3600.0) if len(slots) else 0.0,
        "max_step_s": float(np.max(np.abs(dt - interval))) if len(dt) else 0.0,
    }
    return frame, present, stats, o.notes


def _run_lengths_causal(values: np.ndarray) -> np.ndarray:
    """Position within the current run of exactly-equal consecutive values (1-based).
    NaN breaks a run."""
    n = len(values)
    out = np.ones(n, dtype=int)
    same = np.zeros(n, dtype=bool)
    same[1:] = (values[1:] == values[:-1]) & ~np.isnan(values[1:])
    for i in range(1, n):
        if same[i]:
            out[i] = out[i - 1] + 1
    return out


def ingest_asset(
    raw: pd.DataFrame,
    spec: AssetSpec,
    cfg: IngestConfig | None = None,
    maintenance: list[MaintenanceEvent] | None = None,
    as_of: pd.Timestamp | None = None,
) -> CleanAsset:
    """Data-quality gate for one asset.

    ``as_of`` replays the gate as it stood at that instant: rows stamped later and maintenance
    records dated later are dropped *before* anything is computed, so the block decision, the
    masks and the advisories use nothing after ``as_of``.
    """
    cfg = cfg or IngestConfig()
    maintenance = [m for m in (maintenance or []) if m.asset_id == spec.asset_id]
    expected = [*REGIME_CHANNELS, *spec.channels]
    checks: list[CheckResult] = []

    if "timestamp" not in raw.columns:
        raise SchemaError("telemetry frame needs a 'timestamp' column")
    missing_cols = [c for c in expected if c not in raw.columns]
    if not missing_cols:
        raw = raw[["timestamp", *expected]].copy()
        ts = to_utc_naive_series(raw["timestamp"])
        raw = raw.loc[ts.notna().to_numpy()].copy()
        raw["timestamp"] = ts[ts.notna()].to_numpy()
        if as_of is not None:
            cut = to_utc_naive(as_of)
            raw = raw.loc[raw["timestamp"] <= cut].reset_index(drop=True)
            maintenance = [m for m in maintenance if pd.Timestamp(m.timestamp) <= cut]
    if missing_cols or len(raw) < 2:
        msg = f"missing columns: {missing_cols}" if missing_cols else "fewer than 2 rows"
        checks.append(
            CheckResult(
                name="schema", status=DQStatus.FAIL, action=DQAction.ASSET_BLOCKED, message=msg
            )
        )
        return _blocked(spec, cfg, checks, maintenance, len(raw))
    checks.append(CheckResult(name="schema", status=DQStatus.PASS, message="all channels present"))

    try:
        frame, present, st, order_notes = _retime(raw, cfg)
    except StructureError as exc:
        checks.append(
            CheckResult(
                name=exc.check,
                status=DQStatus.FAIL,
                action=DQAction.ASSET_BLOCKED,
                message=str(exc),
            )
        )
        return _blocked(spec, cfg, checks, maintenance, len(raw))
    checks.extend(order_notes)
    n = len(frame)
    index = frame.index

    # ---- timestamps
    dup_status = DQStatus.WARN if st["n_dup"] or st["n_back"] else DQStatus.PASS
    checks.append(
        CheckResult(
            name="timestamps",
            status=dup_status,
            action=DQAction.DEDUPLICATED if st["n_dup"] else DQAction.NONE,
            message=(
                f"{int(st['n_dup'])} duplicate rows dropped, {int(st['n_back'])} backwards steps"
            ),
            value=st["n_dup"] + st["n_back"],
        )
    )
    drift = st["drift"]
    if abs(drift) >= cfg.clock_drift_fail_s_per_day:
        cstat = DQStatus.FAIL
    elif abs(drift) >= cfg.clock_drift_warn_s_per_day or st["n_back"]:
        cstat = DQStatus.WARN
    else:
        cstat = DQStatus.PASS
    checks.append(
        CheckResult(
            name="clock_skew",
            status=cstat,
            action=DQAction.RETIMED if cstat != DQStatus.PASS else DQAction.NONE,
            message=(
                f"estimated logger drift {drift:+.1f} s/day; "
                "timestamps rebuilt from sample sequence"
            ),
            value=drift,
        )
    )

    # ---- gaps
    missing_frac = 1.0 - float(present.mean())
    gstat = (
        DQStatus.FAIL
        if missing_frac >= cfg.fail_missing_fraction
        else DQStatus.WARN
        if missing_frac >= cfg.warn_missing_fraction
        else DQStatus.PASS
    )
    checks.append(
        CheckResult(
            name="gaps",
            status=gstat,
            message=(
                f"{missing_frac:.1%} of slots have no row; longest gap {st['longest_gap_h']:.0f} h"
            ),
            value=missing_frac,
        )
    )

    # ---- per-channel value checks
    limits = channel_limits(spec)
    values = frame.to_numpy(dtype=float, copy=True)
    cols = list(frame.columns)
    stopped = _stopped_mask(frame, spec, cfg)
    erratic = (
        _erratic_vibration(frame, stopped, limits, cfg) if cfg.gating else np.zeros(n, dtype=bool)
    )
    masked_total = {"range": 0, "stuck": 0, "spike": 0, "erratic": 0}
    chan: dict[str, ChannelDQ] = {}
    valid = np.zeros_like(values, dtype=bool)
    erratic_segs = _segments(erratic, index, min_len=4)
    for j, name in enumerate(cols):
        v = values[:, j]
        was_nan = np.isnan(v)
        lo, hi = limits.get(name, (-np.inf, np.inf))
        oor = ~was_nan & ((v < lo) | (v > hi)) & cfg.gating
        v = np.where(oor, np.nan, v)
        if name in VIBRATION_CHANNELS:
            n_err = int((erratic & ~np.isnan(v)).sum())
            masked_total["erratic"] += n_err
            v = np.where(erratic, np.nan, v)
        # stuck: run of identical readings while the machine is running
        runs = _run_lengths_causal(np.where(stopped, np.nan, v))
        stuck = (runs >= cfg.stuck_min_run) & cfg.gating
        v = np.where(stuck, np.nan, v)
        # spikes (not on regime channels): isolated jump vs trailing median of running samples
        spike = np.zeros(n, dtype=bool)
        if name not in REGIME_CHANNELS and cfg.gating:
            run_only = pd.Series(np.where(stopped, np.nan, v))
            med = (
                run_only.shift(1)
                .rolling(cfg.spike_history, min_periods=cfg.spike_min_history)
                .median()
            )
            spike = ((run_only > cfg.spike_ratio * med) & (med > 0) & ~stopped).to_numpy()
            v = np.where(spike, np.nan, v)
        values[:, j] = v
        ok = ~np.isnan(v) & present.to_numpy()
        valid[:, j] = ok
        n_oor, n_stuck, n_spk = int(oor.sum()), int(stuck.sum()), int(spike.sum())
        masked_total["range"] += n_oor
        masked_total["stuck"] += n_stuck
        masked_total["spike"] += n_spk
        vf = float(ok.mean())
        dropout = np.isnan(frame.iloc[:, j].to_numpy()) & present.to_numpy()
        stuck_segs = _segments(stuck, index, min_len=1)
        drop_segs = _segments(dropout, index, min_len=int(cfg.advisory_hours))
        usable = vf >= cfg.min_channel_valid
        status = (
            DQStatus.FAIL
            if not usable
            else DQStatus.WARN
            if (vf < cfg.warn_channel_valid or n_oor or n_stuck or n_spk)
            else DQStatus.PASS
        )
        chan[name] = ChannelDQ(
            channel=name,
            n_samples=n,
            n_missing=int((~ok).sum()),
            n_out_of_range=n_oor,
            n_stuck=n_stuck,
            n_spikes=n_spk,
            valid_fraction=vf,
            stuck_segments=stuck_segs,
            dropout_segments=drop_segs,
            status=status,
            usable=usable,
        )

    checks.append(
        CheckResult(
            name="out_of_range",
            status=DQStatus.WARN if masked_total["range"] else DQStatus.PASS,
            action=DQAction.MASKED if masked_total["range"] else DQAction.NONE,
            message=f"{masked_total['range']} readings outside physical limits masked",
            value=float(masked_total["range"]),
        )
    )
    checks.append(
        CheckResult(
            name="stuck_sensors",
            status=DQStatus.WARN if masked_total["stuck"] else DQStatus.PASS,
            action=DQAction.MASKED if masked_total["stuck"] else DQAction.NONE,
            message=(
                f"{masked_total['stuck']} samples in frozen runs (>= {cfg.stuck_min_run}) masked"
            ),
            value=float(masked_total["stuck"]),
        )
    )
    checks.append(
        CheckResult(
            name="vibration_sensor",
            status=DQStatus.WARN if masked_total["erratic"] else DQStatus.PASS,
            action=DQAction.MASKED if masked_total["erratic"] else DQAction.NONE,
            message=(
                f"{int(erratic.sum())} hours of erratic vibration readings (loose mount or cable "
                "fault suspected) masked on every vibration channel"
            ),
            value=float(erratic.sum()),
        )
    )
    checks.append(
        CheckResult(
            name="spikes",
            status=DQStatus.WARN if masked_total["spike"] else DQStatus.PASS,
            action=DQAction.MASKED if masked_total["spike"] else DQAction.NONE,
            message=(
                f"{masked_total['spike']} isolated spikes "
                f"(> {cfg.spike_ratio}x trailing median) masked"
            ),
            value=float(masked_total["spike"]),
        )
    )

    unusable = [c for c, d in chan.items() if not d.usable]
    blocked = _gate_blocked(chan, spec, n, missing_frac, cfg)
    advisories = _advisories(chan, erratic_segs, cfg)
    if unusable:
        checks.append(
            CheckResult(
                name="channel_coverage",
                status=DQStatus.FAIL if blocked else DQStatus.WARN,
                action=DQAction.ASSET_BLOCKED if blocked else DQAction.CHANNEL_DISABLED,
                message=(
                    f"unusable channels (< {cfg.min_channel_valid:.0%} valid): {sorted(unusable)}"
                ),
            )
        )
    else:
        checks.append(
            CheckResult(
                name="channel_coverage",
                status=DQStatus.PASS,
                message="all channels have usable coverage",
            )
        )
    if blocked and not unusable:
        checks.append(
            CheckResult(
                name="asset_gate",
                status=DQStatus.FAIL,
                action=DQAction.ASSET_BLOCKED,
                message="insufficient clean data for any decision",
            )
        )

    overall = max((c.status for c in checks), key=lambda s: s.rank)
    report = DataQualityReport(
        asset_id=spec.asset_id,
        n_rows_in=len(raw),
        n_rows_grid=n,
        start=index[0].to_pydatetime(),
        end=index[-1].to_pydatetime(),
        overall=overall,
        asset_blocked=blocked,
        clock_drift_s_per_day=drift,
        n_duplicates=int(st["n_dup"]),
        n_backwards=int(st["n_back"]),
        n_gap_samples=int(st["n_gap"]),
        longest_gap_hours=st["longest_gap_h"],
        checks=checks,
        channels=chan,
        sensor_advisories=advisories,
        erratic_segments=erratic_segs,
    )
    clean = pd.DataFrame(values, index=index, columns=cols)
    return CleanAsset(
        spec=spec,
        frame=clean,
        valid=pd.DataFrame(valid, index=index, columns=cols),
        present=present,
        report=report,
        maintenance=maintenance,
        cfg=cfg,
    )


def _gate_blocked(
    chan: dict[str, ChannelDQ], spec: AssetSpec, n: int, missing_frac: float, cfg: IngestConfig
) -> bool:
    """The asset gate: no decision without the regime channels, two usable sensors, enough rows
    and enough received slots."""
    unusable = [c for c, d in chan.items() if not d.usable]
    regime_dead = [c for c in ("load_pct", "speed_rpm") if c in unusable]
    sensors_ok = [c for c in spec.channels if c in chan and chan[c].usable]
    return (
        bool(regime_dead)
        or len(sensors_ok) < 2
        or n < cfg.min_rows
        or missing_frac >= cfg.fail_missing_fraction
    )


def _advisories(
    chan: dict[str, ChannelDQ], erratic_segs: list[Segment], cfg: IngestConfig
) -> list[str]:
    out: list[str] = []
    for seg in erratic_segs:
        out.append(
            f"vibration: erratic readings for {seg.hours:.0f} h from "
            f"{seg.start:%Y-%m-%d %H:%M}; check the accelerometer mount and cable"
        )
    for name, d in chan.items():
        for seg in d.stuck_segments:
            if seg.hours >= cfg.advisory_hours:
                out.append(
                    f"{name}: stuck/flat-lined for {seg.hours:.0f} h from "
                    f"{seg.start:%Y-%m-%d %H:%M}; check transmitter and PLC tag"
                )
        for seg in d.dropout_segments:
            out.append(f"{name}: no data for {seg.hours:.0f} h from {seg.start:%Y-%m-%d %H:%M}")
    return out


def _clip_segments(segs: list[Segment], until: pd.Timestamp) -> list[Segment]:
    out = []
    for s in segs:
        if pd.Timestamp(s.start) > until:
            continue
        if pd.Timestamp(s.end) > until:
            hours = int((until - pd.Timestamp(s.start)) / pd.Timedelta(hours=1)) + 1
            s = s.model_copy(update={"end": until.to_pydatetime(), "n_samples": hours})
        out.append(s)
    return out


def report_as_of(clean: CleanAsset, as_of: pd.Timestamp | None) -> DataQualityReport:
    """The data-quality report as it stood at ``as_of``: coverage, the asset gate, segments and
    advisories recomputed from the (causal) masks up to that instant. Equal to ingesting the
    history truncated at ``as_of``, without re-reading the export."""
    r = clean.report
    if as_of is None or r.end is None or pd.Timestamp(r.end) <= as_of:
        return r
    cfg = clean.cfg or IngestConfig()
    cut = clean.present.index <= as_of
    n = int(cut.sum())
    if n == 0:
        return r.model_copy(update={"asset_blocked": True, "n_rows_grid": 0, "end": None})
    present = clean.present.to_numpy()[cut]
    missing_frac = 1.0 - float(present.mean())
    chans: dict[str, ChannelDQ] = {}
    for c, d in r.channels.items():
        ok = clean.valid[c].to_numpy()[cut]
        vf = float(ok.mean())
        chans[c] = d.model_copy(
            update={
                "n_samples": n,
                "n_missing": int((~ok).sum()),
                "valid_fraction": vf,
                "usable": vf >= cfg.min_channel_valid,
                "stuck_segments": _clip_segments(d.stuck_segments, as_of),
                "dropout_segments": _clip_segments(d.dropout_segments, as_of),
            }
        )
    erratic = _clip_segments(r.erratic_segments, as_of)
    return r.model_copy(
        update={
            "n_rows_grid": n,
            "end": clean.present.index[cut][-1].to_pydatetime(),
            "asset_blocked": _gate_blocked(chans, clean.spec, n, missing_frac, cfg),
            "channels": chans,
            "erratic_segments": erratic,
            "sensor_advisories": _advisories(chans, erratic, cfg),
        }
    )


def _erratic_vibration(
    frame: pd.DataFrame,
    stopped: np.ndarray,
    limits: dict[str, tuple[float, float]],
    cfg: IngestConfig,
) -> np.ndarray:
    """Hours in which the vibration sensor itself looks faulty (loose mount, cable fault).

    A loose accelerometer reads high *and erratic*: the hour-to-hour change of overall vibration
    jumps far above its usual size, while genuine wear moves the level smoothly. The test, all of
    it causal:

    * roughness = median of the last ``erratic_window_h`` absolute hour-to-hour log changes of
      ``vib_rms`` while running, skipping hours where load or speed moved by more than 10% (a
      regime change is not a sensor fault);
    * erratic when roughness exceeds ``erratic_ratio`` x its own trailing median (the asset's
      normal roughness) and at least ``erratic_floor``; the flag is held for ``erratic_window_h``
      hours after it was last raised, so a flickering test does not let readings through;
    * a level that doubles in one hour at steady load is masked on that hour only (isolated
      spikes are removed first; they belong to the spike check).

    Every vibration channel is masked in a flagged hour. A sensor that steps cleanly to a new,
    steady level is indistinguishable from a real step change and is not caught here.
    """
    n = len(frame)
    if "vib_rms" not in frame.columns or n < 3:
        return np.zeros(n, dtype=bool)
    lo, hi = limits.get("vib_rms", (0.0, np.inf))
    v = frame["vib_rms"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = np.isnan(v) | (v <= max(lo, 0.0)) | (v > hi) | stopped
        v = np.where(bad, np.nan, v)
        # isolated spikes are the spike check's business, not evidence of a loose sensor
        med = (
            pd.Series(v)
            .shift(1)
            .rolling(cfg.spike_history, min_periods=cfg.spike_min_history)
            .median()
            .to_numpy()
        )
        v = np.where(v > cfg.spike_ratio * med, np.nan, v)
        lv = pd.Series(np.log(v))
        d = (lv - lv.shift(1)).abs()
        steady = pd.Series(True, index=d.index)
        for c in ("load_pct", "speed_rpm"):
            x = frame[c].to_numpy(dtype=float)
            lx = pd.Series(np.log(np.where(x > 0, x, np.nan)))
            steady &= ((lx - lx.shift(1)).abs() <= 0.1).fillna(False)
    d = d.where(steady)
    w = cfg.erratic_window_h
    rough = d.rolling(w, min_periods=3).median()
    ref = rough.shift(w).rolling(cfg.erratic_reference_h, min_periods=48).median()
    thr = np.maximum(cfg.erratic_ratio * ref.fillna(0.0), cfg.erratic_floor)
    held = (rough > thr).astype(float).rolling(w, min_periods=1).max().to_numpy() > 0
    jump = (d > np.log(2.0)).to_numpy()  # that hour only: a lone jump is not yet a faulty sensor
    return (held | jump) & ~stopped


def _stopped_mask(frame: pd.DataFrame, spec: AssetSpec, cfg: IngestConfig) -> np.ndarray:
    """Machine appears stopped (so frozen zeros are legitimate). Unknown speed => not stopped."""
    speed = frame["speed_rpm"].to_numpy(dtype=float)
    load = frame["load_pct"].to_numpy(dtype=float)
    lo = cfg.stopped_speed_frac * spec.nominal_rpm
    with np.errstate(invalid="ignore"):
        return (speed < lo) | (load < 2.0)


def _blocked(
    spec: AssetSpec,
    cfg: IngestConfig,
    checks: list[CheckResult],
    maintenance: list[MaintenanceEvent],
    n_in: int,
) -> CleanAsset:
    report = DataQualityReport(
        asset_id=spec.asset_id,
        n_rows_in=n_in,
        n_rows_grid=0,
        start=None,
        end=None,
        overall=DQStatus.FAIL,
        asset_blocked=True,
        clock_drift_s_per_day=0.0,
        n_duplicates=0,
        n_backwards=0,
        n_gap_samples=0,
        longest_gap_hours=0.0,
        checks=checks,
        channels={},
    )
    empty = pd.DataFrame(index=pd.DatetimeIndex([]))
    return CleanAsset(
        spec=spec,
        frame=empty,
        valid=empty.astype(bool),
        present=pd.Series(dtype=bool),
        report=report,
        maintenance=maintenance,
    )
