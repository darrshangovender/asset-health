"""One rule for time zones: everything inside the package is naive UTC.

Plant exports arrive as naive local wall-clock time, as UTC with a ``Z`` suffix, or as local time
with an offset that changes at daylight saving. Mixing them silently shifts events by hours or
crashes a comparison (``tz-naive`` vs ``tz-aware``). Every timestamp that crosses into the
package goes through these helpers:

* a tz-aware value (or a string with an offset) is converted to UTC and the zone dropped;
* a naive value is taken to be UTC already (the adapter contract in ``docs/ADAPTERS.md``).

Naive *local* wall-clock time with daylight saving cannot be repaired here (the autumn hour is
ambiguous); an adapter must attach the offset or convert to UTC before handing data over.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def to_utc_naive_series(values) -> pd.Series:
    """Parse a column of timestamps (strings, datetimes, mixed offsets) to naive UTC ``ns``.

    Unparseable values become ``NaT``; callers decide what to do with them.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is None:
        return s.astype("datetime64[ns]")
    try:  # fast path: ISO 8601 strings, with or without offsets
        ts = pd.to_datetime(s, utc=True, format="ISO8601")
    except (ValueError, TypeError):
        ts = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    return ts.dt.tz_localize(None).astype("datetime64[ns]")


def to_utc_naive(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """One timestamp to naive UTC (see module docstring)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.as_unit("ns")


def to_utc_naive_datetime(value: datetime) -> datetime:
    """Same as ``to_utc_naive`` for a plain ``datetime`` (used by pydantic validators)."""
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value
