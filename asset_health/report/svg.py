"""Tiny dependency-free SVG chart helpers (inline, theme-aware through CSS variables)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd

PAD_L, PAD_R, PAD_T, PAD_B = 46, 12, 14, 26


@dataclass
class Line:
    x: Sequence[pd.Timestamp]
    y: Sequence[float]
    cls: str = "s1"
    label: str = ""
    width: float = 1.4


@dataclass
class Band:
    start: pd.Timestamp
    end: pd.Timestamp
    cls: str = "band"
    label: str = ""


@dataclass
class VLine:
    at: pd.Timestamp
    label: str
    cls: str = "mark"


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    ticks = []
    v = first
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def downsample(s: pd.Series, freq: str = "6h") -> pd.Series:
    """Median per bucket; NaN buckets stay NaN so gaps are drawn as gaps."""
    return s.resample(freq).median()


def line_chart(
    lines: list[Line],
    x_range: tuple[pd.Timestamp, pd.Timestamp],
    y_range: tuple[float, float] | None = None,
    width: int = 760,
    height: int = 190,
    bands: Sequence[Band] = (),
    vlines: Sequence[VLine] = (),
    hlines: Sequence[tuple[float, str]] = (),
    y_label: str = "",
    title: str = "",
    log_y: bool = False,
    aria: str = "",
    compact: bool = False,
) -> str:
    pad_l, pad_r, pad_t, pad_b = (3, 3, 3, 3) if compact else (PAD_L, PAD_R, PAD_T, PAD_B)
    x0, x1 = x_range[0].value, x_range[1].value
    ys = [v for ln in lines for v in ln.y if v == v and not math.isinf(v)]
    if y_range is None:
        lo, hi = (min(ys), max(ys)) if ys else (0.0, 1.0)
        lo = min(lo, 0.0)
        hi = max(hi, max((h for h, _ in hlines), default=0.0), lo + 1e-6)
        y_range = (lo, hi * 1.06)
    ylo, yhi = y_range

    def sx(t: pd.Timestamp) -> float:
        return pad_l + (t.value - x0) / max(x1 - x0, 1) * (width - pad_l - pad_r)

    def sy(v: float) -> float:
        if log_y:
            v, a, b = (
                math.log10(max(v, 1e-9)),
                math.log10(max(ylo, 1e-9)),
                math.log10(max(yhi, 1e-9)),
            )
        else:
            a, b = ylo, yhi
        return pad_t + (1 - (v - a) / max(b - a, 1e-12)) * (height - pad_t - pad_b)

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(aria or title)}" xmlns="http://www.w3.org/2000/svg">'
    ]
    if title:
        out.append(f"<title>{escape(title)}</title>")
    for b in bands:
        xa, xb = sx(max(b.start, x_range[0])), sx(min(b.end, x_range[1]))
        if xb > xa:
            out.append(
                f'<rect class="{b.cls}" x="{xa:.1f}" y="{pad_t}" width="{xb - xa:.1f}" '
                f'height="{height - pad_t - pad_b}"><title>{escape(b.label)}</title></rect>'
            )
    for v in (
        [] if compact else (_nice_ticks(ylo, yhi) if not log_y else [10**k for k in range(-1, 3)])
    ):
        if not (ylo - 1e-9 <= v <= yhi + 1e-9):
            continue
        y = sy(v)
        out.append(
            f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end">{v:g}</text>'
        )
    months = [] if compact else pd.date_range(x_range[0].normalize(), x_range[1], freq="MS")
    for m in months:
        if m < x_range[0]:
            continue
        x = sx(m)
        out.append(
            f'<line class="grid v" x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{height - pad_b}"/>'
        )
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 8}" text-anchor="middle">{m:%b}</text>'
        )
    for h, lab in hlines:
        y = sy(h)
        out.append(
            f'<line class="thr" x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick thr-l" x="{width - pad_r - 2}" y="{y - 3:.1f}" text-anchor="end">{escape(lab)}</text>'
            if lab
            else ""
        )
    for v in vlines:
        if x_range[0] <= v.at <= x_range[1]:
            x = sx(v.at)
            out.append(
                f'<line class="{v.cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{height - pad_b}"><title>{escape(v.label)}</title></line>'
            )
    for ln in lines:
        d: list[str] = []
        pen = False
        for t, v in zip(ln.x, ln.y, strict=True):
            if v != v or math.isinf(v):
                pen = False
                continue
            d.append(f"{'L' if pen else 'M'}{sx(t):.1f},{sy(v):.1f}")
            pen = True
        if d:
            out.append(
                f'<path class="{ln.cls}" d="{" ".join(d)}" fill="none" stroke-width="{ln.width}"/>'
            )
    if y_label and not compact:
        out.append(
            f'<text class="axis" x="10" y="{pad_t + 8}" transform="rotate(-90 10 {pad_t + 8})" text-anchor="end">{escape(y_label)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def sparkline(
    s: pd.Series,
    x_range: tuple[pd.Timestamp, pd.Timestamp],
    thr: float | None,
    width: int = 220,
    height: int = 44,
) -> str:
    s = downsample(s, "12h")
    return line_chart(
        [
            Line(
                list(s.index),
                [float(v) if v == v else float("nan") for v in s.to_numpy()],
                "s1",
                width=1.1,
            )
        ],
        x_range,
        (0.0, max(1.6, float(np.nanmax(s.to_numpy())) if s.notna().any() else 1.6)),
        width=width,
        height=height,
        hlines=[(thr, "")] if thr else (),
        aria="fused score sparkline",
        compact=True,
    )
