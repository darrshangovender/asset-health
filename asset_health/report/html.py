"""Single-file HTML report: no JavaScript, no external requests, charts are inline SVG."""

from __future__ import annotations

from html import escape

import pandas as pd

import asset_health
from asset_health.analysis import AssetAnalysis, FleetAnalysis
from asset_health.models import SYNTHETIC_NOTICE, DQStatus, Severity
from asset_health.report.svg import Band, Line, VLine, downsample, line_chart, sparkline

CSS = """
:root{--bg:#fbfbf9;--ink:#1c2430;--mut:#5b6573;--line:#d9dcd5;--card:#ffffff;--s1:#2457c5;--s2:#c2410c;
--band:rgba(194,65,12,.12);--band2:rgba(36,87,197,.08);--thr:#8a5a00;--mark:#b42318;--ok:#1b6b3a;--warn:#9a5b00;
--urg:#b42318;--crit:#7a1111;--grid:#e6e8e2}
@media (prefers-color-scheme:dark){:root{--bg:#14181f;--ink:#e7eaee;--mut:#9aa4b2;--line:#2c3440;--card:#1b212b;
--s1:#7aa2ff;--s2:#ffa56b;--band:rgba(255,165,107,.16);--band2:rgba(122,162,255,.12);--thr:#e0b25a;--mark:#ff7b72;
--ok:#5fd08a;--warn:#e0b25a;--urg:#ff7b72;--crit:#ff9b94;--grid:#262d38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1020px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:16px;margin:0}.mut{color:var(--mut)}.banner{border:1px solid var(--warn);background:var(--card);padding:10px 14px;border-radius:8px;margin:14px 0;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--mut)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
.chip{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid currentColor}
.sev-normal{color:var(--ok)}.sev-watch{color:var(--warn)}.sev-warning{color:var(--warn)}.sev-urgent{color:var(--urg)}.sev-critical{color:var(--crit);background:rgba(180,35,24,.1)}
.dq-pass{color:var(--ok)}.dq-warn{color:var(--warn)}.dq-fail{color:var(--urg)}
svg.chart{width:100%;height:auto;display:block}.s1{stroke:var(--s1)}.s2{stroke:var(--s2)}.band{fill:var(--band)}.band2{fill:var(--band2)}
.grid{stroke:var(--grid);stroke-width:1}.thr{stroke:var(--thr);stroke-dasharray:5 4;stroke-width:1.2}.mark{stroke:var(--mark);stroke-width:1.4}
.mark2{stroke:var(--mut);stroke-width:1;stroke-dasharray:2 3}.tick{fill:var(--mut);font-size:10.5px}.axis{fill:var(--mut);font-size:10.5px}.thr-l{fill:var(--thr)}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(228px,1fr));gap:10px}.mini{border:1px solid var(--line);border-radius:8px;padding:6px 8px;background:var(--card);font-size:12.5px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;font-size:12px}
.legend{font-size:12.5px;color:var(--mut);margin:2px 0 8px}.legend i{display:inline-block;width:18px;height:0;border-top:2px solid var(--mut);vertical-align:middle;margin:0 4px 0 10px}.legend i.s1{border-color:var(--s1)}.legend i.thr{border-color:var(--thr);border-top-style:dashed}.legend i.mark{border-color:var(--mark)}.legend i.mark2{border-top-style:dotted}
details summary{cursor:pointer;color:var(--mut);font-size:13px;margin-top:8px}.wo{border-left:3px solid var(--warn);padding-left:12px;margin-top:10px}
footer{margin-top:40px;color:var(--mut);font-size:13px}
"""


def _chip(sev: Severity) -> str:
    return f'<span class="chip sev-{sev.value}">{sev.value.upper()}</span>'


def _dq_chip(s: DQStatus, label: str = "") -> str:
    return f'<span class="chip dq-{s.value}">{label}{s.value.upper()}</span>'


def _fmt_days(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def _asset_chart(a: AssetAnalysis, x_range, truth_events) -> str:
    fused = downsample(a.run.frame["fused"].loc[: x_range[1]], "6h")
    bands = [
        Band(
            pd.Timestamp(tr.episode.opened_at),
            pd.Timestamp(tr.episode.closed_at) if tr.episode.closed_at else x_range[1],
            "band",
            f"alert {pd.Timestamp(tr.episode.opened_at):%Y-%m-%d} ({tr.episode.peak_severity.value})",
        )
        for tr in a.traces
    ]
    vlines = [VLine(t, lab, cls) for t, lab, cls in truth_events]
    vlines += [VLine(t, "planned service", "mark2") for t in a.run.services if t <= x_range[1]]
    return line_chart(
        [
            Line(
                list(fused.index),
                [float(v) if v == v else float("nan") for v in fused.to_numpy()],
                "s1",
            )
        ],
        x_range,
        (0.0, max(2.0, float(fused.max()) * 1.05 if fused.notna().any() else 2.0)),
        bands=bands,
        vlines=vlines,
        hlines=[(a.threshold, f"alert threshold {a.threshold:.2f}")],
        y_label="fused score",
        title=f"{a.run.spec.name}: fused detection score",
    )


def _hi_chart(a: AssetAnalysis, x_range, level: float | None) -> str:
    hi = downsample(a.hi.loc[: x_range[1]], "12h")
    return line_chart(
        [Line(list(hi.index), [float(v) if v == v else float("nan") for v in hi.to_numpy()], "s2")],
        x_range,
        (0.0, max(level or 0.0, float(hi.max()) if hi.notna().any() else 1.0) * 1.1),
        hlines=[(level, "typical failure level")] if level else [],
        height=150,
        y_label="health index (sigma)",
        title=f"{a.run.spec.name}: health index",
    )


def _truth_events(fa: FleetAnalysis, asset_id: str) -> list[tuple[pd.Timestamp, str, str]]:
    if fa.truth is None:
        return []
    out = []
    for f in fa.truth.failures:
        if f.asset_id == asset_id:
            out.append(
                (pd.Timestamp(f.onset), f"ground truth: {f.mode.value} degradation starts", "mark2")
            )
            out.append((pd.Timestamp(f.failure), f"ground truth: {f.mode.value} failure", "mark"))
    return out


def _work_order_html(a: AssetAnalysis) -> str:
    if not a.work_orders:
        return ""
    wo = a.work_orders[0]
    ev = "".join(
        f"<tr><td>{escape(e.feature)}</td><td class='n'>{e.observed:.3g} {escape(e.unit)}</td>"
        f"<td class='n'>{e.expected:.3g} {escape(e.unit)}</td><td class='n'>{e.z_score:+.1f}</td>"
        f"<td class='mut'>{escape(e.description)}</td></tr>"
        for e in wo.evidence
    )
    r = wo.rul
    rul = (
        f"{_fmt_days(r.p50_days)} d (80% interval {_fmt_days(r.p10_days)} to {_fmt_days(r.p90_days)} d)"
        if r and r.p50_days is not None
        else (
            f"not available ({escape(r.status.value)}: {escape(r.note)})" if r else "not available"
        )
    )
    safe = f"{wo.latest_safe_date:%Y-%m-%d}" if wo.latest_safe_date else "not set"
    alt = ", ".join(h.value for h in wo.alternative_hypotheses) or "none"
    return (
        f"<div class='wo'><b>Work order {escape(wo.work_order_id)}</b> {_chip(wo.severity)}"
        f"<p>Hypothesis: <b>{escape(wo.failure_mode_hypothesis.value)}</b> (alternatives: {escape(alt)}).<br>"
        f"RUL: {rul}. Latest safe date: <b>{safe}</b>.<br>Action: {escape(wo.recommended_action)}</p>"
        f"<table><tr><th>Evidence</th><th class='n'>Observed</th><th class='n'>Expected</th>"
        f"<th class='n'>z</th><th></th></tr>{ev}</table>"
        f"<p>{escape(wo.narrative)} <span class='mut'>[narrative: {wo.narrative_source}]</span></p>"
        + (
            f"<p class='mut'>Data quality: {escape(wo.data_quality_note)}</p>"
            if wo.data_quality_note
            else ""
        )
        + f"<details><summary>structured work order (JSON)</summary><pre>{escape(wo.model_dump_json(indent=1))}</pre></details></div>"
    )


def _history(a: AssetAnalysis, fa: FleetAnalysis) -> str:
    if not a.traces:
        return ""
    fails = [f for f in (fa.truth.failures if fa.truth else []) if f.asset_id == a.asset_id]
    rows = ""
    for tr in a.traces:
        e = tr.episode
        lead = ""
        for f in fails:
            if pd.Timestamp(f.onset) <= pd.Timestamp(e.opened_at) < pd.Timestamp(f.failure):
                lead = f"{(pd.Timestamp(f.failure) - pd.Timestamp(e.opened_at)).total_seconds() / 86400:.1f} d before the injected failure"
        closed = f"{pd.Timestamp(e.closed_at):%Y-%m-%d %H:%M}" if e.closed_at else "open"
        rows += (
            f"<tr><td>{pd.Timestamp(e.opened_at):%Y-%m-%d %H:%M}</td><td>{closed}</td>"
            f"<td>{_chip(e.peak_severity)}</td><td>{escape(e.hypothesis.value)}</td>"
            f"<td class='n'>{e.peak_score:.2f}</td><td>{escape(e.closed_reason or '')}</td><td class='mut'>{lead}</td></tr>"
        )
    return (
        "<table><tr><th>Opened</th><th>Closed</th><th>Peak</th><th>Hypothesis</th>"
        f"<th class='n'>Peak score</th><th>Closed because</th><th>Ground truth</th></tr>{rows}</table>"
    )


def render_report(fa: FleetAnalysis, title: str = "asset-health fleet report") -> str:
    assets = sorted(fa.assets.values(), key=lambda a: (-a.severity_now.rank, a.asset_id))
    if not fa.assets:
        return "<html><body>No assets.</body></html>"
    starts = [a.run.frame.index[0] for a in assets if len(a.run.frame)]
    x_range = (min(starts), fa.as_of)
    level = fa.rul_model.level_mean if fa.rul_model else None

    overview = ""
    for a in assets:
        wo = a.work_orders[0] if a.work_orders else None
        r = wo.rul if wo else None
        rul = (
            f"{_fmt_days(r.p50_days)} ({_fmt_days(r.p10_days)} to {_fmt_days(r.p90_days)})"
            if r and r.p50_days is not None
            else "-"
        )
        safe = f"{wo.latest_safe_date:%Y-%m-%d}" if wo and wo.latest_safe_date else "-"
        hypothesis = escape(wo.failure_mode_hypothesis.value) if wo else "-"
        overview += (
            f"<tr><td><b>{escape(a.asset_id)}</b> <span class='mut'>{escape(a.run.spec.name)}</span></td>"
            f"<td>{_dq_chip(a.run.clean.report.overall)}</td><td>{_chip(a.severity_now)}</td>"
            f"<td>{hypothesis}</td><td class='n'>{rul}</td><td class='n'>{safe}</td></tr>"
        )

    cards = ""
    quiet = ""
    for a in assets:
        truth_ev = _truth_events(fa, a.asset_id)
        if a.traces or truth_ev:
            wo_html = _work_order_html(a)
            hi_html = ""
            if a.traces and a.hi.notna().any():
                hi_html = _hi_chart(a, x_range, level)
            cards += (
                f"<section class='card'><h3>{escape(a.run.spec.name)} <span class='mut'>({escape(a.asset_id)}, "
                f"{escape(a.run.spec.asset_type.value)})</span> {_chip(a.severity_now)} {_dq_chip(a.run.clean.report.overall, 'data ')}</h3>"
                "<div class='legend'><i class='s1'></i>fused score (6 h median)<i class='thr'></i>alert threshold"
                "<i class='mark'></i>injected failure (synthetic answer key)<i class='mark2'></i>planned service / degradation start</div>"
                f"{_asset_chart(a, x_range, truth_ev)}{hi_html}{wo_html}{_history(a, fa)}</section>"
            )
        else:
            fused = a.run.frame["fused"].loc[: fa.as_of]
            quiet += (
                f"<div class='mini'><b>{escape(a.asset_id)}</b> {_chip(a.severity_now)} "
                f"{_dq_chip(a.run.clean.report.overall)}"
                f"{sparkline(fused, x_range, a.threshold) if fused.notna().any() else '<div class=mut>no decisions (data gate)</div>'}</div>"
            )

    dq = ""
    for a in assets:
        r = a.run.clean.report
        ch = r.channels.values()
        dq += (
            f"<tr><td>{escape(a.asset_id)}</td><td>{_dq_chip(r.overall)}</td>"
            f"<td class='n'>{r.usable_fraction:.1%}</td><td class='n'>{r.n_duplicates}</td>"
            f"<td class='n'>{r.n_backwards}</td><td class='n'>{r.clock_drift_s_per_day:+.0f}</td>"
            f"<td class='n'>{r.longest_gap_hours:.0f}</td><td class='n'>{sum(c.n_out_of_range for c in ch)}</td>"
            f"<td class='n'>{sum(c.n_stuck for c in ch)}</td><td class='n'>{sum(c.n_spikes for c in ch)}</td>"
            f"<td class='mut'>{escape('; '.join(r.sensor_advisories[:2]))}{' ...' if len(r.sensor_advisories) > 2 else ''}</td></tr>"
        )

    notice_text = SYNTHETIC_NOTICE.removeprefix("SYNTHETIC DATA. ")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{escape(title)}</h1>
<div class="mut">asset-health {asset_health.__version__} - data as of {fa.as_of:%Y-%m-%d %H:%M} - {len(fa.assets)} assets</div>
<div class="banner"><b>SYNTHETIC DATA.</b> {escape(notice_text)} Red lines mark failures that the simulator injected; the pipeline never sees them.</div>
<h2>Fleet status</h2>
<table><tr><th>Asset</th><th>Data</th><th>Severity</th><th>Hypothesis</th><th class="n">RUL p50 days (p10 to p90)</th><th class="n">Latest safe date</th></tr>{overview}</table>
<h2>Assets with alerts or injected failures</h2>{cards or "<p class='mut'>No alerts.</p>"}
<h2>Assets with no alerts</h2><div class="grid2">{quiet or "<p class='mut'>None.</p>"}</div>
<h2>Data quality</h2>
<table><tr><th>Asset</th><th>Status</th><th class="n">Valid</th><th class="n">Dup rows</th><th class="n">Backwards</th><th class="n">Drift s/day</th><th class="n">Longest gap h</th><th class="n">Out of range</th><th class="n">Stuck</th><th class="n">Spikes</th><th>Advisories</th></tr>{dq}</table>
<p class="mut">Masked values (out of range, stuck, spikes) are removed before any detector sees them; an asset whose data cannot support a decision is blocked rather than scored.</p>
<h2>Method</h2>
<p>Per-asset healthy baselines are learned on a commissioning window; every channel is compared with its regime-expected value (load, speed; temperature as rise above ambient), so a busy shift is not a fault. CUSUM, an Isolation Forest and an explainable rule layer are fused; an alert opens after a debounce and closes with hysteresis. RUL combines health-index trend extrapolation with quantile regression and a conformal margin fitted on separate seeds. Costs behind the alert thresholds are illustrative.</p>
<footer>Built by <a href="https://github.com/darrshangovender">Darrshan Govender</a> - generated offline, no external requests.</footer>
</main></body></html>"""
