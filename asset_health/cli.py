"""Command line: ``asset-health simulate | check | detect | rul | report``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from asset_health import __version__
from asset_health.models import SYNTHETIC_NOTICE, GroundTruth
from asset_health.timeutil import to_utc_naive

EPILOG = (
    "All data produced by 'simulate' is SYNTHETIC. "
    "See the README for how to point this at a real export."
)


def _default_jobs() -> int:
    n = os.cpu_count() or 1
    return max(1, min(3, n - 1))


def _print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))


def _load_truth(path: Path) -> GroundTruth | None:
    p = path / "ground_truth.json"
    if p.exists():
        return GroundTruth.model_validate_json(p.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------- commands


def cmd_simulate(args: argparse.Namespace) -> int:
    from asset_health.config import SimConfig
    from asset_health.simulate import generate_fleet, write_dataset

    cfg = SimConfig(seed=args.seed, days=args.days, failure_fraction=args.failure_fraction)
    if (
        args.small
    ):  # one asset of each type: a 5-asset fleet that analyses in a fraction of the time
        cfg = cfg.model_copy(update={"fleet": {t: 1 for t in cfg.fleet}})
    t0 = time.time()
    fleet = generate_fleet(cfg, keep_latent=False)
    out = write_dataset(fleet, args.out, args.format)
    modes = ", ".join(f"{f.asset_id}:{f.mode.value}" for f in fleet.truth.failures) or "none"
    print(SYNTHETIC_NOTICE)
    print(
        f"Wrote {len(fleet.specs)} assets x {cfg.days} days ({cfg.sample_interval_s // 60} min "
        f"samples) to {out} in {time.time() - t0:.0f}s."
    )
    print(f"Injected failures (answer key, in ground_truth.json): {modes}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from asset_health.config import IngestConfig
    from asset_health.ingest import ingest_asset, open_source

    src = open_source(args.dataset)
    cfg = IngestConfig(resample=bool(getattr(args, "resample", False)))
    maint = src.maintenance_events()
    rows, reports = [], {}
    for spec in src.assets():
        c = ingest_asset(src.read(spec.asset_id), spec, cfg, maint)
        r = c.report
        reports[spec.asset_id] = json.loads(r.model_dump_json())
        ch = list(r.channels.values())
        rows.append(
            [
                spec.asset_id,
                r.overall.value.upper(),
                "BLOCKED" if r.asset_blocked else "ok",
                f"{r.usable_fraction:.1%}",
                str(r.n_duplicates),
                str(r.n_backwards),
                f"{r.clock_drift_s_per_day:+.0f}",
                f"{r.longest_gap_hours:.0f}",
                "/".join(
                    str(sum(getattr(x, a) for x in ch))
                    for a in ("n_out_of_range", "n_stuck", "n_spikes")
                ),
                str(len(r.sensor_advisories)),
            ]
        )
    _print_table(
        rows,
        [
            "asset",
            "status",
            "gate",
            "valid",
            "dups",
            "back",
            "drift s/d",
            "gap h",
            "oor/stuck/spike",
            "advisories",
        ],
    )
    print("\n'oor/stuck/spike' are readings masked before any detector sees them.")
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=1), encoding="utf-8")
        print(f"Full per-asset reports written to {args.json}")
    return 0


def _analyse(args: argparse.Namespace):
    from asset_health.analysis import analyse_source
    from asset_health.ingest import open_source

    src = open_source(args.dataset)
    truth = None if getattr(args, "no_truth", False) else _load_truth(Path(args.dataset))
    t0 = time.time()
    jobs = args.jobs if args.jobs else _default_jobs()
    as_of = to_utc_naive(args.as_of) if getattr(args, "as_of", None) else None
    cfg = None
    if getattr(args, "resample", False):
        from asset_health.analysis import AnalysisConfig
        from asset_health.config import IngestConfig, PipelineConfig

        cfg = AnalysisConfig(pipeline=PipelineConfig(ingest=IngestConfig(resample=True)))
    fa = analyse_source(src, cfg=cfg, truth=truth, n_jobs=jobs, as_of=as_of)
    print(
        f"[analysed {len(fa.assets)} assets in {time.time() - t0:.0f}s with {jobs} worker(s)]",
        file=sys.stderr,
    )
    return fa


def cmd_detect(args: argparse.Namespace) -> int:
    fa = _analyse(args)
    rows = []
    for a in sorted(fa.assets.values(), key=lambda x: (-x.severity_now.rank, x.asset_id)):
        first = a.traces[0].episode if a.traces else None
        rows.append(
            [
                a.asset_id,
                a.severity_now.value.upper(),
                str(len(a.traces)),
                f"{first.opened_at:%Y-%m-%d %H:%M}" if first else "-",
                first.hypothesis.value if first else "-",
                "BLOCKED" if a.blocked else f"{a.run.clean.report.overall.value}",
            ]
        )
    _print_table(rows, ["asset", "severity now", "alerts", "first alert", "hypothesis", "data"])
    if args.json:
        eps = [
            json.loads(tr.episode.model_dump_json()) for a in fa.assets.values() for tr in a.traces
        ]
        Path(args.json).write_text(json.dumps(eps, indent=1), encoding="utf-8")
        print(f"Alert episodes written to {args.json}")
    return 0


def cmd_rul(args: argparse.Namespace) -> int:
    fa = _analyse(args)
    wos = fa.all_work_orders()
    if not wos:
        print("No open alerts: no remaining-useful-life estimates are issued for healthy assets.")
        return 0
    rows = []
    for wo in wos:
        r = wo.rul
        if r is not None and r.p50_days is not None:
            est = f"{r.p50_days:.1f} d  [{r.p10_days:.1f} - {r.p90_days:.1f}]"
        else:
            est = f"none ({r.status.value})" if r else "none"
        rows.append(
            [
                wo.asset_id,
                wo.severity.value.upper(),
                wo.failure_mode_hypothesis.value,
                est,
                f"{wo.latest_safe_date:%Y-%m-%d}" if wo.latest_safe_date else "-",
            ]
        )
    _print_table(
        rows, ["asset", "severity", "hypothesis", "RUL p50 [p10 - p90]", "latest safe date"]
    )
    print(
        "\nRUL is always an interval; the nominal 80% interval is widened by a conformal margin "
        "fitted on separate synthetic seeds and over-covers on the benchmark (conservative)."
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps([json.loads(w.model_dump_json()) for w in wos], indent=1), encoding="utf-8"
        )
        print(f"Work orders written to {args.json}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from asset_health.report import render_report

    fa = _analyse(args)
    html = render_report(fa)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(html) / 1024:.0f} KB, self-contained, no external requests)")
    return 0


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asset-health",
        description="Offline predictive maintenance for mining assets (SYNTHETIC data).",
        epilog=EPILOG,
    )
    p.add_argument("--version", action="version", version=f"asset-health {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a seeded SYNTHETIC fleet dataset")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--days", type=int, default=270)
    s.add_argument("--failure-fraction", type=float, default=0.65)
    s.add_argument("--format", choices=("csv", "parquet"), default="csv")
    s.add_argument("--small", action="store_true", help="one asset per type (5 assets)")
    s.add_argument("--out", required=True, help="output directory")
    s.set_defaults(fn=cmd_simulate)

    resample_help = (
        "average a finer-than-hourly export onto the hourly grid instead of blocking the asset"
    )
    c = sub.add_parser("check", help="data-quality report per asset")
    c.add_argument("dataset")
    c.add_argument("--json", help="write full reports to this file")
    c.add_argument("--resample", action="store_true", help=resample_help)
    c.set_defaults(fn=cmd_check)

    for name, fn, help_ in (
        ("detect", cmd_detect, "run detection and list alerts"),
        ("rul", cmd_rul, "remaining useful life (with interval) for open alerts"),
        ("report", cmd_report, "write a self-contained HTML report"),
    ):
        d = sub.add_parser(name, help=help_)
        d.add_argument("dataset")
        d.add_argument("--jobs", type=int, default=0, help="worker processes (default: auto)")
        d.add_argument("--no-truth", action="store_true", help="ignore ground_truth.json")
        d.add_argument("--resample", action="store_true", help=resample_help)
        d.add_argument(
            "--as-of",
            metavar="YYYY-MM-DD",
            help="analyse the moment of this date: nothing after it is used (default: end of data)",
        )
        if name == "report":
            d.add_argument("--out", default="report.html")
        else:
            d.add_argument("--json", help="also write machine-readable output here")
        d.set_defaults(fn=fn)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
