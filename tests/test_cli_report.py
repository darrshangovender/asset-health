from __future__ import annotations

import json
import re

import pandas as pd
import pytest

import asset_health.cli as cli
from asset_health import __version__
from asset_health.models import SYNTHETIC_NOTICE
from asset_health.report import render_report
from asset_health.report.svg import Band, Line, VLine, _nice_ticks, line_chart, sparkline

# ----------------------------------------------------------------------------- report


@pytest.fixture(scope="module")
def html(small_analysis):
    return render_report(small_analysis)


def test_report_is_one_self_contained_file(html):
    assert "<script" not in html and "<link" not in html and "<img" not in html
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", html))
    assert urls <= {"http://www.w3.org/2000/svg", "https://github.com/darrshangovender"}


def test_report_carries_the_synthetic_banner(html):
    assert "SYNTHETIC DATA" in html and SYNTHETIC_NOTICE.split(".")[0] in html


def test_report_lists_every_asset(html, small_analysis):
    for aid in small_analysis.assets:
        assert aid in html


def test_report_shows_charts_and_a_work_order_or_no_alerts_message(html):
    assert html.count("<svg") >= 2
    assert "Fleet status" in html and "Data quality" in html


def test_report_embeds_ground_truth_markers_only_when_truth_is_supplied(small_analysis):
    with_truth = render_report(small_analysis)
    small_analysis.truth, saved = None, small_analysis.truth
    try:
        without = render_report(small_analysis)
    finally:
        small_analysis.truth = saved
    assert "ground truth:" in with_truth and "ground truth:" not in without


def test_report_escapes_html_in_names(small_analysis):
    a = next(iter(small_analysis.assets.values()))
    old = a.run.spec
    a.run.spec = old.model_copy(update={"name": "<b>x</b>&"})
    try:
        out = render_report(small_analysis)
    finally:
        a.run.spec = old
    assert "<b>x</b>" not in out and "&lt;b&gt;x&lt;/b&gt;&amp;" in out


def test_report_supports_dark_mode_without_javascript(html):
    assert "prefers-color-scheme:dark" in html


def test_svg_chart_has_only_finite_coordinates():
    idx = pd.date_range("2025-01-01", periods=50, freq="D")
    svg = line_chart(
        [Line(list(idx), [float("nan") if i % 7 == 0 else float(i) for i in range(50)])],
        (idx[0], idx[-1]),
        bands=[Band(idx[5], idx[9], label="alert")],
        vlines=[VLine(idx[20], "failure")],
        hlines=[(10.0, "thr")],
        y_label="x",
        title="t",
    )
    assert "nan" not in svg.lower() and "inf" not in svg.lower()
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_sparkline_renders_a_compact_chart():
    idx = pd.date_range("2025-01-01", periods=400, freq="h")
    s = pd.Series(range(400), index=idx, dtype=float) / 400
    svg = sparkline(s, (idx[0], idx[-1]), 1.0)
    assert 'viewBox="0 0 220 44"' in svg


def test_nice_ticks_are_ascending_and_inside_the_range():
    t = _nice_ticks(0.0, 7.3)
    assert t == sorted(t) and t[0] >= 0 and t[-1] <= 7.3


# ----------------------------------------------------------------------------- cli


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_unknown_dataset_exits_with_an_error_code(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "nope")]) == 2
    assert "error" in capsys.readouterr().err


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli") / "ds"
    assert cli.main(["simulate", "--seed", "5", "--days", "75", "--small", "--out", str(out)]) == 0
    return out


def test_simulate_writes_a_labelled_synthetic_dataset(dataset, capsys):
    assert (dataset / "manifest.json").exists() and (dataset / "SYNTHETIC_DATA.txt").exists()
    assert len(list((dataset / "assets").glob("*.csv"))) == 5
    gt = json.loads((dataset / "ground_truth.json").read_text(encoding="utf-8"))
    assert gt["synthetic"] is True


def test_check_prints_a_row_per_asset_and_writes_json(dataset, tmp_path, capsys):
    out = tmp_path / "dq.json"
    assert cli.main(["check", str(dataset), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    reports = json.loads(out.read_text(encoding="utf-8"))
    assert set(reports) == {p.stem for p in (dataset / "assets").glob("*.csv")}
    for aid in reports:
        assert aid in text
    assert "oor/stuck/spike" in text


def test_detect_and_rul_commands_format_an_analysis(monkeypatch, small_analysis, capsys, tmp_path):
    monkeypatch.setattr(cli, "_analyse", lambda args: small_analysis)
    j = tmp_path / "eps.json"
    assert cli.main(["detect", "x", "--json", str(j)]) == 0
    out = capsys.readouterr().out
    assert "severity now" in out
    assert isinstance(json.loads(j.read_text(encoding="utf-8")), list)
    assert cli.main(["rul", "x"]) == 0
    out = capsys.readouterr().out
    assert "RUL" in out or "No open alerts" in out


def test_report_command_writes_a_file(monkeypatch, small_analysis, tmp_path, capsys):
    monkeypatch.setattr(cli, "_analyse", lambda args: small_analysis)
    out = tmp_path / "sub" / "r.html"
    assert cli.main(["report", "x", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "self-contained" in capsys.readouterr().out


def test_report_command_runs_the_real_pipeline_on_a_written_dataset(dataset, tmp_path, capsys):
    out = tmp_path / "real.html"
    assert cli.main(["report", str(dataset), "--out", str(out), "--jobs", "1"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in text and "Fleet status" in text


def test_as_of_flag_is_parsed_and_the_report_is_cut_at_that_date(
    monkeypatch, small_analysis, tmp_path
):
    seen = {}

    def fake(args):
        seen["as_of"] = args.as_of
        return small_analysis

    monkeypatch.setattr(cli, "_analyse", fake)
    out = tmp_path / "r.html"
    assert cli.main(["report", "x", "--as-of", "2025-05-01", "--out", str(out)]) == 0
    assert seen["as_of"] == "2025-05-01"


def test_report_charts_stop_at_the_as_of_date(small_fleet, small_runs):
    from asset_health.analysis import analyse_runs

    as_of = pd.Timestamp("2025-04-15")
    fa = analyse_runs(small_runs, as_of=as_of, truth=small_fleet.truth)
    html_text = render_report(fa)
    assert "data as of 2025-04-15" in html_text
    # injected events after the as-of date are not drawn: the report shows the past only
    want = sum(
        (pd.Timestamp(f.onset) <= as_of) + (pd.Timestamp(f.failure) <= as_of)
        for f in small_fleet.truth.failures
    )
    assert html_text.count("ground truth:") == want
